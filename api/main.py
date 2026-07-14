from typing import List
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model

MAX_LEN = 128
ARTIFACTS_DIR = "transformer_artifacts_fixed"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(f"{ARTIFACTS_DIR}/bert_tokenizer")

print("Loading DistilBERT...")
bert_model = AutoModel.from_pretrained(f"{ARTIFACTS_DIR}/bert_base")
bert_model.eval()
bert_model.to(device)

print("Loading Keras classifier...")
classifier = load_model(f"{ARTIFACTS_DIR}/transformer_classifier.keras")

print(f"All models loaded. Running on: {device}")
app = FastAPI(title="IMDB Sentiment API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    review: str


class SentimentResponse(BaseModel):
    sentiment: str       
    confidence: float    


class BatchReviewRequest(BaseModel):
    reviews: List[str]


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]


def predict_sentiment(text: str):
    
    with torch.no_grad():
        encoded = tokenizer(
            [text], padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        ).to(device)
        output = bert_model(**encoded)
        cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()

    
    prob = float(classifier.predict(cls_embedding, verbose=0)[0][0])
    label = "positive" if prob > 0.5 else "negative"
    confidence = prob if label == "positive" else 1 - prob

    return label, confidence


@app.get("/")
def root():
    """Landing route -- quick sanity check + pointer to the docs."""
    return {
        "message": "IMDB Sentiment API is running",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict", "/predict/batch"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.get("/model-info")
def model_info():
    """Basic details about what's actually being served -- useful for
    your report/viva to show exactly which model/config is live."""
    return {
        "approach": "frozen DistilBERT embeddings + Keras classifier head",
        "base_model": "distilbert-base-uncased",
        "max_length": MAX_LEN,
        "device": device,
        "classifier_input_dim": classifier.input_shape[-1],
    }


@app.post("/predict", response_model=SentimentResponse)
def predict(request: ReviewRequest):
    sentiment, confidence = predict_sentiment(request.review)
    return SentimentResponse(sentiment=sentiment, confidence=confidence)


@app.post("/predict/batch", response_model=BatchSentimentResponse)
def predict_batch(request: BatchReviewRequest):
    """Predict sentiment for multiple reviews in one call, instead of
    sending them one at a time."""
    results = []
    for review in request.reviews:
        sentiment, confidence = predict_sentiment(review)
        results.append(SentimentResponse(sentiment=sentiment, confidence=confidence))
    return BatchSentimentResponse(results=results)
