"""
streamlit_app.py
Simple Streamlit frontend to test the FastAPI sentiment API.

Run:
    pip install streamlit requests
    streamlit run streamlit_app.py

Make sure your FastAPI server is already running separately:
    python -m uvicorn main:app --reload
"""

import requests
import streamlit as st

st.set_page_config(page_title="IMDB Sentiment API — Test Console", page_icon="🎬")

st.title("🎬 IMDB Sentiment API")
st.caption("Test console for the FastAPI /predict endpoint")

# --------------------------------------------------------------------------
# API URL
# --------------------------------------------------------------------------
api_url = st.text_input("API base URL", value="http://127.0.0.1:8000")

# --------------------------------------------------------------------------
# Example buttons -- fill the text box with one click
# --------------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = (
        "This movie was an absolute masterpiece. The acting, the direction, "
        "everything came together beautifully."
    )

col1, col2 = st.columns(2)
with col1:
    if st.button("Positive example"):
        st.session_state.review_text = (
            "This movie was an absolute masterpiece. The acting, the direction, "
            "everything came together beautifully."
        )
with col2:
    if st.button("Negative example"):
        st.session_state.review_text = (
            "Total waste of time. The plot made no sense and the acting was "
            "wooden throughout."
        )

review = st.text_area("Movie review", key="review_text", height=150)

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if st.button("Predict", type="primary"):
    if not review.strip():
        st.warning("Enter a review first.")
    else:
        with st.spinner("Contacting API..."):
            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/predict",
                    json={"review": review},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                sentiment = data["sentiment"]
                confidence = data["confidence"]

                if sentiment == "positive":
                    st.success(f"**{sentiment.upper()}**")
                else:
                    st.error(f"**{sentiment.upper()}**")

                st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the API. Make sure it's running "
                    "(`python -m uvicorn main:app --reload`) and the URL above is correct."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API returned an error: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# --------------------------------------------------------------------------
# Batch prediction -- optional, uses the /predict/batch route
# --------------------------------------------------------------------------
with st.expander("Batch predict (multiple reviews at once)"):
    batch_text = st.text_area(
        "One review per line", height=150,
        placeholder="Review one...\nReview two...\nReview three..."
    )
    if st.button("Predict batch"):
        reviews = [line.strip() for line in batch_text.split("\n") if line.strip()]
        if not reviews:
            st.warning("Enter at least one review, one per line.")
        else:
            with st.spinner(f"Predicting {len(reviews)} reviews..."):
                try:
                    response = requests.post(
                        f"{api_url.rstrip('/')}/predict/batch",
                        json={"reviews": reviews},
                        timeout=60,
                    )
                    response.raise_for_status()
                    results = response.json()["results"]

                    for review_text, result in zip(reviews, results):
                        label = result["sentiment"]
                        conf = result["confidence"]
                        icon = "🟢" if label == "positive" else "🔴"
                        st.write(f"{icon} **{label}** ({conf*100:.1f}%) — {review_text[:80]}")

                except requests.exceptions.ConnectionError:
                    st.error("Couldn't reach the API. Is it running?")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")