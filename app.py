"""Streamlit demonstration interface for the trained ReviewGuard model."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import METADATA_PATH
from src.explain import influential_words
from src.predict import explain_prediction, load_model, predict_review

st.set_page_config(page_title="ReviewGuard", page_icon="🛡️", layout="centered")


@st.cache_data
def load_metadata() -> dict:
    """Read optional training metadata without requiring a live training run."""
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}


st.title("🛡️ ReviewGuard")
st.caption("NLP-powered deceptive review detection")
st.write("Paste a hotel review to estimate whether its language resembles reviews labelled as deceptive or truthful in the training corpus.")
review = st.text_area("Paste a hotel review", height=180, placeholder="Example: The room was spotless and the staff were very helpful...")

if st.button("Analyze Review", type="primary", use_container_width=True):
    try:
        result = predict_review(review)
        deceptive = result["deceptive_probability"]
        truthful = result["truthful_probability"]
        if result["label"] == "deceptive":
            st.error("Prediction: DECEPTIVE")
        else:
            st.success("Prediction: TRUTHFUL")
        left, right = st.columns(2)
        left.metric("Deceptive probability", f"{deceptive:.1%}")
        right.metric("Truthful probability", f"{truthful:.1%}")
        st.progress(int(deceptive * 100), text=f"Deceptive probability: {deceptive:.1%}")
        contributions = explain_prediction(review)
        st.subheader("Terms influencing this prediction")
        st.dataframe(contributions.style.format({"contribution": "{:+.3f}"}), use_container_width=True, hide_index=True)
        st.caption("These terms indicate model influence, not proof that a review is fake.")
    except (FileNotFoundError, ValueError) as error:
        st.warning(str(error))

with st.expander("About the model"):
    st.write("ReviewGuard turns words and phrases into TF-IDF features, then uses Logistic Regression to estimate the probability of each class. The vectorizer and classifier are saved together after training.")

with st.expander("Model performance"):
    metadata = load_metadata()
    metrics = metadata.get("test_metrics")
    if metrics:
        st.dataframe(pd.DataFrame([metrics]).drop(columns="model", errors="ignore").style.format("{:.3f}"), use_container_width=True, hide_index=True)
    else:
        st.info("Train the model to display actual evaluation metrics here.")

with st.expander("Linguistic insights"):
    try:
        terms = influential_words(load_model(), 10)
        st.dataframe(terms, use_container_width=True, hide_index=True)
        st.caption("These are statistically influential features for this trained classifier, not definitive indicators of deception.")
    except FileNotFoundError:
        st.info("Train the model to display learned terms.")

with st.expander("Disclaimer"):
    st.warning("ReviewGuard is an educational NLP project. Its predictions are probabilistic and should not be treated as definitive evidence that a review is fake.")
