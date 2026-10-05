"""Small prediction API backed by the persisted ReviewGuard pipeline."""
from __future__ import annotations

from functools import lru_cache

import joblib

from .config import MODEL_PATH
from .explain import local_term_contributions
from .preprocessing import clean_review_text


@lru_cache(maxsize=1)
def load_model():
    """Load the persisted model once per process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)


def predict_review(review_text: str) -> dict[str, float | str]:
    """Classify a review and return both class probabilities."""
    if not isinstance(review_text, str) or not review_text.strip():
        raise ValueError("Please provide a non-empty review.")
    model = load_model()
    cleaned = clean_review_text(review_text)
    deceptive_probability = float(model.predict_proba([cleaned])[0, 1])
    label = "deceptive" if deceptive_probability >= 0.5 else "truthful"
    return {
        "label": label, "probability": deceptive_probability if label == "deceptive" else 1 - deceptive_probability,
        "truthful_probability": 1 - deceptive_probability, "deceptive_probability": deceptive_probability,
    }


def explain_prediction(review_text: str, top_n: int = 10):
    """Return terms with largest local linear contributions."""
    return local_term_contributions(load_model(), clean_review_text(review_text), top_n)
