"""Factories for the text representations used by ReviewGuard."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DEFAULT_TFIDF_PARAMS


def make_tfidf_vectorizer(**overrides: object) -> TfidfVectorizer:
    """Create a configurable unigram-and-bigram TF-IDF vectorizer."""
    parameters = {**DEFAULT_TFIDF_PARAMS, **overrides}
    return TfidfVectorizer(**parameters)
