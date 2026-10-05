"""Descriptive linguistic analysis; associations are not causal evidence."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, RESULTS_DIR

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|our|ours|us)\b", re.IGNORECASE)


def add_linguistic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight style and VADER sentiment indicators to a review frame."""
    result = frame.copy()
    texts = result["review_text"].fillna("")
    result["character_count"] = texts.str.len()
    result["word_count"] = texts.str.findall(r"\b\w+\b").str.len()
    result["sentence_count"] = texts.str.count(r"[.!?]+") .clip(lower=1)
    result["avg_sentence_length"] = result["word_count"] / result["sentence_count"]
    result["exclamation_count"] = texts.str.count("!")
    result["question_count"] = texts.str.count(r"\?")
    result["first_person_count"] = texts.map(lambda value: len(FIRST_PERSON.findall(value)))
    result["uppercase_ratio"] = texts.map(_uppercase_ratio)
    scores = _vader_scores(texts)
    for key, values in scores.items():
        result[f"sentiment_{key}"] = values
    return result


def _uppercase_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    return sum(character.isupper() for character in letters) / len(letters) if letters else 0.0


def _vader_scores(texts: pd.Series) -> dict[str, list[float]]:
    """Use VADER if locally available; otherwise provide missing values gracefully."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        all_scores = [analyzer.polarity_scores(text) for text in texts]
        return {key: [score[key] for score in all_scores] for key in ("neg", "neu", "pos", "compound")}
    except (ImportError, LookupError):
        return {key: [float("nan")] * len(texts) for key in ("neg", "neu", "pos", "compound")}


def save_linguistic_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Save class means and selected visual comparisons."""
    enriched = add_linguistic_features(frame)
    numeric = [column for column in enriched.columns if column in {
        "character_count", "word_count", "sentence_count", "avg_sentence_length",
        "exclamation_count", "question_count", "first_person_count", "uppercase_ratio", "sentiment_compound",
    }]
    summary = enriched.groupby("label")[numeric].mean().T.rename(columns={0: "truthful", 1: "deceptive"})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "linguistic_summary.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for feature in ("word_count", "sentiment_compound"):
        if enriched[feature].notna().any():
            figure, axis = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=enriched, x="label", y=feature, ax=axis)
            axis.set_xticks([0, 1], ["Truthful", "Deceptive"])
            axis.set(title=f"{feature.replace('_', ' ').title()} by class")
            figure.tight_layout()
            figure.savefig(FIGURES_DIR / f"{feature}_by_class.png", dpi=160)
            plt.close(figure)
    return summary
