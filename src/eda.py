"""Reusable exploratory plots for the review corpus."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR
from .linguistic_analysis import add_linguistic_features


def save_eda_figures(frame: pd.DataFrame, hotel_column: str | None = None) -> None:
    """Save class, length, and optional hotel-distribution charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = frame["label"].map({0: "Truthful", 1: "Deceptive"})
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.countplot(x=labels, hue=labels, order=["Truthful", "Deceptive"], legend=False, ax=axis, palette="Set2")
    axis.set(title="Class distribution", xlabel="Review class", ylabel="Reviews")
    figure.tight_layout(); figure.savefig(FIGURES_DIR / "class_distribution.png", dpi=160); plt.close(figure)
    enriched = add_linguistic_features(frame)
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(data=enriched, x="word_count", hue="label", bins=30, element="step", stat="density", common_norm=False, ax=axis)
    axis.set(title="Review-length distribution", xlabel="Words per review")
    figure.tight_layout(); figure.savefig(FIGURES_DIR / "review_length_distribution.png", dpi=160); plt.close(figure)
    if hotel_column and hotel_column in frame:
        counts = frame.groupby(hotel_column).size().sort_values(ascending=False).head(20)
        figure, axis = plt.subplots(figsize=(9, 5))
        counts.sort_values().plot.barh(ax=axis)
        axis.set(title="Reviews per hotel (top 20)", xlabel="Reviews", ylabel="Hotel")
        figure.tight_layout(); figure.savefig(FIGURES_DIR / "hotel_distribution.png", dpi=160); plt.close(figure)
