"""Model evaluation, plots, and artifact writers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from .config import FIGURES_DIR, RESULTS_DIR, TARGET_NAMES


def evaluate_predictions(y_true: pd.Series, y_pred: Any) -> tuple[dict[str, float], dict[str, Any]]:
    """Calculate the four required binary classification metrics and report."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    report = classification_report(y_true, y_pred, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    return metrics, report


def save_confusion_matrix(y_true: pd.Series, y_pred: Any, model_name: str, output_dir: Path = FIGURES_DIR) -> Path:
    """Save a readable labelled confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=axis)
    axis.set(title=f"{model_name} confusion matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_model_comparison(comparison: pd.DataFrame, output_dir: Path = RESULTS_DIR) -> Path:
    """Write model scores and a comparison chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    comparison.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot.bar(ax=axis)
    axis.set(title="Model comparison", ylabel="Score", ylim=(0, 1))
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)
    return output_dir / "model_comparison.csv"


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save JSON with values serialized in a human-readable form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
