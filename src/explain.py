"""Global and local explanations for the logistic-regression pipeline."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import FIGURES_DIR, RESULTS_DIR


def influential_words(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Rank terms by learned logistic-regression association with each class."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    table = pd.DataFrame({"feature": vectorizer.get_feature_names_out(), "coefficient": classifier.coef_[0]})
    truthful = table.nsmallest(top_n, "coefficient").sort_values("coefficient")
    deceptive = table.nlargest(top_n, "coefficient").sort_values("coefficient", ascending=False)
    result = pd.concat([
        truthful.assign(associated_class="truthful"),
        deceptive.assign(associated_class="deceptive"),
    ], ignore_index=True)
    result.insert(0, "rank", result.groupby("associated_class").cumcount() + 1)
    return result[["rank", "feature", "coefficient", "associated_class"]]


def save_influential_words(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Save the global coefficient table and an explanatory visualisation."""
    result = influential_words(pipeline, top_n)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_DIR / "influential_words.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 7))
    chart_data = result.sort_values("coefficient")
    colors = chart_data["associated_class"].map({"truthful": "#4c78a8", "deceptive": "#e45756"})
    axis.barh(chart_data["feature"], chart_data["coefficient"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(title="Statistically influential TF-IDF features", xlabel="Logistic regression coefficient")
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "influential_words.png", dpi=160)
    plt.close(figure)
    return result


def local_term_contributions(pipeline: Pipeline, review_text: str, top_n: int = 10) -> pd.DataFrame:
    """Return non-zero TF-IDF terms with the strongest local linear influence."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    matrix = vectorizer.transform([review_text])
    indices = matrix.nonzero()[1]
    values = matrix.data
    features = vectorizer.get_feature_names_out()[indices]
    contributions = values * classifier.coef_[0][indices]
    table = pd.DataFrame({"feature": features, "contribution": contributions})
    return table.reindex(table["contribution"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
