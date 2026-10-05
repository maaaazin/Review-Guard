"""Reproducible ReviewGuard training pipeline.

Run from the repository root with: python -m src.train
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import (METADATA_PATH, MODEL_PATH, RANDOM_STATE, RESULTS_DIR, TEST_SIZE,
                     ensure_project_directories)
from .data_loader import DatasetSchemaError, load_reviews
from .eda import save_eda_figures
from .evaluate import evaluate_predictions, save_confusion_matrix, save_json, save_model_comparison
from .explain import save_influential_words
from .features import make_tfidf_vectorizer
from .linguistic_analysis import save_linguistic_summary


def make_pipeline(model: object, **tfidf_options: object) -> Pipeline:
    """Build a leak-safe vectorizer-plus-classifier pipeline."""
    return Pipeline([("tfidf", make_tfidf_vectorizer(**tfidf_options)), ("classifier", model)])


def _cross_validation(pipeline: Pipeline, text: pd.Series, labels: pd.Series) -> dict[str, float]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(pipeline, text, labels, cv=folds, scoring={"accuracy": "accuracy", "f1": "f1"})
    return {
        "cv_accuracy_mean": float(result["test_accuracy"].mean()),
        "cv_accuracy_std": float(result["test_accuracy"].std()),
        "cv_f1_mean": float(result["test_f1"].mean()),
        "cv_f1_std": float(result["test_f1"].std()),
    }


def _tfidf_configuration_experiment(text: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Compare representations with CV on training data only, never the test set."""
    configurations = {
        "unigrams": {"ngram_range": (1, 1)},
        "unigrams_bigrams": {"ngram_range": (1, 2)},
        "unigrams_bigrams_stopwords": {"ngram_range": (1, 2), "stop_words": "english"},
    }
    rows = []
    for name, options in configurations.items():
        pipeline = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), **options)
        scores = _cross_validation(pipeline, text, labels)
        rows.append({"configuration": name, **scores})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "tfidf_configuration_experiment.csv", index=False)
    return result


def _hotel_holdout_experiment(frame: pd.DataFrame, hotel_column: str, pipeline: Pipeline) -> dict[str, float] | None:
    """Measure transfer to unseen hotels when enough distinct hotel groups exist."""
    groups = frame[hotel_column].fillna("unknown").astype(str)
    if groups.nunique() < 2:
        return None
    split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_index, test_index = next(split.split(frame["review_text"], frame["label"], groups=groups))
    train, test = frame.iloc[train_index], frame.iloc[test_index]
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        return None
    pipeline.fit(train["review_text"], train["label"])
    metrics, _ = evaluate_predictions(test["label"], pipeline.predict(test["review_text"]))
    return metrics


def train(args: argparse.Namespace) -> pd.DataFrame:
    """Load, train, evaluate, persist artifacts, and return the model comparison."""
    ensure_project_directories()
    frame, schema = load_reviews(args.data_file, args.text_column, args.label_column)
    save_eda_figures(frame, schema["hotel_column"])
    linguistic_summary = save_linguistic_summary(frame)
    train_text, test_text, train_y, test_y = train_test_split(
        frame["review_text"], frame["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=frame["label"],
    )
    models: dict[str, Pipeline] = {
        "Multinomial NB": make_pipeline(MultinomialNB()),
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    }
    records: list[dict[str, float | str]] = []
    trained: dict[str, Pipeline] = {}
    for name, pipeline in models.items():
        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)
        probabilities = pipeline.predict_proba(test_text)[:, 1]
        metrics, report = evaluate_predictions(test_y, predictions)
        records.append({"model": name, **metrics})
        save_confusion_matrix(test_y, predictions, name)
        save_json(report, RESULTS_DIR / f"classification_report_{name.lower().replace(' ', '_')}.json")
        trained[name] = pipeline
        if name == "Logistic Regression":
            errors = pd.DataFrame({
                "actual_label": test_y.map({0: "truthful", 1: "deceptive"}),
                "predicted_label": pd.Series(predictions, index=test_y.index).map({0: "truthful", 1: "deceptive"}),
                "deceptive_probability": probabilities, "review_text": test_text,
            })
            errors = errors[errors["actual_label"] != errors["predicted_label"]].copy()
            errors["error_type"] = errors.apply(
                lambda row: "false_positive" if row.actual_label == "truthful" else "false_negative", axis=1)
            errors.to_csv(RESULTS_DIR / "misclassified_reviews.csv", index=False)
    comparison = pd.DataFrame(records)
    save_model_comparison(comparison)
    logistic = trained["Logistic Regression"]
    cv_results = _cross_validation(
        make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)), train_text, train_y)
    save_json(cv_results, RESULTS_DIR / "cross_validation.json")
    if args.compare_tfidf:
        _tfidf_configuration_experiment(train_text, train_y)
    influential = save_influential_words(logistic)
    hotel_result = None
    if schema["hotel_column"]:
        hotel_result = _hotel_holdout_experiment(
            frame, schema["hotel_column"],
            make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        )
        if hotel_result:
            save_json(hotel_result, RESULTS_DIR / "hotel_holdout_experiment.json")
    joblib.dump(logistic, MODEL_PATH)
    metadata = {
        "training_date_utc": datetime.now(timezone.utc).isoformat(), "dataset_size": len(frame),
        "model_name": "TF-IDF + Logistic Regression", "random_state": RANDOM_STATE,
        "schema": schema, "tfidf_parameters": logistic.named_steps["tfidf"].get_params(),
        "test_metrics": next(record for record in records if record["model"] == "Logistic Regression"),
        "cross_validation": cv_results, "hotel_holdout_experiment": hotel_result,
        "linguistic_summary": linguistic_summary.to_dict(),
        "note": "Feature associations describe this trained corpus, not proof of deception.",
    }
    save_json(metadata, METADATA_PATH)
    print("Training complete. Artifacts saved in models/ and reports/.")
    print(comparison.to_string(index=False))
    print("Cross-validation:", cv_results)
    print("Top influential features:")
    print(influential.head(5).to_string(index=False))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReviewGuard on a labelled hotel review dataset.")
    parser.add_argument("--data-file", help="Optional CSV/TSV path; inferred from data/ when omitted.")
    parser.add_argument("--text-column", help="Explicit review text column name.")
    parser.add_argument("--label-column", help="Explicit target column name.")
    parser.add_argument(
        "--compare-tfidf", action="store_true",
        help="Compare n-grams and stopword removal with CV on the training split only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except (FileNotFoundError, DatasetSchemaError, ValueError) as error:
        raise SystemExit(f"Training could not start: {error}") from error
