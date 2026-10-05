"""Dataset discovery, schema inference, cleaning, and descriptive helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATA_DIR
from .preprocessing import clean_review_text

TEXT_CANDIDATES = ("review", "review_text", "text", "content", "reviewtext", "comments")
LABEL_CANDIDATES = ("deceptive", "label", "class", "target", "spam", "fake", "truthful")
POLARITY_CANDIDATES = ("polarity", "sentiment", "rating", "review_polarity")
HOTEL_CANDIDATES = ("hotel", "hotel_name", "property", "venue")


class DatasetSchemaError(ValueError):
    """Raised when a usable review/label schema cannot be inferred."""


def discover_data_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return possible delimited datasets below the configured data folder."""
    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    )


def _read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else None
    try:
        return pd.read_csv(path, sep=separator, engine="python")
    except Exception as error:
        raise DatasetSchemaError(f"Could not read {path.name}: {error}") from error


def _match_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower().replace(" ", "_"): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _infer_text_column(frame: pd.DataFrame) -> str | None:
    direct = _match_column(frame.columns, TEXT_CANDIDATES)
    if direct:
        return direct
    object_columns = [column for column in frame.columns if frame[column].dtype == "object"]
    long_text = [column for column in object_columns if frame[column].dropna().astype(str).str.len().mean() > 40]
    return long_text[0] if len(long_text) == 1 else None


def _infer_label_column(frame: pd.DataFrame) -> str | None:
    direct = _match_column(frame.columns, LABEL_CANDIDATES)
    if direct:
        return direct
    possible = [column for column in frame.columns if 1 < frame[column].nunique(dropna=True) <= 4]
    return possible[0] if len(possible) == 1 else None


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Map commonly used binary labels to truthful=0 and deceptive=1."""
    values = labels.astype(str).str.strip().str.lower()
    mapping = {
        "truthful": 0, "genuine": 0, "real": 0, "authentic": 0, "true": 0,
        "0": 0, "deceptive": 1, "fake": 1, "spam": 1, "fraudulent": 1,
        "false": 1, "1": 1,
    }
    normalized = values.map(mapping)
    if normalized.isna().any():
        unknown = sorted(values[normalized.isna()].unique())
        raise DatasetSchemaError(
            "Could not normalize label values: " + ", ".join(unknown[:10]) +
            ". Expected values such as truthful/deceptive, real/fake, or 0/1."
        )
    if normalized.nunique() != 2:
        raise DatasetSchemaError("The target must contain both truthful and deceptive classes.")
    return normalized.astype(int)


def load_reviews(
    data_file: str | Path | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Load and clean a review dataset, returning canonical columns and schema info."""
    if data_file is None:
        candidates = discover_data_files(data_dir)
        if not candidates:
            raise FileNotFoundError(
                f"No CSV/TSV/TXT dataset found in {data_dir}. See data/README.md for setup."
            )
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise DatasetSchemaError(f"Multiple dataset candidates found ({names}). Use --data-file.")
        path = candidates[0]
    else:
        path = Path(data_file)
    frame = _read_table(path)
    inferred_text = text_column or _infer_text_column(frame)
    inferred_label = label_column or _infer_label_column(frame)
    if not inferred_text or not inferred_label:
        raise DatasetSchemaError(
            "Could not identify review text and label columns. Available columns: " +
            ", ".join(map(str, frame.columns)) +
            ". Pass --text-column and --label-column explicitly."
        )
    if inferred_text not in frame or inferred_label not in frame:
        raise DatasetSchemaError("Configured text or label column is not in the dataset.")
    schema = {
        "source_file": str(path), "text_column": inferred_text, "label_column": inferred_label,
        "polarity_column": _match_column(frame.columns, POLARITY_CANDIDATES),
        "hotel_column": _match_column(frame.columns, HOTEL_CANDIDATES),
    }
    result = frame.copy()
    result = result.dropna(subset=[inferred_text, inferred_label])
    result["review_text"] = result[inferred_text].map(clean_review_text)
    result = result[result["review_text"].ne("")].copy()
    result["label"] = normalize_labels(result[inferred_label])
    result = result.drop_duplicates(subset=["review_text", "label"]).reset_index(drop=True)
    return result, schema


def dataset_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact dataset quality and class-balance indicators."""
    return pd.DataFrame({
        "rows": [len(frame)], "columns": [len(frame.columns)],
        "duplicate_reviews": [frame.duplicated(subset=["review_text"]).sum()],
        "truthful": [(frame["label"] == 0).sum()], "deceptive": [(frame["label"] == 1).sum()],
    })
