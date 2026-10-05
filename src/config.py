"""Central project configuration and stable paths."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_NAMES = ["Truthful", "Deceptive"]
MODEL_PATH = MODELS_DIR / "reviewguard_logistic_regression.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

DEFAULT_TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}


def ensure_project_directories() -> None:
    """Create directories used for generated artifacts."""
    for directory in (DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
