"""
Data loading and setup for Fanuel's model.
Handles repository paths, configuration, metadata tracking, and loading raw CSVs.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

# ============================================================
# 1. FROZEN CONTRACT SETUP
# ============================================================
def _find_repo_root() -> Path:
    """Locates the repository root directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src" / "common" / "evaluate.py").exists():
            return candidate
    return Path.cwd()

_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.common.evaluate import (
        HORIZON_DAYS, TARGET, check_lags, rmsle, split_train_valid,
        TRAIN_END, VALID_START, VALID_END
    )
except ImportError:
    print(f"ERROR: Could not import src.common.evaluate from {_REPO_ROOT}.")
    sys.exit(1)

# ============================================================
# 2. METADATA TRACKING
# ============================================================
def get_git_info() -> tuple[str, str]:
    """Fetches current git branch and commit SHA for MLflow."""
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return branch, sha
    except Exception:
        return "undefined", "undefined"

def get_raw_data_md5() -> str:
    """Reads MD5 hash from data/raw.dvc to track data lineage."""
    dvc_file = _REPO_ROOT / "data" / "raw.dvc"
    try:
        with open(dvc_file, "r") as f:
            dvc_data = yaml.safe_load(f)
        return dvc_data["outs"][0]["md5"]
    except Exception:
        return "undefined"

# ============================================================
# 3. CONFIGURATION
# ============================================================
DATA_DIR = _REPO_ROOT / "data" / "raw"
OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "fanuel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_EXPERIMENT = "store-sales-fanuel"
AUTHOR = "fanuel"

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)

# ============================================================
# 4. DATA LOADING
# ============================================================
def load_data():
    """Loads CSVs and ensures 'date' columns are parsed correctly."""
    print("Loading data...")

    def read(name, **kwargs):
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run 'dvc pull'.")

        header_cols = pd.read_csv(path, nrows=0).columns
        if "date" in header_cols:
            kwargs["parse_dates"] = ["date"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, **kwargs)

        if "date" not in df.columns or pd.api.types.is_datetime64_any_dtype(df["date"]):
            return df

        # Robust fallback date parser
        raw = df["date"].copy()
        for parser in [
            lambda s: pd.to_datetime(s, errors="coerce", dayfirst=False),
            lambda s: pd.to_datetime(s, errors="coerce", dayfirst=True),
            lambda s: pd.to_datetime(s, errors="coerce", format="mixed"),
        ]:
            res = parser(raw.astype(str).str.strip())
            if res.isna().sum() == 0:
                df["date"] = res
                return df
        raise ValueError(f"Could not parse dates for {name}")

    return read("train.csv"), read("test.csv"), read("stores.csv"), read("holidays_events.csv")