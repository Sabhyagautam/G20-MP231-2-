"""
data.py – Load datasets and audit the seven kinds of data problem.

Run as a module to print a full audit report:
    python -m src.atm_demand.data
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths (read from params.yaml so the script works from the repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARAMS_PATH = _REPO_ROOT / "params.yaml"

with open(_PARAMS_PATH) as _f:
    _PARAMS = yaml.safe_load(_f)

TRAIN_PATH = _REPO_ROOT / _PARAMS["paths"]["train_raw"]
LATER_PATH = _REPO_ROOT / _PARAMS["paths"]["later_raw"]
HOLDOUT_PATH = _REPO_ROOT / _PARAMS["paths"]["holdout_raw"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_train() -> pd.DataFrame:
    """Load the raw training CSV exactly as given (no changes)."""
    return pd.read_csv(TRAIN_PATH)


def load_later() -> pd.DataFrame:
    """Load the later.csv (July–August 2026) dataset."""
    return pd.read_csv(LATER_PATH)


def load_holdout() -> pd.DataFrame:
    """Load holdout features (September 2026, no target)."""
    return pd.read_csv(HOLDOUT_PATH)


def sha256_of_train() -> str:
    """Return the SHA-256 hex digest of the original train.csv file."""
    with open(TRAIN_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
# Seven data-quality checks
# ---------------------------------------------------------------------------

def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """
    A leaking column is present if a column exists in train but NOT in holdout,
    AND it has a very high rank correlation with the target.
    """
    extra = set(train.columns) - set(holdout.columns) - {"withdrawal_thousand_rs"}
    result = {"present": False, "column": None}

    for col in extra:
        if train[col].dtype in [np.float64, np.int64]:
            spearman_r = train[col].rank().corr(train["withdrawal_thousand_rs"].rank())
            if abs(spearman_r) > 0.95:
                result = {"present": True, "column": col}
                break

    return result


def check_unit_mixup(train: pd.DataFrame) -> dict:
    """
    A unit mixup exists when one value of the group column records a numeric column
    at a different scale (10x, 100x, 1000x) relative to the other groups.
    """
    numeric_cols = _PARAMS["preprocessing"]["numeric_columns"]
    group_col = "area"
    factor_thresholds = [10, 12, 60, 100, 1000]

    for col in numeric_cols:
        group_means = train.groupby(group_col)[col].mean()
        if group_means.min() == 0:
            continue
        ratio = group_means.max() / group_means.min()
        for thresh in factor_thresholds:
            if ratio >= thresh * 0.8:
                # Find the outlier group
                overall_median = train[col].median()
                group_diffs = (group_means - overall_median).abs()
                outlier_group = group_diffs.idxmax()
                return {
                    "present": True,
                    "column": col,
                    "group_value": outlier_group,
                    "factor": thresh,
                }

    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(train: pd.DataFrame) -> dict:
    """
    A sentinel value is an impossible number (like 9999, -1, -999) used as a placeholder
    for missing data. Detected by finding values that appear many times and are impossible.
    """
    numeric_cols = _PARAMS["preprocessing"]["numeric_columns"]
    sentinel_candidates = [-999, -99, -9, -1, 999, 9999, 99999]

    for col in numeric_cols:
        for candidate in sentinel_candidates:
            count = int((train[col] == candidate).sum())
            if count >= 10:  # repeated enough to be a code, not a real value
                return {
                    "present": True,
                    "column": col,
                    "value": float(candidate),
                    "count": count,
                }

    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train: pd.DataFrame) -> dict:
    """
    Duplicate rows are records that appear more than once.
    Returns the count of EXTRA copies (rows you would delete).
    """
    extra = int(train.duplicated().sum())
    return {"present": extra > 0, "count": extra if extra > 0 else None}


def check_stuck_sensor(train: pd.DataFrame) -> dict:
    """
    A stuck sensor repeatedly sends the same value for a long consecutive run of days.
    Sort by date and look for the longest consecutive same-value run in numeric columns.
    """
    numeric_cols = _PARAMS["preprocessing"]["numeric_columns"]
    df = train.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    best_run = {"length": 0, "column": None, "value": None, "start": None, "end": None}

    for col in numeric_cols:
        vals = df[col].values
        dates = df["date"].values
        i = 0
        while i < len(vals):
            j = i
            while j < len(vals) and vals[j] == vals[i]:
                j += 1
            run_len = j - i
            if run_len > best_run["length"]:
                best_run = {
                    "length": run_len,
                    "column": col,
                    "value": float(vals[i]),
                    "start": pd.Timestamp(dates[i]),
                    "end": pd.Timestamp(dates[j - 1]),
                }
            i = j

    # Only flag as stuck sensor if the run spans at least 14 consecutive calendar days
    if best_run["column"] is not None and best_run["length"] >= 14:
        span_days = (best_run["end"] - best_run["start"]).days
        if span_days >= 14:
            return {
                "present": True,
                "column": best_run["column"],
                "start_date": best_run["start"].strftime("%Y-%m-%d"),
                "end_date": best_run["end"].strftime("%Y-%m-%d"),
            }

    return {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }


def check_target_typos(train: pd.DataFrame) -> dict:
    """
    Target typos are values 10× too large – a small cluster sits far above all others
    with a large empty gap. Identified by finding a gap in the sorted target.
    """
    target = train["withdrawal_thousand_rs"].sort_values().values
    threshold = float(_PARAMS["preprocessing"]["target_typo_threshold"])
    count = int((target > threshold).sum())

    if count > 0:
        return {"present": True, "count": count}
    return {"present": False, "count": None}


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    """
    Spelling variants are the same category written differently.
    later.csv has the clean spellings, so its unique count = true number of levels.
    """
    text_cols = _PARAMS["preprocessing"]["categorical_columns"]

    for col in text_cols:
        raw_levels = train[col].nunique()
        clean_levels = later[col].nunique()
        if raw_levels > clean_levels:
            return {
                "present": True,
                "column": col,
                "true_levels": int(clean_levels),
            }

    return {"present": False, "column": None, "true_levels": None}


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def audit_all(verbose: bool = True) -> dict:
    """Run all seven checks and return a dict of results."""
    train = load_train()
    later = load_later()
    holdout = load_holdout()

    results = {
        "leaking_column": check_leaking_column(train, holdout),
        "unit_mixup": check_unit_mixup(train),
        "sentinel_value": check_sentinel_value(train),
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train),
        "target_typos": check_target_typos(train),
        "spelling_variants": check_spelling_variants(train, later),
    }

    if verbose:
        print("=" * 55)
        print("DATA QUALITY AUDIT – MP231 / G020")
        print("=" * 55)
        for issue, detail in results.items():
            status = "PRESENT" if detail.get("present") else "absent"
            print(f"\n[{status}] {issue}")
            for k, v in detail.items():
                if k != "present":
                    print(f"       {k}: {v}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    audit_all(verbose=True)
    print(f"\ntrain.csv SHA-256: {sha256_of_train()}")
