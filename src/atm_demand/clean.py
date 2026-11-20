"""
clean.py – Reproduce the cleaning pipeline that fixes all five present data issues.

What this module does:
1. Remove leaking column (cash_reloaded_next_day)
2. Replace sentinel values (9999 in distance_to_market_km) with NaN, then impute
3. Remove duplicate rows
4. Fix target typos (divide values > threshold by 10)
5. Standardise atm_type spelling variants to the three clean levels

Run as a module:
    python -m src.atm_demand.clean
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARAMS_PATH = _REPO_ROOT / "params.yaml"

with open(_PARAMS_PATH) as _f:
    _PARAMS = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
# Spelling-variant normalisation map for atm_type
# ---------------------------------------------------------------------------
# later.csv has the three canonical spellings: bank_branch, mall, standalone

_ATM_TYPE_MAP = {
    # extra spaces
    " mall": "mall",
    " standalone": "standalone",
    " bank_branch": "bank_branch",
    # all-caps
    "MALL": "mall",
    "STANDALONE": "standalone",
    "BANK_BRANCH": "bank_branch",
    # mixed case
    "Mall": "mall",
    "Standalone": "standalone",
    "Bank_Branch": "bank_branch",
    # abbreviations / typos
    "mal.": "mall",
    "sta.": "standalone",
    "ban.": "bank_branch",
    # space instead of underscore
    "bank branch": "bank_branch",
    # repeated canonical (already correct)
    "mall": "mall",
    "standalone": "standalone",
    "bank_branch": "bank_branch",
}


def fix_spelling_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Map all atm_type variants to the three canonical spellings."""
    df = df.copy()
    df["atm_type"] = (
        df["atm_type"]
        .str.strip()
        .map(lambda x: _ATM_TYPE_MAP.get(x, x.lower().strip()
             .replace(" ", "_").replace(".", "")))
    )
    return df


def remove_leaking_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that would not be known at prediction time."""
    cols_to_drop = [
        c for c in _PARAMS["preprocessing"]["leaking_columns"] if c in df.columns
    ]
    return df.drop(columns=cols_to_drop)


def fix_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel 9999 in distance_to_market_km with the column median."""
    df = df.copy()
    col = _PARAMS["preprocessing"]["sentinel_column"]
    sentinel = _PARAMS["preprocessing"]["sentinel_value"]
    df[col] = df[col].replace(sentinel, np.nan)
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extra copies of duplicate rows, keeping the first occurrence."""
    return df.drop_duplicates().reset_index(drop=True)


def fix_target_typos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide by 10 any target value that sits far above the normal distribution.
    The gap between ~1135 and 4076 identifies the typo boundary.
    """
    df = df.copy()
    target_col = _PARAMS["preprocessing"]["target_column"]
    if target_col not in df.columns:
        return df
    threshold = _PARAMS["preprocessing"]["target_typo_threshold"]
    mask = df[target_col] > threshold
    df.loc[mask, target_col] = df.loc[mask, target_col] / 10.0
    return df


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw training data.
    Returns a clean DataFrame ready for model training.
    """
    df = fix_spelling_variants(df)
    df = remove_leaking_columns(df)
    df = fix_sentinel_values(df)
    df = remove_duplicates(df)
    df = fix_target_typos(df)
    return df


def clean_later(df: pd.DataFrame) -> pd.DataFrame:
    """
    later.csv is already clean (per the problem statement).
    We still run spelling fix and sentinel fix for safety,
    but skip duplicate removal and target-typo fix.
    """
    df = fix_spelling_variants(df)
    df = remove_leaking_columns(df)
    df = fix_sentinel_values(df)
    return df


def clean_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean holdout features (no target column present).
    """
    df = fix_spelling_variants(df)
    df = fix_sentinel_values(df)
    return df


# ---------------------------------------------------------------------------
# CLI entry point – print before/after stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.atm_demand.data import load_train, load_later, load_holdout

    raw = load_train()
    cleaned = clean_train(raw)

    print("=== CLEANING REPORT ===")
    print(f"Raw rows:            {len(raw)}")
    print(f"After dedup:         {len(cleaned)} (removed {len(raw) - len(cleaned)} duplicates)")
    print(f"Sentinel fixes:      distance_to_market_km 9999→median")
    print(f"Target typo fixes:   {(raw['withdrawal_thousand_rs'] > 1200).sum()} values divided by 10")
    print(f"Leaking col removed: cash_reloaded_next_day")
    print(f"atm_type levels:     {raw['atm_type'].nunique()} → {cleaned['atm_type'].nunique()}")
    print(f"\nFinal train shape:   {cleaned.shape}")
    print(f"Columns: {list(cleaned.columns)}")
