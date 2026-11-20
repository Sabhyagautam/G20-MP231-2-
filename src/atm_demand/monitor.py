"""
monitor.py – Drift monitoring, retraining, and promotion logic.

Steps:
1. Compute PSI for every numeric column (cleaned train vs later)
2. Compute v1 MAE per area group on later.csv
3. Retrain v2 on train + first 70% of later.csv
4. Compare v1 and v2 on the last 30% of later.csv
5. Promote v2 only if its MAE is at least 15% lower than v1's

Run as a module:
    python -m src.atm_demand.monitor
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.atm_demand.clean import clean_train, clean_later
from src.atm_demand.data import load_train, load_later, sha256_of_train
from src.atm_demand.train import (
    FEATURE_COLS, TARGET_COL, REGISTERED_MODEL_NAME,
    EXPERIMENT_NAME, TRACKING_URI, MODEL_PATH, PARAMS,
    build_pipeline, get_features_target, run_experiment,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) between reference and comparison arrays.
    Uses quantile-based binning on the reference distribution.
    """
    # Build bin edges from reference
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, quantiles)

    # Avoid duplicate edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Clip both arrays to range
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cmp_counts, _ = np.histogram(comparison, bins=bin_edges)

    # Convert to proportions with small smoothing to avoid log(0)
    eps = 1e-6
    ref_pct = ref_counts / (len(reference) + eps) + eps
    cmp_pct = cmp_counts / (len(comparison) + eps) + eps

    # Normalise
    ref_pct = ref_pct / ref_pct.sum()
    cmp_pct = cmp_pct / cmp_pct.sum()

    psi = float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))
    return psi


def compute_all_psi(train_clean: pd.DataFrame, later_clean: pd.DataFrame) -> dict:
    """Compute PSI for every numeric feature column."""
    numeric_cols = PARAMS["preprocessing"]["numeric_columns"]
    psi_results = {}
    for col in numeric_cols:
        ref = train_clean[col].dropna().values
        cmp = later_clean[col].dropna().values
        psi_results[col] = round(compute_psi(ref, cmp), 4)
    return psi_results


# ---------------------------------------------------------------------------
# MAE per group
# ---------------------------------------------------------------------------

def mae_by_area(model, later_clean: pd.DataFrame) -> dict:
    """Compute v1 MAE separately for each area value."""
    results = {}
    for area in later_clean["area"].unique():
        subset = later_clean[later_clean["area"] == area]
        X, y = get_features_target(subset)
        preds = model.predict(X)
        results[area] = round(float(mean_absolute_error(y, preds)), 4)
    return results


# ---------------------------------------------------------------------------
# Retrain v2 and compare
# ---------------------------------------------------------------------------

def retrain_v2_and_compare():
    """
    Full monitoring + retraining workflow.
    Returns a dict with all monitoring results and the promotion decision.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_clean = clean_train(load_train())
    later_clean = clean_later(load_later())

    # Sort later.csv by date (required by problem spec)
    later_clean["date"] = pd.to_datetime(later_clean["date"])
    later_clean = later_clean.sort_values("date").reset_index(drop=True)

    # 1. PSI
    print("=== Computing PSI ===")
    psi_dict = compute_all_psi(train_clean, later_clean)
    for col, psi in psi_dict.items():
        print(f"  {col}: PSI = {psi:.4f}")
    most_drifted = max(psi_dict, key=psi_dict.get)
    print(f"\n  Most drifted feature: {most_drifted} (PSI={psi_dict[most_drifted]:.4f})")

    # 2. Load v1 champion and evaluate per area
    print("\n=== v1 MAE per area (on full later.csv) ===")
    v1_model = joblib.load(MODEL_PATH)
    mae_area = mae_by_area(v1_model, later_clean)
    for area, mae in mae_area.items():
        print(f"  {area}: MAE = {mae:.2f}")
    worst_group = max(mae_area, key=mae_area.get)
    print(f"  Worst served group: {worst_group}")

    # 3. Split later.csv 70/30 by date
    split_idx = int(len(later_clean) * 0.70)
    later_train_part = later_clean.iloc[:split_idx]
    later_check = later_clean.iloc[split_idx:]
    print(f"\n=== later.csv split: train_part={len(later_train_part)}, check={len(later_check)} ===")

    # 4. v1 MAE on check set
    X_check, y_check = get_features_target(later_check)
    v1_check_preds = v1_model.predict(X_check)
    mae_v1_check = float(mean_absolute_error(y_check, v1_check_preds))
    print(f"  v1 MAE on check set: {mae_v1_check:.2f}")

    # 5. Build v2 training data: cleaned train + first 70% of later
    v2_train = pd.concat([train_clean, later_train_part], ignore_index=True)
    print(f"\n=== Training v2 on {len(v2_train)} rows ===")

    mp = PARAMS["model"]
    v2_run_id = run_experiment(
        name="v2_retrain",
        train_df=v2_train,
        n_estimators=mp["n_estimators"],
        max_depth=mp["max_depth"],
        learning_rate=mp["learning_rate"],
        random_state=mp["random_state"],
        register=True,
        version_note="v2: trained on train + first 70% of later",
    )

    # Reload the newly registered model
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    v2_version = max(all_versions, key=lambda mv: int(mv.version))
    v2_model_uri = f"models:/{REGISTERED_MODEL_NAME}/{v2_version.version}"
    v2_model = mlflow.sklearn.load_model(v2_model_uri)

    # 6. v2 MAE on check set
    v2_check_preds = v2_model.predict(X_check)
    mae_v2_check = float(mean_absolute_error(y_check, v2_check_preds))
    print(f"  v2 MAE on check set: {mae_v2_check:.2f}")

    # 7. Log check-set MAEs to their respective runs
    with mlflow.start_run(run_id=v2_run_id):
        mlflow.log_metric("mae_check", mae_v2_check)

    # 8. Promotion decision: promote v2 if MAE is at least 15% lower
    threshold = PARAMS["monitoring"]["promotion_threshold"]
    improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
    if improvement >= threshold:
        decision = "promote"
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", v2_version.version)
        # Overwrite champion on disk
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(v2_model, MODEL_PATH)
        print(f"\n  PROMOTED v2 to champion (improvement={improvement:.1%} >= {threshold:.0%})")
    else:
        decision = "keep"
        print(f"\n  KEPT v1 as champion (improvement={improvement:.1%} < {threshold:.0%})")

    result = {
        "psi": psi_dict,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_area,
        "worst_served_group": worst_group,
        "mae_v1_on_check": round(mae_v1_check, 4),
        "mae_v2_on_check": round(mae_v2_check, 4),
        "decision": decision,
    }

    return result


# ---------------------------------------------------------------------------
# Holdout predictions
# ---------------------------------------------------------------------------

def predict_holdout():
    """Generate predictions for holdout_features.csv using the champion model."""
    from src.atm_demand.clean import clean_holdout
    from src.atm_demand.data import load_holdout

    model = joblib.load(MODEL_PATH)
    holdout = clean_holdout(load_holdout())

    X = holdout[FEATURE_COLS].copy()
    preds = model.predict(X)

    output = pd.DataFrame({
        "record_id": holdout[PARAMS["preprocessing"]["id_column"]],
        "prediction": preds,
    })

    out_path = _REPO_ROOT / PARAMS["paths"]["predictions_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"Holdout predictions saved to {out_path}  ({len(output)} rows)")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = retrain_v2_and_compare()
    predict_holdout()

    print("\n=== MONITORING SUMMARY ===")
    print("PSI values:")
    for col, psi in results["psi"].items():
        print(f"  {col}: {psi:.4f}")
    print(f"Most drifted: {results['most_drifted_feature']}")
    print("MAE by area (v1 on later.csv):")
    for area, mae in results["mae_by_group"].items():
        print(f"  {area}: {mae:.2f}")
    print(f"Worst served: {results['worst_served_group']}")
    print(f"mae_v1_check: {results['mae_v1_on_check']:.2f}")
    print(f"mae_v2_check: {results['mae_v2_on_check']:.2f}")
    print(f"Decision:     {results['decision']}")
