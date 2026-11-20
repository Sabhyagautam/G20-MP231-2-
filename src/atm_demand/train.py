"""
train.py – Train the ATM cash demand model and track experiments with MLflow.

Steps:
1. Load and clean train.csv
2. Build a scikit-learn pipeline (preprocessing + GradientBoosting)
3. Train with cross-validation, log to MLflow
4. Register model as v1 and assign the champion alias

Run as a module:
    python -m src.atm_demand.train
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.atm_demand.clean import clean_train, clean_later
from src.atm_demand.data import load_train, load_later, sha256_of_train

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARAMS_PATH = _REPO_ROOT / "params.yaml"

with open(_PARAMS_PATH) as _f:
    PARAMS = yaml.safe_load(_f)

FEATURE_COLS = PARAMS["preprocessing"]["feature_columns"]
TARGET_COL = PARAMS["preprocessing"]["target_column"]
NUMERIC_COLS = PARAMS["preprocessing"]["numeric_columns"]
CATEGORICAL_COLS = PARAMS["preprocessing"]["categorical_columns"]
MODEL_PATH = _REPO_ROOT / PARAMS["paths"]["model_output"]
TRACKING_URI = PARAMS["mlflow"]["tracking_uri"]
EXPERIMENT_NAME = PARAMS["mlflow"]["experiment_name"]
REGISTERED_MODEL_NAME = PARAMS["mlflow"]["registered_model_name"]


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def build_pipeline(n_estimators: int, max_depth: int, learning_rate: float,
                   random_state: int) -> Pipeline:
    """Build a full scikit-learn preprocessing + model pipeline."""

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


def get_features_target(df: pd.DataFrame):
    """Split DataFrame into X (features) and y (target)."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ---------------------------------------------------------------------------
# Single MLflow training run
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    train_df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    random_state: int = 42,
    register: bool = False,
    version_note: str = "",
) -> str:
    """
    Train one model, log to MLflow, optionally register.
    Returns the MLflow run_id.
    """
    data_sha = sha256_of_train()

    X, y = get_features_target(train_df)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(n_estimators, max_depth, learning_rate, random_state)

    # 5-fold CV MAE on training split
    cv_scores = cross_val_score(
        pipeline, X_tr, y_tr,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    cv_mae = float(-cv_scores.mean())

    # Fit on full training split
    pipeline.fit(X_tr, y_tr)

    val_preds = pipeline.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, val_preds))

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        # Mandatory parameter: data_sha256
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("version_note", version_note)

        mlflow.log_metric("mae_cv", cv_mae)
        mlflow.log_metric("mae_val", val_mae)

        # Evaluate on later.csv as well (after cleaning)
        later_df = clean_later(load_later())
        X_later, y_later = get_features_target(later_df)
        later_preds = pipeline.predict(X_later)
        mae_later = float(mean_absolute_error(y_later, later_preds))
        mlflow.log_metric("mae_later", mae_later)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        print(f"  run_id:    {run_id}")
        print(f"  mae_cv:    {cv_mae:.2f}")
        print(f"  mae_val:   {val_mae:.2f}")
        print(f"  mae_later: {mae_later:.2f}")

        if register:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
            print(f"  registered model version: {mv.version}")

    return run_id


# ---------------------------------------------------------------------------
# Main: run three experiments and pick best v1
# ---------------------------------------------------------------------------

def train_v1() -> tuple[str, float]:
    """
    Run three experiments (different hyperparameters) for v1.
    Returns (best_run_id, best_mae_val).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df = clean_train(load_train())

    experiments = [
        dict(name="v1_lr01_d4_n200", n_estimators=200, max_depth=4, learning_rate=0.10),
        dict(name="v1_lr05_d3_n100", n_estimators=100, max_depth=3, learning_rate=0.05),
        dict(name="v1_lr02_d5_n300", n_estimators=300, max_depth=5, learning_rate=0.08),
    ]

    results = []
    print("=== Running v1 experiments ===")
    for exp in experiments:
        print(f"\nExperiment: {exp['name']}")
        run_id = run_experiment(
            name=exp["name"],
            train_df=train_df,
            n_estimators=exp["n_estimators"],
            max_depth=exp["max_depth"],
            learning_rate=exp["learning_rate"],
        )
        client = MlflowClient()
        run = client.get_run(run_id)
        mae_val = run.data.metrics["mae_val"]
        results.append((run_id, mae_val, exp))

    # Pick the best run by mae_val
    best_run_id, best_mae, best_exp = min(results, key=lambda x: x[1])
    print(f"\nBest v1 run: {best_run_id}  mae_val={best_mae:.2f}")

    # Register the best model as v1
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        pass  # already exists

    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    print(f"Registered model version: {mv.version}")

    # Set champion alias to v1 initially
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
    print(f"Champion alias set to version {mv.version}")

    # Save champion model to models/model.joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    champion_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    loaded = mlflow.sklearn.load_model(champion_uri)
    joblib.dump(loaded, MODEL_PATH)
    print(f"Champion model saved to {MODEL_PATH}")

    return best_run_id, best_mae


# ---------------------------------------------------------------------------
# Load the current champion model
# ---------------------------------------------------------------------------

def load_champion() -> Pipeline:
    """Load the champion model from disk (models/model.joblib)."""
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best_run_id, best_mae = train_v1()
    print(f"\nDone. Best run id: {best_run_id}, MAE: {best_mae:.2f}")
