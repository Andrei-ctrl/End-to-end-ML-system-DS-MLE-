import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

import joblib
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_PATH = DATA_DIR / "raw" / "churn-bigml-80.csv"

# Ensure folders exist
ARTIFACTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# MLflow configuration (CRITICAL)
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
mlflow.set_experiment("churn_baseline")


TARGET_COL = "Churn"


def main():

    # Load data
    df = pd.read_csv(DATA_PATH)

    # normalize target
    df[TARGET_COL] = df[TARGET_COL].map(
        {"Yes": 1, "No": 0, True: 1, False: 0}
    )
    if df[TARGET_COL].isna().any():
        raise ValueError("Target contains NaN after mapping. Check label encoding.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Feature groups
    categorical_features = [
        "State",
        "International plan",
        "Voice mail plan",
        "Area code",
    ]

    numerical_features = [
        col for col in X.columns if col not in categorical_features
    ]


    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", acc)

        # Save artifacts
        joblib.dump(pipeline, ARTIFACTS_DIR / "model.joblib")

        metrics = {
            "roc_auc": roc_auc,
            "accuracy": acc,
        }
        with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        schema = {
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "all_features": list(X.columns),
            "target": TARGET_COL,
        }
        with open(ARTIFACTS_DIR / "train_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

        # baseline statistics for drift
        baseline_stats = {
            "numerical": {
                col: {
                    "mean": float(X_train[col].mean()),
                    "std": float(X_train[col].std()),
                }
                for col in numerical_features
            }
        }

        with open(ARTIFACTS_DIR / "baseline_stats.json", "w") as f:
            json.dump(baseline_stats, f, indent=2)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("n_features", X.shape[1])

        print("Training completed")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
