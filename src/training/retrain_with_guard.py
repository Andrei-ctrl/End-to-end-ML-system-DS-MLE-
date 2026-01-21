import json
from pathlib import Path

from src.training.train import main as train_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def is_model_better(
    new_auc: float,
    old_auc: float | None,
    tolerance: float = 0.0,
) -> bool:
    if old_auc is None:
        return True
    return new_auc >= old_auc - tolerance


def retrain_with_guard():
    print("Starting retraining with performance guard...")

    # Load current production metrics
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            current_metrics = json.load(f)
        current_roc_auc = current_metrics["roc_auc"]
    else:
        current_roc_auc = None

    # Train new model
    new_metrics = train_model()
    new_roc_auc = new_metrics["roc_auc"]

    # Guard decision
    if not is_model_better(new_roc_auc, current_roc_auc, tolerance=0.01):
        print(
            f"New model rejected: ROC-AUC {new_roc_auc:.4f} "
            f"< current {current_roc_auc:.4f}"
        )
        return

    print(
        f"New model accepted: ROC-AUC {new_roc_auc:.4f} "
        f"(previous {current_roc_auc})"
    )


if __name__ == "__main__":
    retrain_with_guard()
