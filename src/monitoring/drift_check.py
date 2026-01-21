import json
from pathlib import Path
import subprocess
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import sys
from datetime import datetime, timedelta, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn-bigml-80.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "api.log"
REPORTS_DIR = PROJECT_ROOT / "reports" / "drift"
STATE_PATH = PROJECT_ROOT / "artifacts" / "retrain_state.json"
COOLDOWN_HOURS = 0  # 0h cooldown retraining for debugging

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


import ast

def load_inference_data():
    records = []

    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # Loguru stores the message as a string
            msg = obj.get("record", {}).get("message")
            if not msg:
                continue

            # message is a string representation of a dict
            try:
                payload = ast.literal_eval(msg)
            except Exception:
                continue

            if payload.get("event") == "prediction":
                records.append(payload["features"])

    return pd.DataFrame(records)

def should_retrain(drift_report: dict, threshold: float = 0.3) -> bool:
    metrics = drift_report.get("metrics", [])

    drift_metric = next(
        (
            m["result"]
            for m in metrics
            if m.get("metric") == "DatasetDriftMetric"
        ),
        None,
    )

    if drift_metric is None:
        print("No DatasetDriftMetric found — cannot decide retraining.")
        return False

    drifted = drift_metric["number_of_drifted_columns"]
    total = drift_metric["number_of_columns"]
    share = drift_metric["share_of_drifted_columns"]

    print(f"Drifted features: {drifted}/{total} ({share:.2%})")

    return share >= threshold

def can_retrain() -> bool:
    now = datetime.now(timezone.utc)

    # No state file → allow retraining
    if not STATE_PATH.exists():
        return True

    try:
        with open(STATE_PATH) as f:
            content = f.read().strip()
            if not content:
                return True  # empty file → allow retraining
            state = json.loads(content)
    except (json.JSONDecodeError, OSError):
        return True  # corrupted file → allow retraining

    last_retrain = datetime.fromisoformat(state["last_retrain"])

    elapsed = (now - last_retrain).total_seconds() / 3600
    if elapsed < COOLDOWN_HOURS:
        print(f"Cooldown active ({elapsed:.2f}h elapsed)")
        return False

    return True


def update_retrain_state():
    with open(STATE_PATH, "w") as f:
        json.dump(
            {"last_retrain": datetime.now(timezone.utc).isoformat()},
            f,
            indent=2,
        )

def trigger_retraining():
    print("Triggering automatic retraining...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.training.retrain_with_guard",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )



def main():
    if not LOG_PATH.exists():
        print("No inference logs found. Run predictions first.")
        return

    current_data = load_inference_data()
    if current_data.empty:
        print("No prediction records found.")
        return

    reference_data = pd.read_csv(DATA_PATH)
    reference_data = reference_data.drop(columns=["Churn"])

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    drift_json = report.as_dict()

    with open(REPORTS_DIR / "drift_summary.json", "w") as f:
        json.dump(drift_json, f, indent=2)

    output_path = REPORTS_DIR / "drift_report.html"
    report.save_html(str(output_path))

    print(f"Drift report saved to: {output_path}")

    if should_retrain(drift_json):
        if can_retrain():
            print("Drift threshold exceeded — retraining triggered.")
            trigger_retraining()
            update_retrain_state()
        else:
            print("Drift detected but cooldown prevents retraining.")
    else:
        print("Drift within acceptable range — no retraining needed.")


if __name__ == "__main__":
    main()
