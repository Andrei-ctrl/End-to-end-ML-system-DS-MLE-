import json
from pathlib import Path
from src.monitoring.drift_check import load_inference_data

def test_log_parsing(tmp_path, monkeypatch):
    log_file = tmp_path / "api.log"

    record = {
        "record": {
            "message": str({
                "event": "prediction",
                "features": {
                    "State": "CA",
                    "Account length": 10
                }
            })
        }
    }

    log_file.write_text(json.dumps(record) + "\n")

    monkeypatch.setattr(
        "src.monitoring.drift_check.LOG_PATH", log_file
    )

    df = load_inference_data()

    assert not df.empty
    assert "State" in df.columns
