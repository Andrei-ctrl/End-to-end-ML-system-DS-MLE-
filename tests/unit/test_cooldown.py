import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.monitoring.drift_check import can_retrain

def test_can_retrain_when_no_state(tmp_path, monkeypatch):
    fake_state = tmp_path / "retrain_state.json"
    monkeypatch.setattr(
        "src.monitoring.drift_check.STATE_PATH", fake_state
    )
    assert can_retrain() is True


def test_can_retrain_blocked_by_cooldown(tmp_path, monkeypatch):
    fake_state = tmp_path / "retrain_state.json"
    now = datetime.now(timezone.utc)

    fake_state.write_text(
        json.dumps({"last_retrain": now.isoformat()})
    )

    monkeypatch.setattr(
        "src.monitoring.drift_check.STATE_PATH", fake_state
    )
    monkeypatch.setattr(
        "src.monitoring.drift_check.COOLDOWN_HOURS", 24
    )

    assert can_retrain() is False
