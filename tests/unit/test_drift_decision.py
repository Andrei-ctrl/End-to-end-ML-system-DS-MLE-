from src.monitoring.drift_check import should_retrain

def test_should_retrain_true():
    drift_report = {
        "metrics": [
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "number_of_columns": 10,
                    "number_of_drifted_columns": 5,
                    "share_of_drifted_columns": 0.5,
                },
            }
        ]
    }
    assert should_retrain(drift_report, threshold=0.3) is True


def test_should_retrain_false():
    drift_report = {
        "metrics": [
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "number_of_columns": 10,
                    "number_of_drifted_columns": 2,
                    "share_of_drifted_columns": 0.2,
                },
            }
        ]
    }
    assert should_retrain(drift_report, threshold=0.3) is False
