from src.training.retrain_with_guard import is_model_better

def test_accept_when_no_previous_model():
    assert is_model_better(0.7, None) is True


def test_accept_when_better():
    assert is_model_better(0.75, 0.73) is True


def test_reject_when_worse():
    assert is_model_better(0.72, 0.73) is False


def test_accept_with_tolerance():
    assert is_model_better(0.729, 0.73, tolerance=0.01) is True

