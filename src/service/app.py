from pathlib import Path
import joblib
import json
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from loguru import logger

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "train_schema.json"

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.add(
    LOGS_DIR / "api.log",
    serialize=True,
    level="INFO",
)

# -------------------------------------------------
# Load artifacts
# -------------------------------------------------
model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

FEATURES = schema["all_features"]

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(title="Churn Prediction API")

# -------------------------------------------------
# Prometheus metrics
# -------------------------------------------------
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests",
)

# -------------------------------------------------
# Schemas
# -------------------------------------------------
class PredictionRequest(BaseModel):
    features: dict


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()
    REQUEST_COUNT.inc()

    # validate features
    missing = set(FEATURES) - set(request.features.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}",
        )

    df = pd.DataFrame([request.features])

    proba = model.predict_proba(df)[0, 1]
    prediction = int(proba >= 0.5)

    logger.info({
        "event": "prediction",
        "features": request.features,
        "probability": float(proba),
        "prediction": prediction,
    })

    REQUEST_LATENCY.observe(time.time() - start_time)

    return PredictionResponse(
        churn_probability=float(proba),
        churn_prediction=prediction,
    )


@app.get("/metrics")
def metrics():
    return generate_latest()
