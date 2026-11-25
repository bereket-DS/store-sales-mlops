import logging
import time
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Gauge
from logging_loki import LokiHandler
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Fanuel Store Sales API")

# 1. Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

# Custom ML Metrics
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Time spent generating a prediction')
PREDICTION_VALUE = Gauge('model_prediction_value', 'The predicted sales value')

# 2. Loki Logging Setup
logger = logging.getLogger("fanuel_api")
logger.setLevel(logging.INFO)
loki_handler = LokiHandler(
    url="http://loki:3100/loki/api/v1/push",
    tags={"application": "fanuel-api", "environment": "production"},
    version="1"
)
logger.addHandler(loki_handler)

# 3. Load the REAL LightGBM Model from .pkl
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "fanuel" / "lgbm_model.pkl"

try:
    model = joblib.load(str(MODEL_PATH))
    logger.info("LightGBM model loaded successfully from .pkl")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


# 4. Define the Input Schema
class PredictionInput(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    year: int
    month: int
    day: int
    weekday: int
    recent_mean_4w: float
    recent_mean_8w: float
    recent_mean_12w: float
    recent_mean_dow: float
    is_holiday: int


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Fanuel's Real Store Sales Model API is running!"}


@app.post("/predict")
def predict_sales(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    logger.info(f"Real prediction requested for Store {data.store_nbr}, Family {data.family}")

    # Convert input to DataFrame
    df = pd.DataFrame([data.model_dump()])
    df["store_nbr"] = df["store_nbr"].astype("category")
    df["family"] = df["family"].astype("category")

    # Track Latency
    start_time = time.time()
    raw_pred = model.predict(df)
    latency = time.time() - start_time

    real_pred = float(np.clip(np.expm1(raw_pred), 0, None)[0])

    # Record ML Metrics
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_VALUE.set(real_pred)

    return {
        "store_nbr": data.store_nbr,
        "family": data.family,
        "predicted_sales": round(real_pred, 2)
    }