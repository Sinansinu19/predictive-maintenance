import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone

# Import strictly from the contract-approved files
from src import config
from src.features import make_input_row
from src.decision import load_threshold
from src.failure_model import load_failure_model, predict_failure_proba
from src.anomaly import detect_anomalies

# --- Initialize FastAPI ---
app = FastAPI(title="Predictive Maintenance Live API - Contract v4.1")

# --- In-Memory Store for Live Dashboard ---
READINGS_STORE = []
MAX_STORE = 200
SEQ_COUNTER = 0

# --- Load Models at Startup (Fail Fast) ---
try:
    failure_model = load_failure_model()
    
    # Strictly check paths from config.py to fail fast if missing
    if not config.FAILURE_MODEL_PATH.exists() or not config.ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError("Model files not found in the config paths.")
    
    active_threshold = load_threshold()
    print(f"✅ Models loaded successfully. Active threshold: {active_threshold}")
except Exception as e:
    raise RuntimeError(f"Startup failed. Ensure models are trained and saved: {e}")

# --- Pydantic Schemas for Validation ---
class MachineInputs(BaseModel):
    type: Literal["L", "M", "H"] = Field(..., description="Machine type must be L, M, or H")
    air_temp_k: float = Field(..., gt=0)
    process_temp_k: float = Field(..., gt=0)
    rot_speed_rpm: float = Field(..., gt=0)
    torque_nm: float = Field(..., ge=0)
    tool_wear_min: float = Field(..., ge=0)

class PredictRequest(BaseModel):
    machine_id: str
    inputs: MachineInputs
    weather: Optional[dict] = None

# --- Endpoints ---

@app.post("/predict")
def predict(data: PredictRequest):
    global SEQ_COUNTER
    
    # 1. Generate full features using Member 1's feature engineering script
   # 1. Generate full features using Member 1's feature engineering script
    try:
        input_df = make_input_row(
            type_=data.inputs.type,
            air_temp_k=data.inputs.air_temp_k,
            process_temp_k=data.inputs.process_temp_k,
            rot_speed_rpm=data.inputs.rot_speed_rpm,
            torque_nm=data.inputs.torque_nm,
            tool_wear_min=data.inputs.tool_wear_min
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

    # 2. Get predictions
    prob = float(predict_failure_proba(input_df)[0])
    
    flags, scores = detect_anomalies(input_df)
    a_flag = bool(flags[0])
    a_score = float(scores[0])

    # 3. Apply dynamically loaded threshold
    trigger = bool(prob >= active_threshold)

    # Extract the computed values
    temp_diff_k = float(input_df["temp_diff_k"].iloc[0])
    power_w = float(input_df["power_w"].iloc[0])

    # 4. Build exact contract-compliant response
    SEQ_COUNTER += 1
    now = datetime.now(timezone.utc).isoformat()
    
    response = {
        "seq": SEQ_COUNTER,
        "machine_id": data.machine_id,
        "timestamp": now,
        "probability": round(prob, 4),
        "threshold": active_threshold,
        "trigger_alert": trigger,
        "anomaly_flag": a_flag,
        "anomaly_score": round(a_score, 4),
        "temp_diff_k": round(temp_diff_k, 4),
        "power_w": round(power_w, 4),
        "inputs": data.inputs.model_dump(),
        "weather": data.weather
    }

    # 5. Store in memory for the Streamlit dashboard
    READINGS_STORE.append(response)
    if len(READINGS_STORE) > MAX_STORE:
        READINGS_STORE.pop(0)

    return response

@app.get("/latest")
def get_latest(n: int = Query(50, le=200, gt=0)):
    return READINGS_STORE[-n:]

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models": {
            "failure": config.FAILURE_MODEL_PATH.exists(),
            "anomaly": config.ANOMALY_MODEL_PATH.exists()
        },
        "active_threshold": active_threshold,
        "stored_readings": len(READINGS_STORE)
    }