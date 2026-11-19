import json
import os
import joblib
import pandas as pd
import shap

MODEL_DIR = "models/vehicle"
MODEL_PATH = os.path.join(MODEL_DIR, "vehicle_failure_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "vehicle_metrics.json")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "vehicle_threshold.json")

# Load artifacts once at import time for fast API/dashboard responses
_model = joblib.load(MODEL_PATH)
_explainer = shap.TreeExplainer(_model)

with open(METRICS_PATH, "r") as f:
    _feature_columns = json.load(f)["feature_columns"]

with open(THRESHOLD_PATH, "r") as f:
    _threshold = float(json.load(f)["threshold"])


def predict_vehicle_failure(
    vehicle_type: str,
    sensor_rpm: float,
    sensor_speed: float,
    sensor_temp: float,
) -> dict:
    """
    Predict failure probability and SHAP explanations for a single vehicle reading.
    vehicle_type must be 'heavy_truck' or 'passenger_car'.
    """
    if vehicle_type not in ("heavy_truck", "passenger_car"):
        raise ValueError("vehicle_type must be 'heavy_truck' or 'passenger_car'")

    row = pd.DataFrame(
        [
            {
                "sensor_rpm": float(sensor_rpm),
                "sensor_speed": float(sensor_speed),
                "sensor_temp": float(sensor_temp),
                "vehicle_type_heavy_truck": 1.0 if vehicle_type == "heavy_truck" else 0.0,
                "vehicle_type_passenger_car": 1.0 if vehicle_type == "passenger_car" else 0.0,
            }
        ]
    )
    row = row.reindex(columns=_feature_columns, fill_value=0.0).astype(float)

    prob = float(_model.predict_proba(row)[0, 1])
    is_failure = prob >= _threshold

    if prob >= _threshold:
        risk_level = "HIGH"
    elif prob >= _threshold * 0.6:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    shap_vals = _explainer.shap_values(row)[0]
    shap_contributions = {
        col: round(float(val), 4)
        for col, val in sorted(
            zip(_feature_columns, shap_vals), key=lambda x: abs(x[1]), reverse=True
        )
    }

    return {
        "vehicle_type": vehicle_type,
        "failure_probability": round(prob, 4),
        "threshold": _threshold,
        "failure_predicted": bool(is_failure),
        "risk_level": risk_level,
        "shap_contributions": shap_contributions,
    }


if __name__ == "__main__":
    # Quick local sanity check for both vehicle types
    print("Truck sample:", predict_vehicle_failure("heavy_truck", 45000, 1200, 800))
    print("Car sample:", predict_vehicle_failure("passenger_car", 58.0, 26.0, 2.0))