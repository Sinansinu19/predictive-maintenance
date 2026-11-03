"""Integration pipeline for explainable predictive maintenance."""

from typing import Any

from src.features import make_input_row
from src.failure_model import predict_failure_proba
from src.anomaly import detect_anomalies
from src.failure_mode import predict_failure_mode, predict_failure_mode_proba
from src.explain import explain_failure_model
from src.decision import load_threshold
from src.recommend import recommend_maintenance


def run_pipeline(
    machine_id: str,
    type_: str,
    air_temp_k: float,
    process_temp_k: float,
    rot_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> dict[str, Any]:
    """Run all available AI4I prediction and explanation components."""

    # 1. Generate engineered features.
    features = make_input_row(
        type_=type_,
        air_temp_k=air_temp_k,
        process_temp_k=process_temp_k,
        rot_speed_rpm=rot_speed_rpm,
        torque_nm=torque_nm,
        tool_wear_min=tool_wear_min,
    )

    # 2. Predict binary failure probability.
    probability = float(predict_failure_proba(features)[0])

    # 3. Detect anomalies.
    anomaly_flags, anomaly_scores = detect_anomalies(features)

    anomaly_flag = bool(anomaly_flags[0])
    anomaly_score = float(anomaly_scores[0])

    # 4. Load the active failure threshold.
    threshold = load_threshold()
    trigger_alert = probability >= threshold

    # 5. Predict failure mode.
    failure_modes = predict_failure_mode(features)
    failure_mode = failure_modes[0]

    mode_probabilities = predict_failure_mode_proba(features)

    # 6. Generate SHAP explanation.
    explanation = explain_failure_model(features)

    # 7. Generate maintenance recommendation.
    recommendation = recommend_maintenance(
        probability=probability,
        anomaly_flag=anomaly_flag,
        failure_mode=failure_mode,
        threshold=threshold,
    )

    # 8. Return the integrated result.
    return {
        "machine_id": machine_id,
        "inputs": {
            "type": type_,
            "air_temp_k": air_temp_k,
            "process_temp_k": process_temp_k,
            "rot_speed_rpm": rot_speed_rpm,
            "torque_nm": torque_nm,
            "tool_wear_min": tool_wear_min,
        },
        "features": features.iloc[0].to_dict(),
        "failure_prediction": {
            "probability": round(probability, 4),
            "threshold": threshold,
            "trigger_alert": trigger_alert,
        },
        "anomaly_detection": {
            "anomaly_flag": anomaly_flag,
            "anomaly_score": round(anomaly_score, 4),
        },
        "failure_mode": {
            "prediction": failure_mode,
            "probabilities": mode_probabilities.iloc[0].to_dict(),
        },
        "explanation": explanation,
        "maintenance": recommendation,
        "rul_status": "PENDING_MODEL_TRAINING",
    }


if __name__ == "__main__":
    result = run_pipeline(
        machine_id="TEST-MACHINE-001",
        type_="M",
        air_temp_k=300.0,
        process_temp_k=310.0,
        rot_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=100.0,
    )

    print("\nPREDICTIVE MAINTENANCE PIPELINE RESULT")
    print("=" * 55)

    for section, values in result.items():
        print(f"\n{section.upper()}:")
        print(values)