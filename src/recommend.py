"""Maintenance recommendation logic for predictive maintenance."""

from typing import Any


def recommend_maintenance(
    probability: float,
    anomaly_flag: bool,
    failure_mode: str | None = None,
    threshold: float = 0.04,
) -> dict[str, Any]:
    """Return a maintenance recommendation using the active threshold."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    trigger_alert = probability >= threshold

    if trigger_alert:
        priority = "URGENT"
        action = (
            "Trigger an inspection alert and ask maintenance "
            "personnel to assess the machine."
        )

    elif anomaly_flag:
        priority = "INVESTIGATE"
        action = (
            "Inspect the sensor readings and machine operating "
            "conditions before deciding on maintenance."
        )

    else:
        priority = "NORMAL"
        action = (
            "Continue routine monitoring and follow the scheduled "
            "preventive maintenance plan."
        )

    if failure_mode:
        mode_text = failure_mode.strip()

        if mode_text and mode_text.upper() != "NONE":
            action += f" Review the reported failure mode: {mode_text}."

    return {
        "failure_probability": round(probability, 4),
        "anomaly_flag": bool(anomaly_flag),
        "threshold": threshold,
        "trigger_alert": trigger_alert,
        "priority": priority,
        "recommendation": action,
    }