"""Simulate live machine readings and send them to the prediction API."""

import time
from typing import Any

import pandas as pd
import requests

from src import config
from src.environment import apply_weather_variance


# Sinan's API is running on port 8001.
API_URL = "http://127.0.0.1:8001/predict"

RAW_INPUT_FIELDS = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


def create_payload(
    row: dict[str, Any],
    machine_id: str,
) -> dict[str, Any]:
    """Create the nested request payload expected by the API."""

    inputs = {
        field: row[field]
        for field in RAW_INPUT_FIELDS
    }

    for field in RAW_INPUT_FIELDS[1:]:
        inputs[field] = float(inputs[field])

    return {
        "machine_id": machine_id,
        "inputs": inputs,
    }


def send_sensor_reading(
    session: requests.Session,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Send one simulated sensor reading to the prediction API."""

    try:
        response = session.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

        print("\nAPI RESPONSE")
        print("-" * 40)
        print(f"Sequence: {result.get('seq')}")
        print(f"Machine ID: {result.get('machine_id')}")
        print(f"Failure Probability: {result.get('probability')}")
        print(f"Threshold: {result.get('threshold')}")
        print(f"Failure Alert: {result.get('trigger_alert')}")
        print(f"Anomaly Detected: {result.get('anomaly_flag')}")
        print(f"Anomaly Score: {result.get('anomaly_score')}")
        print(f"Temperature Difference: {result.get('temp_diff_k')}")
        print(f"Power: {result.get('power_w')}")

        return result

    except requests.RequestException as exc:
        print(f"API request failed: {exc}")

    except ValueError as exc:
        print(f"Invalid API response: {exc}")

    return None


def run_simulator(interval_seconds: float = 2.0) -> None:
    """Replay dataset rows as simulated live machine readings."""

    if interval_seconds <= 0:
        raise ValueError("Interval must be greater than zero.")

    if not config.TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {config.TEST_PATH}"
        )

    data = pd.read_csv(config.TEST_PATH)

    missing = set(RAW_INPUT_FIELDS) - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    data = data.dropna(subset=RAW_INPUT_FIELDS)

    print(f"Loaded {len(data)} machine readings.")
    print(f"Sending readings to: {API_URL}")
    print("Press CTRL+C to stop the simulator.")

    try:
        with requests.Session() as session:

            for index, (_, row) in enumerate(
                data.iterrows(),
                start=1,
            ):
                sensor_row = apply_weather_variance(row.to_dict())

                payload = create_payload(
                    sensor_row,
                    machine_id="MACHINE_001",
                )

                print(f"\nReading {index}/{len(data)}")

                send_sensor_reading(session, payload)

                if index < len(data):
                    time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nSimulator stopped by user.")

    print("\nSimulation completed.")


if __name__ == "__main__":
    run_simulator()