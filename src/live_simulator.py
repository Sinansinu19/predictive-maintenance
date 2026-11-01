"""Simulate live machine readings and send them to the prediction API."""

import time
from typing import Any

import pandas as pd
import requests

from src import config
from src.environment import apply_weather_variance


# Member 1's API is currently running on port 8001.
API_URL = "http://127.0.0.1:8001/predict"

RAW_INPUT_FIELDS = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


def create_payload(row: dict[str, Any], machine_id: str) -> dict[str, Any]:
    """Create the request structure expected by Member 1's API."""

    inputs = {
        field: row[field]
        for field in RAW_INPUT_FIELDS
    }

    # Ensure numeric inputs are JSON-compatible floats.
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
    """Send one simulated sensor reading to the API."""

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
        print(f"Machine ID: {result.get('machine_id')}")
        print(f"Probability: {result.get('probability')}")
        print(f"Threshold: {result.get('threshold')}")
        print(f"Failure alert: {result.get('trigger_alert')}")
        print(f"Anomaly detected: {result.get('anomaly_flag')}")

        return result

    except requests.RequestException as exc:
        print(f"API request failed: {exc}")

    except ValueError as exc:
        print(f"Invalid API response: {exc}")

    return None


def run_simulator(interval_seconds: float = 2.0) -> None:
    """Replay dataset rows as simulated live sensor readings."""

    if interval_seconds <= 0:
        raise ValueError("Interval must be greater than zero.")

    if not config.TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {config.TEST_PATH}"
        )

    data = pd.read_csv(config.TEST_PATH)

    missing = set(RAW_INPUT_FIELDS) - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required inputs: {sorted(missing)}"
        )

    print(f"Loaded {len(data)} machine readings.")
    print(f"Sending readings to: {API_URL}")

    with requests.Session() as session:

        for index, (_, row) in enumerate(data.iterrows(), start=1):

            # Apply small simulated environmental variation.
            sensor_row = apply_weather_variance(row.to_dict())

            # Build the API's expected nested payload.
            payload = create_payload(
                sensor_row,
                machine_id="MACHINE_001",
            )

            print(f"\nReading {index}/{len(data)}")

            send_sensor_reading(session, payload)

            if index < len(data):
                time.sleep(interval_seconds)

    print("\nSimulation completed.")


if __name__ == "__main__":
    run_simulator()