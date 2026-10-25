"""Simulate live machine sensor readings and send them to the prediction API."""

import time
from typing import Any

import pandas as pd
import requests

from src import config
from src.environment import apply_weather_variance


def create_payload(row: pd.Series) -> dict[str, Any]:
    """Convert one dataset row into the eight-feature API payload."""
    payload: dict[str, Any] = {}

    for column in config.FEATURE_COLUMNS:
        payload[column] = float(row[column])

    return payload


def send_sensor_reading(
    session: requests.Session,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Send one sensor reading and return the API response, if successful."""
    url = f"http://{config.API_HOST}:{config.API_PORT}/predict"

    try:
        response = session.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        print(f"API response: {result}")
        return result
    except requests.RequestException as exc:
        print(f"API request failed: {exc}")
    except ValueError as exc:
        print(f"API returned invalid JSON: {exc}")

    return None


def run_simulator(interval_seconds: float = 2.0) -> None:
    """Stream dataset rows sequentially to the prediction API."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")

    if not config.TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {config.TEST_PATH}")

    data = pd.read_csv(config.TEST_PATH)
    missing = set(config.FEATURE_COLUMNS) - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required features: {sorted(missing)}"
        )

    print(f"Starting simulator with {len(data)} sensor rows.")

    url = f"http://{config.API_HOST}:{config.API_PORT}/predict"
    print(f"API endpoint: {url}")

    with requests.Session() as session:
        for index, (_, row) in enumerate(data.iterrows(), start=1):
            sensor_row = apply_weather_variance(row.to_dict())
            payload = create_payload(pd.Series(sensor_row))

            print(f"Sending sensor reading {index}/{len(data)}")
            send_sensor_reading(session, payload)

            if index < len(data):
                time.sleep(interval_seconds)

    print("Sensor simulation completed.")


if __name__ == "__main__":
    run_simulator()