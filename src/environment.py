"""Environmental variation utilities for the predictive maintenance simulator."""

import random
from typing import Any

from src import config


def apply_weather_variance(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a sensor row with small simulated temperature variation."""
    updated_row = row.copy()

    # Keep the original row unchanged.
    variation = random.uniform(-0.5, 0.5)
    updated_row["air_temp_k"] = float(updated_row["air_temp_k"]) + variation

    # Recompute temperature difference using the shared feature definitions.
    updated_row["temp_diff_k"] = (
        float(updated_row["process_temp_k"]) - updated_row["air_temp_k"]
    )

    return updated_row


if __name__ == "__main__":
    print("Environment module loaded.")