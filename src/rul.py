"""Remaining Useful Life (RUL) utilities for NASA C-MAPSS FD001."""

from typing import Any

import numpy as np
import pandas as pd

from src import config


def load_cmapss_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load FD001 training, test, and test-set RUL data."""
    data_dir = config.CMAPSS_DIR

    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    for path in (train_path, test_path, rul_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required C-MAPSS file not found: {path}")

    train = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=config.CMAPSS_COLUMNS,
    )

    test = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=config.CMAPSS_COLUMNS,
    )

    test_rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    if len(test_rul) != test["unit"].nunique():
        raise ValueError(
            "RUL_FD001.txt must contain one RUL value per test engine."
        )

    return train, test, test_rul


def add_rul_target(
    train: pd.DataFrame,
    cap: int | None = None,
) -> pd.DataFrame:
    """Calculate remaining useful life for each training cycle."""
    required = {"unit", "cycle"}
    missing = required - set(train.columns)

    if missing:
        raise ValueError(
            f"Training data is missing columns: {sorted(missing)}"
        )

    result = train.copy()
    last_cycle = result.groupby("unit")["cycle"].transform("max")
    result["RUL"] = last_cycle - result["cycle"]

    if cap is None:
        cap = config.RUL_CAP

    if cap is not None:
        if cap <= 0:
            raise ValueError("RUL cap must be positive.")

        result["RUL"] = result["RUL"].clip(upper=cap)

    return result


def build_cycle_features(
    data: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Create rolling sensor features for each engine."""
    required = {"unit", "cycle"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Input data is missing columns: {sorted(missing)}"
        )

    if window is None:
        window = config.RUL_WINDOW

    if window < 1:
        raise ValueError("Rolling window must be at least 1.")

    result = data.copy()
    result = result.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    sensor_columns = [
        column
        for column in result.columns
        if column.startswith("sensor_")
    ]

    if not sensor_columns:
        raise ValueError("No sensor columns were found in the input data.")

    for column in sensor_columns:
        result[f"{column}_rolling_mean"] = (
            result.groupby("unit", sort=False)[column]
            .transform(
                lambda values: values.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
        )

    return result


def predict_rul_bounds(
    lower: float,
    median: float,
    upper: float,
) -> dict[str, Any]:
    """Format lower, median, and upper RUL estimates."""
    values = np.asarray([lower, median, upper], dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("RUL estimates must be finite numbers.")

    values = np.clip(values, 0, None)
    values.sort()

    return {
        "rul_low": float(values[0]),
        "rul_median": float(values[1]),
        "rul_high": float(values[2]),
        "rul_unit": "cycles",
    }


def get_rul_status(median_rul: float) -> str:
    """Classify maintenance urgency using configured thresholds."""
    if not np.isfinite(median_rul) or median_rul < 0:
        raise ValueError(
            "Median RUL must be a finite, non-negative number."
        )

    if median_rul <= config.RUL_URGENT:
        return "URGENT"

    if median_rul <= config.RUL_PLAN:
        return "PLAN_MAINTENANCE"

    return "NORMAL"


if __name__ == "__main__":
    train_data, test_data, test_rul = load_cmapss_data()

    train_with_rul = add_rul_target(train_data)
    test_features = build_cycle_features(test_data)

    print(f"Training rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Test engines: {test_data['unit'].nunique()}")
    print(f"Test RUL labels: {len(test_rul)}")
    print(
        f"Training RUL range: "
        f"{train_with_rul['RUL'].min()}–{train_with_rul['RUL'].max()}"
    )
    print(f"Test feature rows: {len(test_features)}")
    print(f"Example RUL status for 20 cycles: {get_rul_status(20)}")