"""Remaining Useful Life (RUL) prediction for NASA C-MAPSS FD001."""

from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import config


# ============================================================
# 1. LOAD NASA C-MAPSS DATASET
# ============================================================

def load_cmapss_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load FD001 training, test and test-set RUL data."""

    data_dir = config.CMAPSS_DIR

    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    for path in (train_path, test_path, rul_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required C-MAPSS file not found: {path}"
            )

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


# ============================================================
# 2. CALCULATE RUL TARGET
# ============================================================

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


# ============================================================
# 3. CREATE ROLLING SENSOR FEATURES
# ============================================================

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
        raise ValueError("No sensor columns were found.")

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


# ============================================================
# 4. FORMAT RUL PREDICTION BOUNDS
# ============================================================

def predict_rul_bounds(
    lower: float,
    median: float,
    upper: float,
) -> dict[str, Any]:
    """Format lower, median and upper RUL estimates."""

    values = np.asarray(
        [lower, median, upper],
        dtype=float,
    )

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


# ============================================================
# 5. CLASSIFY MAINTENANCE URGENCY
# ============================================================

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


# ============================================================
# 6. TRAIN THREE QUANTILE RUL MODELS
# ============================================================

def train_rul_models() -> dict[str, Any]:
    """
    Train three Gradient Boosting quantile regression models.

    Quantiles:
        0.10 = Lower RUL estimate
        0.50 = Median RUL estimate
        0.90 = Upper RUL estimate
    """

    print("\nLoading NASA C-MAPSS FD001 dataset...")

    train, test, test_rul = load_cmapss_data()

    print("Dataset loaded successfully.")

    # Prepare training RUL targets.
    train = add_rul_target(train)

    # Generate rolling sensor features.
    train_features = build_cycle_features(train)

    test_features = build_cycle_features(test)

    # Select the final observed cycle for each test engine.
    test_latest = (
        test_features
        .sort_values(["unit", "cycle"])
        .groupby("unit", sort=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )

    # Select numerical features.
    excluded_columns = {"unit", "RUL"}

    feature_columns = [
        column
        for column in train_features.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            train_features[column]
        )
    ]

    X_train = train_features[feature_columns]

    y_train = train_features["RUL"].astype(float)

    X_test = test_latest[feature_columns]

    # Check feature consistency.
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test features do not match.")

    models = {}

    # Train lower, median and upper quantile models.
    for quantile in config.RUL_QUANTILES:

        print(
            f"\nTraining RUL model for quantile {quantile}..."
        )

        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            random_state=config.RANDOM_STATE,
        )

        model.fit(X_train, y_train)

        models[str(quantile)] = model

        print(
            f"Quantile {quantile} model trained successfully."
        )

    # Generate predictions.
    predictions = {}

    for quantile, model in models.items():
        predictions[quantile] = model.predict(X_test)

    # Save trained models and feature configuration.
    config.MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "models": models,
        "feature_columns": feature_columns,
        "quantiles": config.RUL_QUANTILES,
        "rul_cap": config.RUL_CAP,
    }

    joblib.dump(
        artifact,
        config.RUL_MODEL_PATH,
    )

    print(
        f"\nRUL models saved successfully: {config.RUL_MODEL_PATH}"
    )

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    actual_rul = np.asarray(test_rul, dtype=float)

    median_predictions = np.clip(
        predictions["0.5"],
        0,
        None,
    )

    lower_predictions = np.clip(
        predictions["0.1"],
        0,
        None,
    )

    upper_predictions = np.clip(
        predictions["0.9"],
        0,
        None,
    )

    # Calculate MAE.
    mae = mean_absolute_error(
        actual_rul,
        median_predictions,
    )

    # Calculate RMSE.
    rmse = np.sqrt(
        mean_squared_error(
            actual_rul,
            median_predictions,
        )
    )

    # Calculate prediction interval coverage.
    lower_bound = np.minimum(
        lower_predictions,
        upper_predictions,
    )

    upper_bound = np.maximum(
        lower_predictions,
        upper_predictions,
    )

    coverage = np.mean(
        (actual_rul >= lower_bound)
        & (actual_rul <= upper_bound)
    )

    print("\n========== RUL MODEL EVALUATION ==========")

    print(f"Test engines: {len(actual_rul)}")

    print(f"MAE: {mae:.2f} cycles")

    print(f"RMSE: {rmse:.2f} cycles")

    print(
        f"Prediction interval coverage: {coverage:.2%}"
    )

    print("==========================================")

    return artifact


# ============================================================
# 7. PREDICT RUL FOR AN ENGINE
# ============================================================

def predict_engine_rul(
    engine_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Predict lower, median and upper RUL for an engine.

    engine_data must contain the historical sensor readings
    for one engine.
    """

    if not config.RUL_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "RUL model not found. Train the model first."
        )

    if engine_data.empty:
        raise ValueError("Engine data cannot be empty.")

    if engine_data["unit"].nunique() != 1:
        raise ValueError(
            "Provide sensor readings for exactly one engine."
        )

    # Load saved model.
    artifact = joblib.load(
        config.RUL_MODEL_PATH
    )

    # Generate rolling features.
    features = build_cycle_features(engine_data)

    # Select the latest engine reading.
    latest = (
        features
        .sort_values("cycle")
        .tail(1)
    )

    feature_columns = artifact["feature_columns"]

    X = latest[feature_columns]

    models = artifact["models"]

    # Generate quantile predictions.
    lower = models["0.1"].predict(X)[0]

    median = models["0.5"].predict(X)[0]

    upper = models["0.9"].predict(X)[0]

    # Format prediction bounds.
    result = predict_rul_bounds(
        lower,
        median,
        upper,
    )

    # Determine maintenance status.
    result["maintenance_status"] = get_rul_status(
        result["rul_median"]
    )

    return result


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    train_rul_models()