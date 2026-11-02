
"""Train and evaluate the NASA C-MAPSS FD001 RUL model."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from src import config
from src.rul import (
    add_rul_target,
    build_cycle_features,
    load_cmapss_data,
)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create selected sensor and operating-condition features."""

    rolling = build_cycle_features(
        data,
        window=config.RUL_WINDOW,
    )

    # Exclude constant sensors identified in FD001 training data.
    selected_sensors = [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_6",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_17",
        "sensor_20",
        "sensor_21",
    ]

    feature_columns = (
        ["cycle", "op_1", "op_2", "op_3"]
        + selected_sensors
        + [
            f"{sensor}_rolling_mean"
            for sensor in selected_sensors
        ]
    )

    return rolling[feature_columns].astype(float)


def train_rul_model() -> None:
    """Train, evaluate and save the FD001 RUL regression model."""

    train, test, actual_test_rul = load_cmapss_data()

    # Calculate capped RUL targets for training data.
    train = add_rul_target(
        train,
        cap=config.RUL_CAP,
    )

    X = prepare_features(train)
    y = train["RUL"].astype(float)
    groups = train["unit"]

    # Split by engine to prevent engine-level data leakage.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
    )

    train_indices, validation_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_validation = X.iloc[validation_indices]

    y_train = y.iloc[train_indices]
    y_validation = y.iloc[validation_indices]

    # Initialize XGBoost regression model.
    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training NASA C-MAPSS FD001 RUL model...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    # Validation evaluation.
    validation_predictions = np.clip(
        model.predict(X_validation),
        0,
        config.RUL_CAP,
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_predictions,
        )
    )

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    print("\nVALIDATION RESULTS")
    print("------------------")
    print(f"RMSE: {validation_rmse:.3f}")
    print(f"MAE:  {validation_mae:.3f}")

    # --------------------------------------------------
    # OFFICIAL TEST EVALUATION
    # --------------------------------------------------

    # Step 1: Generate rolling features using complete
    # sensor histories for all test engines.
    test_features = prepare_features(test)

    # Step 2: Keep engine IDs aligned with sorted features.
    sorted_test = (
        test.sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    if len(test_features) != len(sorted_test):
        raise ValueError(
            "Test features and original test rows do not match."
        )

    test_feature_data = test_features.copy()

    test_feature_data.insert(
        0,
        "unit",
        sorted_test["unit"].to_numpy(),
    )

    # Step 3: Select the final cycle of each engine
    # AFTER calculating rolling sensor features.
    test_last = (
        test_feature_data
        .groupby("unit", sort=True, group_keys=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )

    # Step 4: Select exactly the same features used in training.
    X_test = test_last[X.columns]

    # Verify one prediction per test engine.
    if len(X_test) != len(actual_test_rul):
        raise ValueError(
            "Test predictions and actual RUL labels do not match."
        )

    # Step 5: Predict remaining useful life.
    test_predictions = np.clip(
        model.predict(X_test),
        0,
        config.RUL_CAP,
    )

    # Official FD001 test evaluation.
    test_rmse = np.sqrt(
        mean_squared_error(
            actual_test_rul,
            test_predictions,
        )
    )

    test_mae = mean_absolute_error(
        actual_test_rul,
        test_predictions,
    )

    print("\nOFFICIAL FD001 TEST RESULTS")
    print("---------------------------")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"MAE:  {test_mae:.3f}")

    # --------------------------------------------------
    # SAVE MODEL AND METRICS
    # --------------------------------------------------

    config.MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "rul_cap": config.RUL_CAP,
        "rolling_window": config.RUL_WINDOW,
        "dataset": "NASA C-MAPSS FD001",
        "validation_rmse": float(validation_rmse),
        "validation_mae": float(validation_mae),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
    }

    joblib.dump(
        artifact,
        config.RUL_MODEL_PATH,
    )

    metrics_path = config.MODELS_DIR / "rul_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                key: value
                for key, value in artifact.items()
                if key not in {
                    "model",
                    "feature_columns",
                }
            },
            file,
            indent=4,
        )

    print("\nMODEL SAVED")
    print("------------")
    print(f"Model: {config.RUL_MODEL_PATH}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    train_rul_model()
