"""
Multiclass machine failure-mode classification for the AI4I dataset.

Member 2 module: trains, evaluates, saves, and uses the failure-mode model.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

from src import config


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Check Member 1's preprocessing and configuration."
        )


def _prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    """Prepare features and labels while excluding invalid failure-mode rows."""
    required_columns = (
        list(config.FEATURE_COLUMNS)
        + [
            config.TARGET_MODE,
            config.TARGET_FAILURE,
            *config.FLAG_PRIORITY,
        ]
    )
    _validate_columns(df, required_columns)

    data = df.copy()

    # Drop machine failures that have no individual failure-mode flag.
    has_failure_flag = data[config.FLAG_PRIORITY].eq(1).any(axis=1)

    invalid_failure_rows = (
        data[config.TARGET_FAILURE].eq(1)
        & ~has_failure_flag
    )

    dropped_count = int(invalid_failure_rows.sum())

    data = data.loc[~invalid_failure_rows].copy()

    # Features must contain only the eight contract-defined columns.
    X = data.loc[:, config.FEATURE_COLUMNS].copy()
    y = data[config.TARGET_MODE].astype(str).str.upper()

    # Reject unexpected labels instead of silently changing them.
    allowed_classes = set(config.MODE_CLASSES)
    unexpected = sorted(set(y.unique()) - allowed_classes)

    if unexpected:
        raise ValueError(
            f"Unexpected failure-mode labels: {unexpected}. "
            f"Allowed labels are {config.MODE_CLASSES}."
        )

    if X.empty:
        raise ValueError(
            "No valid training rows remain after failure-mode filtering."
        )

    if X.isna().any().any():
        raise ValueError(
            "Feature data contains missing values. "
            "Check Member 1's preprocessing."
        )

    return X, y, dropped_count


def _load_saved_model() -> RandomForestClassifier:
    """Load the trained failure-mode classifier."""
    model_path = config.FAILURE_MODE_MODEL_PATH

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Failure-mode model not found at {model_path}. "
            "Run `python -m src.failure_mode` to train it first."
        )

    model = joblib.load(model_path)

    if not isinstance(model, RandomForestClassifier):
        raise TypeError(
            "The saved failure-mode model is not a "
            "RandomForestClassifier. Check the saved model file."
        )

    return model


def train_failure_mode_model() -> dict[str, Any]:
    """Train, evaluate, and save the multiclass failure-mode model."""
    train_path = config.TRAIN_CSV
    test_path = config.TEST_CSV

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training CSV not found at {train_path}. "
            "Ask Member 1 to run the preprocessing pipeline."
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test CSV not found at {test_path}. "
            "Ask Member 1 to run the preprocessing pipeline."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train, dropped_train = _prepare_dataset(train_df)
    X_test, y_test, dropped_test = _prepare_dataset(test_df)

    print(
        "Rows excluded because machine_failure == 1 "
        "but no failure-mode flag was set:"
    )
    print(f"Training: {dropped_train}")
    print(f"Testing:  {dropped_test}")

    # Balanced sample weights compensate for unequal class frequencies.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    y_pred = model.predict(X_test)

    # Report every contract-defined class, including absent test classes.
    precision_values = precision_score(
        y_test,
        y_pred,
        labels=config.MODE_CLASSES,
        average=None,
        zero_division=0,
    )

    recall_values = recall_score(
        y_test,
        y_pred,
        labels=config.MODE_CLASSES,
        average=None,
        zero_division=0,
    )

    per_class_metrics: dict[str, dict[str, float]] = {}

    print("\nFailure-mode classification results")
    print("-" * 55)

    for index, mode in enumerate(config.MODE_CLASSES):
        precision = float(precision_values[index])
        recall = float(recall_values[index])

        per_class_metrics[mode] = {
            "precision": precision,
            "recall": recall,
        }

        print(
            f"{mode:>5} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )

    model_path = config.FAILURE_MODE_MODEL_PATH
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")
    print(
        "Class imbalance was handled using balanced class weights "
        "and balanced sample weights."
    )

    return {
        "per_class_metrics": per_class_metrics,
        "dropped_training_rows": dropped_train,
        "dropped_testing_rows": dropped_test,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "trained_classes": [
            str(label) for label in model.classes_
        ],
        "model_path": str(model_path),
    }


def predict_failure_mode(X: pd.DataFrame) -> list[str]:
    """Predict one allowed failure-mode label for each input row."""
    if X.empty:
        return []

    _validate_columns(X, list(config.FEATURE_COLUMNS))

    features = X.loc[:, config.FEATURE_COLUMNS]

    if features.isna().any().any():
        raise ValueError("Input features contain missing values.")

    probabilities = predict_failure_mode_proba(features)

    return [
        str(label)
        for label in probabilities.idxmax(axis=1).tolist()
    ]


def predict_failure_mode_proba(X: pd.DataFrame) -> pd.DataFrame:
    """Return probabilities for all failure modes in contract order."""
    if X.empty:
        return pd.DataFrame(columns=config.MODE_CLASSES, index=X.index)

    _validate_columns(X, list(config.FEATURE_COLUMNS))

    features = X.loc[:, config.FEATURE_COLUMNS]

    if features.isna().any().any():
        raise ValueError("Input features contain missing values.")

    model = _load_saved_model()
    model_probabilities = model.predict_proba(features)

    # Initialize every contract class, including classes absent in training.
    probabilities = pd.DataFrame(
        0.0,
        index=X.index,
        columns=config.MODE_CLASSES,
        dtype=float,
    )

    for index, label in enumerate(model.classes_):
        if label in probabilities.columns:
            probabilities[label] = model_probabilities[:, index]

    # Normalize to protect the interface if the model has an unusual output.
    row_totals = probabilities.sum(axis=1)

    if (row_totals <= 0).any():
        raise ValueError(
            "The model produced a row with no valid class probabilities."
        )

    probabilities = probabilities.div(row_totals, axis=0)

    # Ensure the exact public column order required by Contract v3.
    return probabilities.loc[:, config.MODE_CLASSES]


if __name__ == "__main__":
    """Train the model and test both prediction interfaces."""
    metrics = train_failure_mode_model()

    print("\nReturned evaluation metrics:")
    print(metrics["per_class_metrics"])

    test_path = config.TEST_CSV
    test_df = pd.read_csv(test_path)

    X_test, _, _ = _prepare_dataset(test_df)

    # Test a small sample from the real test dataset.
    X_sample = X_test.head(5)

    predicted_modes = predict_failure_mode(X_sample)
    predicted_probabilities = predict_failure_mode_proba(X_sample)

    print("\nSample predictions:")
    print(predicted_modes)

    print("\nSample probabilities:")
    print(predicted_probabilities)

    print("\nProbability row sums:")
    print(predicted_probabilities.sum(axis=1).tolist())