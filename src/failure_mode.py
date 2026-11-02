"""
Multiclass machine failure-mode classification for the AI4I dataset.

Member 2 module:
- Prepare failure-mode training data.
- Train and evaluate a Random Forest classifier.
- Handle class imbalance.
- Save and load the trained model.
- Predict failure modes and class probabilities.
"""

from typing import Any

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

from src import config


# ============================================================
# 1. VALIDATE INPUT COLUMNS
# ============================================================

def _validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Check whether all required columns exist."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ============================================================
# 2. PREPARE FAILURE-MODE DATASET
# ============================================================

def _prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """
    Prepare model features and failure-mode labels.

    Rows marked as machine failures without an individual
    failure-mode flag are excluded.
    """

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

    # Identify rows having at least one failure-mode flag.
    has_failure_flag = (
        data[config.FLAG_PRIORITY]
        .eq(1)
        .any(axis=1)
    )

    # Identify machine failures without a failure-mode flag.
    invalid_failure_rows = (
        data[config.TARGET_FAILURE].eq(1)
        & ~has_failure_flag
    )

    dropped_count = int(
        invalid_failure_rows.sum()
    )

    # Remove invalid failure-mode records.
    data = data.loc[
        ~invalid_failure_rows
    ].copy()

    # Select exactly the eight approved features.
    X = data.loc[
        :,
        config.FEATURE_COLUMNS
    ].copy()

    # Prepare classification labels.
    y = (
        data[config.TARGET_MODE]
        .astype(str)
        .str.upper()
    )

    # Check allowed failure-mode classes.
    allowed_classes = set(config.MODE_CLASSES)

    unexpected = sorted(
        set(y.unique()) - allowed_classes
    )

    if unexpected:
        raise ValueError(
            f"Unexpected failure-mode labels: {unexpected}. "
            f"Allowed labels: {config.MODE_CLASSES}"
        )

    if X.empty:
        raise ValueError(
            "No valid training rows remain."
        )

    if X.isna().any().any():
        raise ValueError(
            "Feature data contains missing values."
        )

    return X, y, dropped_count


# ============================================================
# 3. LOAD SAVED FAILURE-MODE MODEL
# ============================================================

def _load_saved_model() -> RandomForestClassifier:
    """Load the saved Random Forest failure-mode model."""

    model_path = config.FAILURE_MODE_MODEL_PATH

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Failure-mode model not found: {model_path}. "
            "Run python -m src.failure_mode first."
        )

    model = joblib.load(model_path)

    if not isinstance(
        model,
        RandomForestClassifier
    ):
        raise TypeError(
            "Saved model is not a RandomForestClassifier."
        )

    return model


# ============================================================
# 4. TRAIN AND EVALUATE FAILURE-MODE MODEL
# ============================================================

def train_failure_mode_model() -> dict[str, Any]:
    """
    Train, evaluate and save the multiclass classifier.
    """

    train_path = config.TRAIN_PATH
    test_path = config.TEST_PATH

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Testing dataset not found: {test_path}"
        )

    # Load processed datasets.
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Prepare training and testing data.
    X_train, y_train, dropped_train = (
        _prepare_dataset(train_df)
    )

    X_test, y_test, dropped_test = (
        _prepare_dataset(test_df)
    )

    print("\nFAILURE-MODE DATASET")
    print("-" * 50)

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    print(
        "Invalid training rows removed:",
        dropped_train
    )

    print(
        "Invalid testing rows removed:",
        dropped_test
    )

    # Calculate balanced sample weights.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # Initialize Random Forest classifier.
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest classifier...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    print("Training completed successfully.")

    # Generate predictions.
    y_pred = model.predict(X_test)

    # Calculate precision and recall for all contract classes.
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

    per_class_metrics = {}

    print("\nFAILURE-MODE CLASSIFICATION RESULTS")
    print("-" * 60)

    for index, mode in enumerate(config.MODE_CLASSES):

        precision = float(
            precision_values[index]
        )

        recall = float(
            recall_values[index]
        )

        per_class_metrics[mode] = {
            "precision": precision,
            "recall": recall,
        }

        print(
            f"{mode:>5} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )

    # Save trained model.
    model_path = config.FAILURE_MODE_MODEL_PATH

    config.MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    print("\nMODEL SAVED SUCCESSFULLY")
    print(model_path)

    return {
        "per_class_metrics": per_class_metrics,
        "dropped_training_rows": dropped_train,
        "dropped_testing_rows": dropped_test,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "trained_classes": [
            str(label)
            for label in model.classes_
        ],
        "model_path": str(model_path),
    }


# ============================================================
# 5. PREDICT FAILURE MODES
# ============================================================

def predict_failure_mode(
    X: pd.DataFrame,
) -> list[str]:
    """Predict one failure-mode label for every input row."""

    if X.empty:
        return []

    _validate_columns(
        X,
        list(config.FEATURE_COLUMNS),
    )

    features = X.loc[
        :,
        config.FEATURE_COLUMNS
    ]

    if features.isna().any().any():
        raise ValueError(
            "Input features contain missing values."
        )

    probabilities = predict_failure_mode_proba(
        features
    )

    predictions = probabilities.idxmax(
        axis=1
    )

    return [
        str(label)
        for label in predictions.tolist()
    ]


# ============================================================
# 6. PREDICT FAILURE-MODE PROBABILITIES
# ============================================================

def predict_failure_mode_proba(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return probabilities for all six failure modes.

    Output columns always follow the contract-defined order.
    """

    if X.empty:
        return pd.DataFrame(
            columns=config.MODE_CLASSES,
            index=X.index,
        )

    _validate_columns(
        X,
        list(config.FEATURE_COLUMNS),
    )

    features = X.loc[
        :,
        config.FEATURE_COLUMNS
    ]

    if features.isna().any().any():
        raise ValueError(
            "Input features contain missing values."
        )

    # Load trained model.
    model = _load_saved_model()

    # Generate model probabilities.
    model_probabilities = model.predict_proba(
        features
    )

    # Create output with every contract-defined class.
    probabilities = pd.DataFrame(
        0.0,
        index=X.index,
        columns=config.MODE_CLASSES,
    )

    # Map model classes to contract columns.
    for index, label in enumerate(model.classes_):

        if label in probabilities.columns:

            probabilities[label] = (
                model_probabilities[:, index]
            )

    # Normalize probabilities.
    row_totals = probabilities.sum(
        axis=1
    )

    if (row_totals <= 0).any():
        raise ValueError(
            "Invalid probability values generated."
        )

    probabilities = probabilities.div(
        row_totals,
        axis=0,
    )

    return probabilities.loc[
        :,
        config.MODE_CLASSES
    ]


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FAILURE-MODE CLASSIFICATION")
    print("=" * 60)

    # Train and evaluate model.
    metrics = train_failure_mode_model()

    print("\nEVALUATION SUMMARY")
    print(metrics["per_class_metrics"])

    # Load test dataset.
    test_df = pd.read_csv(
        config.TEST_PATH
    )

    X_test, _, _ = _prepare_dataset(
        test_df
    )

    # Select five test records.
    X_sample = X_test.head(5)

    # Generate failure-mode predictions.
    predictions = predict_failure_mode(
        X_sample
    )

    # Generate class probabilities.
    probabilities = predict_failure_mode_proba(
        X_sample
    )

    print("\nSAMPLE FAILURE-MODE PREDICTIONS")
    print(predictions)

    print("\nFAILURE-MODE PROBABILITIES")
    print(probabilities)

    print("\nPROBABILITY ROW SUMS")
    print(
        probabilities.sum(axis=1).tolist()
    )

    print("\nALL TESTS COMPLETED.")