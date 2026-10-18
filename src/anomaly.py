import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src import config
from src.features import build_features


def train_anomaly_model() -> IsolationForest:
    """Train an IsolationForest model on train features and save to disk."""
    train_df = pd.read_csv(config.TRAIN_PATH)
    X_train = build_features(train_df)

    model = IsolationForest(
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=config.RANDOM_STATE,
        n_estimators=100,
    )
    model.fit(X_train)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.ANOMALY_MODEL_PATH)
    print(f"Saved anomaly model to {config.ANOMALY_MODEL_PATH}")
    return model


def load_anomaly_model() -> IsolationForest:
    """Load the trained IsolationForest model from disk."""
    if not config.ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly model not found at {config.ANOMALY_MODEL_PATH}. "
            "Run 'python -m src.anomaly' first."
        )
    return joblib.load(config.ANOMALY_MODEL_PATH)


def detect_anomalies(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies in input features.

    Returns:
        flags: boolean array where True indicates an anomaly.
        scores: float array where higher values indicate higher anomaly likelihood.
    """
    model = load_anomaly_model()
    features = build_features(X)

    preds = model.predict(features)
    flags = preds == -1

    # Invert score_samples so higher means more anomalous
    scores = -model.score_samples(features)
    return flags, scores


if __name__ == "__main__":
    train_anomaly_model()

    # Evaluation against ground truth failure labels
    train_df = pd.read_csv(config.TRAIN_PATH)
    flags, scores = detect_anomalies(train_df)

    actual_failures = train_df[config.TARGET_FAILURE] == 1
    normal_rows = train_df[config.TARGET_FAILURE] == 0

    failure_capture_rate = (
        (actual_failures & flags).sum() / actual_failures.sum()
        if actual_failures.sum() > 0
        else 0.0
    )
    normal_flag_rate = (
        (normal_rows & flags).sum() / normal_rows.sum()
        if normal_rows.sum() > 0
        else 0.0
    )

    print("\n--- Anomaly Model Diagnostics ---")
    print(f"Isolation Forest flagged {failure_capture_rate:.2%} of actual machine failures.")
    print(f"Isolation Forest flagged {normal_flag_rate:.2%} of normal operating rows.")
    print(f"Sample flags shape: {flags.shape}, dtype: {flags.dtype}")
    print(f"Sample scores shape: {scores.shape}, dtype: {scores.dtype}")