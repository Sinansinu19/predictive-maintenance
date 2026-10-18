import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
from src import config
from src.features import build_features


def train_failure_model() -> dict:
    """Train class-weighted XGBClassifier on train features and evaluate metrics."""
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    X_train = build_features(train_df)
    y_train = train_df[config.TARGET_FAILURE].astype(int)

    X_test = build_features(test_df)
    y_test = test_df[config.TARGET_FAILURE].astype(int)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=100,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    # Metrics at standard threshold 0.5
    preds_05 = (proba >= 0.5).astype(int)
    precision_05 = float(precision_score(y_test, preds_05, zero_division=0))
    recall_05 = float(recall_score(y_test, preds_05, zero_division=0))
    f1_05 = float(f1_score(y_test, preds_05, zero_division=0))

    precision_arr, recall_arr, _ = precision_recall_curve(y_test, proba)
    pr_auc = float(auc(recall_arr, precision_arr))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.FAILURE_MODEL_PATH)

    print("NOTE: AI4I 2020 is a synthetic dataset. Scores represent dataset results only.")
    print(f"Saved failure model to {config.FAILURE_MODEL_PATH}")

    return {
        "precision_at_0.5": precision_05,
        "recall_at_0.5": recall_05,
        "f1_at_0.5": f1_05,
        "pr_auc": pr_auc,
    }


def load_failure_model():
    """Load the trained XGBoost failure model from disk."""
    if not config.FAILURE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Failure model not found at {config.FAILURE_MODEL_PATH}. "
            "Run 'python -m src.failure_model' first."
        )
    return joblib.load(config.FAILURE_MODEL_PATH)


def predict_failure_proba(X: pd.DataFrame) -> np.ndarray:
    """Predict failure probabilities using the trained model, returning a 1D float array in [0, 1]."""
    model = load_failure_model()
    features = build_features(X)
    proba = model.predict_proba(features)[:, 1]
    return np.asarray(proba, dtype=float)


if __name__ == "__main__":
    metrics = train_failure_model()
    print("Test metrics computed (threshold 0.5):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")