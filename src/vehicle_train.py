import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def find_best_threshold(y_true, y_prob, min_recall=0.80):
    best_thresh = 0.50
    best_f1 = -1.0

    for thresh in np.arange(0.50, 0.96, 0.01):
        preds = (y_prob >= thresh).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if rec >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_thresh = float(round(thresh, 2))

    # Fallback: pick highest F1 overall if min_recall constraint isn't met
    if best_f1 < 0:
        for thresh in np.arange(0.50, 0.96, 0.01):
            preds = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(round(thresh, 2))

    return best_thresh


def train_vehicle_model():
    data_path = "data/processed/vehicle_fusion/master_vehicle_data.csv"
    model_dir = "models/vehicle"
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    X = pd.get_dummies(
        df[["vehicle_type", "sensor_rpm", "sensor_speed", "sensor_temp"]],
        columns=["vehicle_type"],
        drop_first=False,
    )
    y = df["vehicle_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / pos_count)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    print("Training isolated multi-vehicle XGBoost model...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    best_thresh = find_best_threshold(y_test, y_prob, min_recall=0.80)
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "best_threshold": best_thresh,
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "feature_columns": list(X.columns),
    }

    model_path = os.path.join(model_dir, "vehicle_failure_model.joblib")
    metrics_path = os.path.join(model_dir, "vehicle_metrics.json")
    threshold_path = os.path.join(model_dir, "vehicle_threshold.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(threshold_path, "w") as f:
        json.dump({"threshold": best_thresh}, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Threshold saved to: {threshold_path}")
    print("\nTuned Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_vehicle_model()