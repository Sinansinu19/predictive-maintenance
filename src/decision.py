import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
from src import config
from src.features import build_features
from src.failure_model import predict_failure_proba


def optimal_threshold(
    y_true, proba, cost_downtime: float, cost_inspection: float
) -> float:
    """
    Minimise total operational loss across candidate thresholds in [0, 1].
    Includes inspection cost for both False Positives and True Positives.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    proba_arr = np.asarray(proba, dtype=float)

    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    min_cost = float("inf")

    for thresh in thresholds:
        preds = (proba_arr >= thresh).astype(int)
        fn = int(np.sum((y_true_arr == 1) & (preds == 0)))
        fp = int(np.sum((y_true_arr == 0) & (preds == 1)))
        tp = int(np.sum((y_true_arr == 1) & (preds == 1)))

        # Assumed costs in $: downtime penalty + inspections conducted
        total_cost = (fn * cost_downtime) + ((fp + tp) * cost_inspection)

        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = float(thresh)

    return float(best_threshold)


def save_threshold(value: float) -> None:
    """Write the decision threshold to THRESHOLD_PATH as a JSON dictionary."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": float(value)}
    with open(config.THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_threshold() -> float:
    """Load the decision threshold from disk, defaulting to 0.5 if missing."""
    if not config.THRESHOLD_PATH.exists():
        return 0.5
    try:
        with open(config.THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("threshold", 0.5))
    except Exception:
        return 0.5


def tune_threshold_cv() -> tuple[float, list[float]]:
    """Tune optimal threshold via 5-fold CV on the train set only, returning threshold and fold F1 scores."""
    train_df = pd.read_csv(config.TRAIN_PATH)
    X_train = build_features(train_df)
    y_train = train_df[config.TARGET_FAILURE].astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    oof_proba = np.zeros(len(train_df), dtype=float)
    cv_f1_scores: list[float] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

        neg_cnt = int((y_tr == 0).sum())
        pos_cnt = int((y_tr == 1).sum())
        scale_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0

        clf = XGBClassifier(
            scale_pos_weight=scale_weight,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=100,
        )
        clf.fit(X_tr, y_tr)
        val_proba = clf.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba

        # Compute validation fold F1 at default 0.5
        fold_pred = (val_proba >= 0.5).astype(int)
        cv_f1_scores.append(float(f1_score(y_val, fold_pred, zero_division=0)))

    best_thresh = optimal_threshold(
        y_true=y_train,
        proba=oof_proba,
        cost_downtime=config.COST_DOWNTIME,
        cost_inspection=config.COST_INSPECTION,
    )
    return best_thresh, cv_f1_scores


def export_metrics(y_true, proba, tuned_thresh: float, cv_f1_scores: list[float]) -> dict:
    """Save comprehensive evaluation metrics and confusion matrix to models/metrics.json."""
    preds_tuned = (proba >= tuned_thresh).astype(int)
    preds_05 = (proba >= 0.5).astype(int)

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, proba)
    pr_auc_val = float(auc(recall_arr, precision_arr))

    tn, fp, fn, tp = confusion_matrix(y_true, preds_tuned).ravel()

    metrics_payload = {
        "cost_assumed_currency": "USD ($)",
        "cost_downtime": config.COST_DOWNTIME,
        "cost_inspection": config.COST_INSPECTION,
        "cv_5fold_f1_scores": cv_f1_scores,
        "mean_cv_f1": float(np.mean(cv_f1_scores)),
        "tuned_threshold": float(tuned_thresh),
        "test_metrics_at_tuned_threshold": {
            "precision": float(precision_score(y_true, preds_tuned, zero_division=0)),
            "recall": float(recall_score(y_true, preds_tuned, zero_division=0)),
            "f1": float(f1_score(y_true, preds_tuned, zero_division=0)),
            "pr_auc": pr_auc_val,
        },
        "test_metrics_at_default_0_5": {
            "precision": float(precision_score(y_true, preds_05, zero_division=0)),
            "recall": float(recall_score(y_true, preds_05, zero_division=0)),
            "f1": float(f1_score(y_true, preds_05, zero_division=0)),
        },
        "confusion_matrix_at_tuned_threshold": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
    }

    metrics_path = config.MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Detailed metrics exported to: {metrics_path}")
    return metrics_payload


if __name__ == "__main__":
    best_thresh, cv_f1_scores = tune_threshold_cv()
    save_threshold(best_thresh)
    print(f"Optimal threshold tuned via Train 5-Fold CV: {best_thresh:.4f}")

    test_df = pd.read_csv(config.TEST_PATH)
    y_test = test_df[config.TARGET_FAILURE].astype(int).values
    test_proba = predict_failure_proba(test_df)

    export_metrics(
        y_true=y_test,
        proba=test_proba,
        tuned_thresh=best_thresh,
        cv_f1_scores=cv_f1_scores,
    )