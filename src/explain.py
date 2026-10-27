"""Explain machine-failure predictions using SHAP."""

from typing import Any

import numpy as np
import pandas as pd
import shap

from src import config


def explain_prediction(
    model: Any,
    features: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Explain one prediction using SHAP TreeExplainer."""
    if features.empty:
        raise ValueError("Features cannot be empty.")

    if len(features) != 1:
        raise ValueError("Provide exactly one row to explain.")

    if feature_names is None:
        feature_names = list(features.columns)

    if len(feature_names) != features.shape[1]:
        raise ValueError(
            "The number of feature names must match the input columns."
        )

    explainer = shap.TreeExplainer(model)
    explanation = explainer(features)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        # Binary classifier: explain the positive class.
        values = values[0, :, -1]
    elif values.ndim == 2:
        values = values[0]
    elif values.ndim != 1:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    contributions = {
        name: float(value)
        for name, value in zip(feature_names, values)
    }

    ranked = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return {
        "base_value": float(base_values.reshape(-1)[-1]),
        "feature_contributions": contributions,
        "top_features": [
            {"feature": name, "shap_value": value}
            for name, value in ranked[:5]
        ],
        "explanation_space": "model_output",
    }


def explain_failure_model(
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Explain one prediction from the saved machine-failure model."""
    if not config.FAILURE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Failure model not found: {config.FAILURE_MODEL_PATH}"
        )

    import joblib

    expected_features = list(config.FEATURE_COLUMNS)
    missing = set(expected_features) - set(features.columns)

    if missing:
        raise ValueError(
            f"Missing required model features: {sorted(missing)}"
        )

    model = joblib.load(config.FAILURE_MODEL_PATH)

    return explain_prediction(
        model=model,
        features=features[expected_features].iloc[[0]],
        feature_names=expected_features,
    )