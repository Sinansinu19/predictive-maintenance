import json
import os
import joblib
import numpy as np
import pandas as pd
import shap


def generate_vehicle_shap():
    data_path = "data/processed/vehicle_fusion/master_vehicle_data.csv"
    model_path = "models/vehicle/vehicle_failure_model.joblib"
    metrics_path = "models/vehicle/vehicle_metrics.json"
    output_path = "models/vehicle/vehicle_shap_summary.json"

    model = joblib.load(model_path)
    with open(metrics_path, "r") as f:
        feature_cols = json.load(f)["feature_columns"]

    df = pd.read_csv(data_path)
    X = pd.get_dummies(
        df[["vehicle_type", "sensor_rpm", "sensor_speed", "sensor_temp"]],
        columns=["vehicle_type"],
        drop_first=False,
    )
    # Ensure exact column order and numeric types for SHAP
    X = X.reindex(columns=feature_cols, fill_value=0).astype(float)

    # Sample up to 1,000 rows per vehicle type for fast, balanced SHAP computation
    truck_idx = df[df["vehicle_type"] == "heavy_truck"].sample(n=min(1000, (df["vehicle_type"] == "heavy_truck").sum()), random_state=42).index
    car_idx = df[df["vehicle_type"] == "passenger_car"].sample(n=min(1000, (df["vehicle_type"] == "passenger_car").sum()), random_state=42).index
    sample_idx = truck_idx.union(car_idx)

    X_sample = X.loc[sample_idx]
    types_sample = df.loc[sample_idx, "vehicle_type"]

    print("Computing SHAP values for multi-vehicle model...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    def summarize(vals):
        mean_abs = np.abs(vals).mean(axis=0)
        return {
            col: round(float(val), 4)
            for col, val in sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)
        }

    summary = {
        "overall_feature_importance": summarize(shap_values),
        "heavy_truck_importance": summarize(shap_values[types_sample.values == "heavy_truck"]),
        "passenger_car_importance": summarize(shap_values[types_sample.values == "passenger_car"]),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSHAP summary saved to: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    generate_vehicle_shap()