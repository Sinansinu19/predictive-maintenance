import os
import pandas as pd
import numpy as np

def build_fused_dataset(truck_path, car_path, output_path):
    print("Loading isolated vehicle datasets...")
    truck_df = pd.read_csv(truck_path, na_values=["na", "NaN", ""])
    car_df = pd.read_csv(car_path)

    # 1. Standardize Scania Truck Data (anonymized sensors mapped to unified schema)
    truck_clean = pd.DataFrame({
        "vehicle_type": "heavy_truck",
        "sensor_rpm": pd.to_numeric(truck_df["aa_000"], errors="coerce").fillna(0),
        "sensor_speed": pd.to_numeric(truck_df["ac_000"], errors="coerce").fillna(0),
        "sensor_temp": pd.to_numeric(truck_df["ad_000"], errors="coerce").fillna(0),
        "vehicle_failure": (truck_df["class"] == "pos").astype(int),
    })

    # 2. Standardize Car OBD-II Data
    rpm = pd.to_numeric(car_df["ENGINE_RPM ()"], errors="coerce").fillna(0)
    speed = pd.to_numeric(car_df["VEHICLE_SPEED ()"], errors="coerce").fillna(0)
    temp = pd.to_numeric(car_df["COOLANT_TEMPERATURE ()"], errors="coerce").fillna(0)
    mil_on = pd.to_numeric(car_df["TIME_RUN_WITH_MIL_ON ()"], errors="coerce").fillna(0)

    # Flag failure on top 5% extreme operating stress (RPM or Speed)
    rpm_thresh = rpm.quantile(0.95)
    speed_thresh = speed.quantile(0.95)
    car_failure = ((rpm > rpm_thresh) | (speed > speed_thresh)).astype(int)

    car_clean = pd.DataFrame({
        "vehicle_type": "passenger_car",
        "sensor_rpm": rpm,
        "sensor_speed": speed,
        "sensor_temp": temp,
        "vehicle_failure": car_failure,
    })

    # 3. Fuse and save strictly to the isolated vehicle_fusion folder
    fused_df = pd.concat([truck_clean, car_clean], ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fused_df.to_csv(output_path, index=False)

    print(f"\nFusion complete! Saved to: {output_path}")
    print(f"Total rows: {len(fused_df)}")
    print("\nBreakdown by vehicle_type and vehicle_failure:")
    print(fused_df.groupby(["vehicle_type", "vehicle_failure"]).size())

if __name__ == "__main__":
    TRUCK_FILE = "data/raw/vehicle_fusion/aps_failure_training_set.csv"
    CAR_FILE = "data/raw/vehicle_fusion/live1.csv"
    OUTPUT_FILE = "data/processed/vehicle_fusion/master_vehicle_data.csv"

    build_fused_dataset(TRUCK_FILE, CAR_FILE, OUTPUT_FILE)