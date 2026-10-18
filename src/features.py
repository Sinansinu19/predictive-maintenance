import math
import pandas as pd
from src import config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame containing exactly FEATURE_COLUMNS in the specified order."""
    missing = [col for col in config.FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")
    return df[config.FEATURE_COLUMNS].copy()


def make_input_row(
    type_: str,
    air_temp_k: float,
    process_temp_k: float,
    rot_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> pd.DataFrame:
    """Build a single-row DataFrame from raw sensor inputs, computing all derived features."""
    if type_ not in config.TYPE_MAP:
        raise ValueError(
            f"Invalid product type '{type_}'. Must be one of {list(config.TYPE_MAP.keys())}"
        )

    type_encoded = config.TYPE_MAP[type_]
    temp_diff_k = float(process_temp_k - air_temp_k)
    power_w = float(torque_nm * rot_speed_rpm * 2.0 * math.pi / 60.0)

    data = {
        "type_encoded": [int(type_encoded)],
        "air_temp_k": [float(air_temp_k)],
        "process_temp_k": [float(process_temp_k)],
        "rot_speed_rpm": [float(rot_speed_rpm)],
        "torque_nm": [float(torque_nm)],
        "tool_wear_min": [float(tool_wear_min)],
        "temp_diff_k": [temp_diff_k],
        "power_w": [power_w],
    }
    row_df = pd.DataFrame(data)
    return row_df[config.FEATURE_COLUMNS]


if __name__ == "__main__":
    train_df = pd.read_csv(config.TRAIN_PATH)
    X_train = build_features(train_df)
    print(f"build_features shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")