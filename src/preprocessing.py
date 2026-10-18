import math
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def load_raw() -> pd.DataFrame:
    """Load the raw AI4I 2020 dataset from the configured raw data path."""
    if not config.AI4I_PATH.exists():
        raise FileNotFoundError(
            f"Missing required raw data file at: {config.AI4I_PATH}. "
            "Please download UCI dataset 601 and place it as data/raw/ai4i2020.csv"
        )
    return pd.read_csv(config.AI4I_PATH)


def load_ai4i() -> pd.DataFrame:
    """Load AI4I 2020 dataset with snake_case renamed columns."""
    df_raw = load_raw()
    return df_raw.rename(columns=config.RAW_RENAME)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, derive engineered features, and assign failure modes without dropping rows."""
    if any(col in df.columns for col in config.RAW_RENAME.keys()):
        df_clean = df.rename(columns=config.RAW_RENAME).copy()
    else:
        df_clean = df.copy()

    df_clean["type_encoded"] = df_clean["type"].map(config.TYPE_MAP)
    df_clean["temp_diff_k"] = df_clean["process_temp_k"] - df_clean["air_temp_k"]
    df_clean["power_w"] = (
        df_clean["torque_nm"] * df_clean["rot_speed_rpm"] * 2.0 * math.pi / 60.0
    )

    # Label failure_mode: priority TWF -> HDF -> PWF -> OSF -> RNF; if no flag -> NONE
    failure_modes: list[str] = []
    for _, row in df_clean.iterrows():
        if row[config.TARGET_FAILURE] == 0:
            failure_modes.append("NONE")
        else:
            assigned = None
            for flag in config.FLAG_PRIORITY:
                if row[flag] == 1:
                    assigned = flag.upper()
                    break
            failure_modes.append(assigned if assigned is not None else "NONE")

    df_clean[config.TARGET_MODE] = failure_modes
    return df_clean


def split_and_save() -> None:
    """Preprocess data and save stratified train/test splits retaining all 10,000 rows."""
    df_raw = load_raw()
    df_processed = preprocess(df_raw)

    train_df, test_df = train_test_split(
        df_processed,
        test_size=config.TEST_SIZE,
        stratify=df_processed[config.TARGET_FAILURE],
        random_state=config.RANDOM_STATE,
    )

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)

    print(f"Total rows kept: {len(df_processed)} (0 dropped)")
    print(f"Saved {len(train_df)} train rows to {config.TRAIN_PATH}")
    print(f"Saved {len(test_df)} test rows to {config.TEST_PATH}")


if __name__ == "__main__":
    split_and_save()