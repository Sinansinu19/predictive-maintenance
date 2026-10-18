from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
CMAPSS_DIR = DATA_RAW_DIR / "cmapss"
STATIC_DIR = BASE_DIR / "app" / "static"

AI4I_PATH = DATA_RAW_DIR / "ai4i2020.csv"
TRAIN_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test.csv"
FAILURE_MODEL_PATH = MODELS_DIR / "failure_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
FAILURE_MODE_MODEL_PATH = MODELS_DIR / "failure_mode_model.joblib"
RUL_MODEL_PATH = MODELS_DIR / "rul_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

# Backwards-compatible aliases
DATA_RAW = DATA_RAW_DIR
DATA_PROCESSED = DATA_PROCESSED_DIR
AI4I_CSV = AI4I_PATH
TRAIN_CSV = TRAIN_PATH
TEST_CSV = TEST_PATH

RAW_RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TYPE_MAP = {"L": 0, "M": 1, "H": 2}
RAW_INPUT_FIELDS = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
FEATURE_COLUMNS = [
    "type_encoded",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_w",
]
LEAKAGE_COLUMNS = [
    "udi",
    "product_id",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "failure_mode",
]
TARGET_FAILURE = "machine_failure"
TARGET_MODE = "failure_mode"

# Spreadsheet Item 3: TWF, HDF, PWF, OSF, RNF order
FLAG_PRIORITY = ["twf", "hdf", "pwf", "osf", "rnf"]

# Spreadsheet Item 2: FAILURE_MODES name
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF", "NONE"]
MODE_CLASSES = FAILURE_MODES

RISK_MEDIUM = 0.30
RISK_HIGH = 0.60
ANOMALY_CONTAMINATION = 0.03

COST_DOWNTIME = 10000.0
COST_INSPECTION = 500.0

CMAPSS_COLUMNS = ["unit", "cycle", "op_1", "op_2", "op_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]
RUL_CAP = 125
RUL_QUANTILES = (0.1, 0.5, 0.9)
RUL_WINDOW = 5
RUL_PLOT_SENSORS = ["sensor_2", "sensor_4", "sensor_7", "sensor_11"]
RUL_URGENT = 30
RUL_PLAN = 75

SCENARIO_DELTAS_K = [-5.0, 0.0, 5.0, 10.0]
AMBIENT_HOT_C = 35.0
HUMIDITY_HIGH_PCT = 80.0
RAIN_HIGH_MM = 1.0
WEATHER_TIMEOUT_S = 5
ENV_DISCLAIMER = (
    "Scenario analysis on a synthetic dataset; "
    "not evidence that weather causes failures."
)

# Optional vehicle dyno test preset (Section 9.3)
VEHICLE_PRESET_RAW = {
    "type": "M",
    "air_temp_k": 305.15,
    "process_temp_k": 368.15,
    "rot_speed_rpm": 2500.0,
    "torque_nm": 180.0,
    "tool_wear_min": 120.0,
}

API_HOST = "127.0.0.1"
API_PORT = 8000

if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Flag Priority: {FLAG_PRIORITY}")
    print(f"Failure Modes: {FAILURE_MODES}")