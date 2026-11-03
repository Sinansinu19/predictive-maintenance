"""Streamlit dashboard for explainable predictive maintenance."""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src import config
from src.pipeline import run_pipeline
from src.recommend import recommend_maintenance
from src.rul import build_cycle_features, get_rul_status


# FastAPI configuration.
# Sinan's API was tested successfully on port 8001.
API_BASE_URL = "http://127.0.0.1:8001"


# ============================================================
# FASTAPI CONNECTION
# ============================================================

def check_api_status() -> bool:
    """Check whether FastAPI is reachable."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=2,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def get_api_prediction(
    machine_id: str,
    machine_type: str,
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """Send machine inputs to FastAPI."""

    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={
            "machine_id": machine_id,
            "inputs": {
                "type": machine_type,
                "air_temp_k": air_temp,
                "process_temp_k": process_temp,
                "rot_speed_rpm": rotational_speed,
                "torque_nm": torque,
                "tool_wear_min": tool_wear,
            },
            "weather": None,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def load_test_data() -> pd.DataFrame:
    """Load processed AI4I test data."""

    if not config.TEST_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(config.TEST_PATH)


# ============================================================
# NASA C-MAPSS RUL MODEL
# ============================================================

@st.cache_resource
def load_rul_artifact() -> dict:
    """Load and validate the NASA FD001 quantile-model artifact."""

    if not config.RUL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RUL model not found: {config.RUL_MODEL_PATH}"
        )

    artifact = joblib.load(config.RUL_MODEL_PATH)

    required_keys = {
        "models",
        "feature_columns",
        "quantiles",
        "rul_cap",
    }

    missing = required_keys - set(artifact)

    if missing:
        raise ValueError(
            f"RUL model artifact is missing keys: {sorted(missing)}"
        )

    if not isinstance(artifact["models"], dict):
        raise ValueError(
            "RUL models must be stored in a dictionary."
        )

    required_models = {"0.1", "0.5", "0.9"}

    missing_models = required_models - set(artifact["models"])

    if missing_models:
        raise ValueError(
            f"RUL quantile models missing: {sorted(missing_models)}"
        )

    return artifact


@st.cache_data
def load_cmapss_test_data() -> pd.DataFrame:
    """Load NASA C-MAPSS FD001 test data."""

    test_path = config.CMAPSS_DIR / "test_FD001.txt"

    if not test_path.exists():
        raise FileNotFoundError(
            f"NASA FD001 test data not found: {test_path}"
        )

    return pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=config.CMAPSS_COLUMNS,
    )


@st.cache_data
def prepare_rul_test_features(
    data: pd.DataFrame,
    rolling_window: int,
    feature_columns: tuple,
) -> pd.DataFrame:
    """Create rolling features over complete engine histories."""

    rolling = build_cycle_features(
        data,
        window=rolling_window,
    )

    last_cycles = (
        rolling.groupby(
            "unit",
            sort=True,
            group_keys=False,
        )
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )

    required = list(
        dict.fromkeys(
            ["unit", "cycle", *feature_columns]
        )
    )

    missing = [
        column
        for column in required
        if column not in last_cycles
    ]

    if missing:
        raise ValueError(
            f"Missing RUL inference features: {missing}"
        )

    return last_cycles[required]


def render_rul_monitoring() -> None:
    """Render NASA C-MAPSS RUL predictions and quantile estimates."""

    st.header("NASA C-MAPSS Engine Health Monitoring")

    st.write(
        "Select an engine from the NASA FD001 test dataset "
        "to estimate its Remaining Useful Life (RUL)."
    )

    try:
        artifact = load_rul_artifact()

        test_data = load_cmapss_test_data()

        feature_columns = tuple(
            artifact["feature_columns"]
        )

        engine_features = prepare_rul_test_features(
            test_data,
            config.RUL_WINDOW,
            feature_columns,
        )

        engines = sorted(
            engine_features["unit"].unique()
        )

        if not engines:
            st.warning(
                "No NASA engine records are available."
            )
            return

        selected_engine = st.selectbox(
            "Select NASA Engine ID",
            engines,
            format_func=lambda value: f"Engine {int(value)}",
            key="rul_engine_selector",
        )

        selected_data = engine_features.loc[
            engine_features["unit"] == selected_engine
        ]

        if selected_data.empty:
            st.error(
                "No sensor data available for the selected engine."
            )
            return

        current_cycle = int(
            selected_data["cycle"].iloc[0]
        )

        X_engine = selected_data.loc[
            :,
            list(feature_columns),
        ]

        models = artifact["models"]

        rul_cap = float(
            artifact["rul_cap"]
        )

        lower_prediction = float(
            np.clip(
                models["0.1"].predict(X_engine)[0],
                0,
                rul_cap,
            )
        )

        median_prediction = float(
            np.clip(
                models["0.5"].predict(X_engine)[0],
                0,
                rul_cap,
            )
        )

        upper_prediction = float(
            np.clip(
                models["0.9"].predict(X_engine)[0],
                0,
                rul_cap,
            )
        )

        status = get_rul_status(
            median_prediction
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Remaining Useful Life",
            f"{median_prediction:.1f} cycles",
        )

        col2.metric(
            "Current Engine Cycle",
            current_cycle,
        )

        col3.metric(
            "Maintenance Status",
            status.replace("_", " "),
        )

        st.subheader("RUL Prediction Interval")

        low_col, mid_col, high_col = st.columns(3)

        low_col.metric(
            "Lower Estimate (10%)",
            f"{lower_prediction:.1f} cycles",
        )

        mid_col.metric(
            "Median Estimate (50%)",
            f"{median_prediction:.1f} cycles",
        )

        high_col.metric(
            "Upper Estimate (90%)",
            f"{upper_prediction:.1f} cycles",
        )

        interval_low = min(
            lower_prediction,
            median_prediction,
            upper_prediction,
        )

        interval_high = max(
            lower_prediction,
            median_prediction,
            upper_prediction,
        )

        st.info(
            f"Estimated model interval: "
            f"{interval_low:.1f}–{interval_high:.1f} cycles. "
            "Maintenance status uses the median estimate."
        )

        if status == "URGENT":

            st.error(
                "URGENT: This engine requires immediate "
                "maintenance attention."
            )

        elif status == "PLAN_MAINTENANCE":

            st.warning(
                "PLAN MAINTENANCE: Schedule maintenance "
                "for this engine."
            )

        else:

            st.success(
                "NORMAL: Continue routine engine monitoring."
            )

        st.divider()

        st.subheader("RUL Model Information")

        info1, info2, info3 = st.columns(3)

        info1.metric(
            "Dataset",
            "NASA FD001",
        )

        info2.metric(
            "Model Features",
            len(feature_columns),
        )

        info3.metric(
            "Prediction Limit",
            f"{rul_cap:.0f} cycles",
        )

        st.caption(
            "RUL estimates use three Gradient Boosting quantile "
            "regression models trained on NASA C-MAPSS FD001. "
            "The 10th, 50th and 90th percentile predictions form "
            "a model-based estimate interval; they are not "
            "guaranteed failure dates."
        )

    except Exception as error:

        st.error(
            f"RUL monitoring error: {error}"
        )


# ============================================================
# AI4I PREDICTION DISPLAY
# ============================================================

def render_prediction(result: dict) -> None:
    """Display integrated AI4I prediction results."""

    st.subheader("Failure Prediction")

    failure = result["failure_prediction"]

    anomaly = result["anomaly_detection"]

    maintenance = result["maintenance"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Failure Probability",
        f"{failure['probability'] * 100:.2f}%",
    )

    col2.metric(
        "Alert Threshold",
        f"{failure['threshold'] * 100:.2f}%",
    )

    col3.metric(
        "Anomaly Score",
        f"{anomaly['anomaly_score']:.4f}",
    )

    if failure["trigger_alert"]:

        st.error(
            "URGENT: Failure probability exceeded the alert threshold."
        )

    else:

        st.success(
            "Failure probability is below the alert threshold."
        )

    st.divider()

    st.subheader("Anomaly Detection")

    if anomaly["anomaly_flag"]:

        st.warning(
            "Anomalous machine behaviour detected."
        )

    else:

        st.success(
            "No anomaly detected."
        )

    st.subheader("Failure Mode Classification")

    mode = result["failure_mode"]

    st.metric(
        "Predicted Failure Mode",
        mode["prediction"],
    )

    mode_df = pd.DataFrame({
        "Failure Mode": list(
            mode["probabilities"].keys()
        ),
        "Probability": list(
            mode["probabilities"].values()
        ),
    })

    st.plotly_chart(
        px.bar(
            mode_df,
            x="Failure Mode",
            y="Probability",
            title="Failure Mode Probabilities",
        ),
        use_container_width=True,
    )

    st.divider()

    st.subheader("SHAP Explainability")

    explanation = result["explanation"]

    shap_df = pd.DataFrame([
        {
            "Feature": item["feature"],
            "SHAP Value": item["shap_value"],
        }
        for item in explanation["top_features"]
    ])

    st.plotly_chart(
        px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            title="Top Five Influential Features",
        ),
        use_container_width=True,
    )

    with st.expander(
        "View complete SHAP explanation"
    ):

        st.json(explanation)

    st.divider()

    st.subheader("Maintenance Recommendation")

    st.metric(
        "Maintenance Priority",
        maintenance["priority"],
    )

    st.info(
        maintenance["recommendation"]
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

def main() -> None:
    """Render the predictive maintenance dashboard."""

    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="🔧",
        layout="wide",
    )

    st.title(
        "Smart Predictive Maintenance"
    )

    st.caption(
        "Explainable AI-based machine monitoring and maintenance support"
    )

    # --------------------------------------------------------
    # API STATUS
    # --------------------------------------------------------

    api_online = check_api_status()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prediction API",
        "Online" if api_online else "Offline",
    )

    col2.metric(
        "Monitoring Mode",
        "FastAPI + Local Explainability",
    )

    col3.metric(
        "Dashboard Refresh",
        datetime.now().strftime("%H:%M:%S"),
    )

    st.divider()

    # --------------------------------------------------------
    # MACHINE MONITORING
    # --------------------------------------------------------

    st.header("Machine Monitoring")

    test_data = load_test_data()

    if test_data.empty:

        st.warning(
            "Processed test dataset unavailable. "
            "Check data/processed/test.csv."
        )

    else:

        required_columns = [
            "air_temp_k",
            "process_temp_k",
            "rot_speed_rpm",
            "torque_nm",
        ]

        missing = [
            column
            for column in required_columns
            if column not in test_data
        ]

        if missing:

            st.error(
                f"Missing dataset columns: {missing}"
            )

        else:

            if "temp_diff_k" not in test_data:

                test_data["temp_diff_k"] = (
                    test_data["process_temp_k"]
                    - test_data["air_temp_k"]
                )

            if "power_w" not in test_data:

                test_data["power_w"] = (
                    test_data["torque_nm"]
                    * test_data["rot_speed_rpm"]
                    * 2
                    * np.pi
                    / 60
                )

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Test Records",
                f"{len(test_data):,}",
            )

            m2.metric(
                "Mean Air Temperature",
                f"{test_data['air_temp_k'].mean():.2f} K",
            )

            m3.metric(
                "Mean Rotational Speed",
                f"{test_data['rot_speed_rpm'].mean():.1f} RPM",
            )

            st.caption(
                "Statistics describe the processed synthetic test dataset."
            )

            st.subheader("Sensor Trends")

            chart_data = (
                test_data.head(200)
                .reset_index(drop=True)
            )

            chart_data["Reading"] = (
                chart_data.index + 1
            )

            left, right = st.columns(2)

            with left:

                st.plotly_chart(
                    px.line(
                        chart_data,
                        x="Reading",
                        y="temp_diff_k",
                        title="Temperature Difference",
                    ),
                    use_container_width=True,
                )

            with right:

                st.plotly_chart(
                    px.line(
                        chart_data,
                        x="Reading",
                        y="power_w",
                        title="Mechanical Power",
                    ),
                    use_container_width=True,
                )

    st.divider()

    # --------------------------------------------------------
    # AI MACHINE FAILURE PREDICTION
    # --------------------------------------------------------

    st.header(
        "AI Machine Failure Prediction"
    )

    st.write(
        "Enter machine operating conditions. FastAPI performs live "
        "failure and anomaly prediction; the local pipeline provides "
        "failure-mode classification and SHAP explanations."
    )

    with st.form("machine_prediction_form"):

        machine_id = st.text_input(
            "Machine ID",
            value="MACHINE-001",
        )

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            index=1,
        )

        c1, c2 = st.columns(2)

        with c1:

            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=0.1,
                value=300.0,
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=0.1,
                value=310.0,
            )

            rotational_speed = st.number_input(
                "Rotational Speed (RPM)",
                min_value=0.1,
                value=1500.0,
            )

        with c2:

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                value=40.0,
            )

            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0.0,
                value=100.0,
            )

        submitted = st.form_submit_button(
            "Analyse Machine",
            use_container_width=True,
        )

    if submitted:

        if not machine_id.strip():

            st.error(
                "Please enter a valid Machine ID."
            )

        elif process_temp <= air_temp:

            st.error(
                "Process temperature must be greater than air temperature."
            )

        else:

            try:

                if not api_online:

                    st.error(
                        "FastAPI is offline. Start the API server "
                        "before analysing a machine."
                    )

                    st.stop()

                with st.spinner(
                    "Sending sensor inputs to FastAPI..."
                ):

                    api_result = get_api_prediction(
                        machine_id=machine_id.strip(),
                        machine_type=machine_type,
                        air_temp=air_temp,
                        process_temp=process_temp,
                        rotational_speed=rotational_speed,
                        torque=torque,
                        tool_wear=tool_wear,
                    )

                with st.spinner(
                    "Generating local SHAP and failure-mode explanations..."
                ):

                    result = run_pipeline(
                        machine_id=machine_id.strip(),
                        type_=machine_type,
                        air_temp_k=air_temp,
                        process_temp_k=process_temp,
                        rot_speed_rpm=rotational_speed,
                        torque_nm=torque,
                        tool_wear_min=tool_wear,
                    )

                api_probability = float(
                    api_result["probability"]
                )

                api_threshold = float(
                    api_result["threshold"]
                )

                api_anomaly_flag = bool(
                    api_result["anomaly_flag"]
                )

                api_anomaly_score = float(
                    api_result["anomaly_score"]
                )

                result["failure_prediction"] = {
                    "probability": api_probability,
                    "threshold": api_threshold,
                    "trigger_alert": bool(
                        api_result["trigger_alert"]
                    ),
                }

                result["anomaly_detection"] = {
                    "anomaly_flag": api_anomaly_flag,
                    "anomaly_score": api_anomaly_score,
                }

                result["maintenance"] = recommend_maintenance(
                    probability=api_probability,
                    anomaly_flag=api_anomaly_flag,
                    failure_mode=result["failure_mode"]["prediction"],
                    threshold=api_threshold,
                )

                result["api_response"] = api_result

                st.success(
                    "FastAPI prediction and local explainability "
                    "analysis completed successfully."
                )

                render_prediction(result)

                with st.expander(
                    "View complete prediction output, including FastAPI response"
                ):

                    st.json(result)

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

    st.divider()

    # --------------------------------------------------------
    # NASA C-MAPSS RUL MONITORING
    # --------------------------------------------------------

    render_rul_monitoring()

    st.divider()

    # --------------------------------------------------------
    # PROJECT DEVELOPMENT STATUS
    # --------------------------------------------------------

    st.header(
        "Project Development Status"
    )

    status_table = pd.DataFrame([
        {
            "Component": "Processed AI4I Dataset",
            "Status": "Available",
        },
        {
            "Component": "Binary Failure Model",
            "Status": "Integrated",
        },
        {
            "Component": "Anomaly Detection",
            "Status": "Integrated",
        },
        {
            "Component": "Failure Mode Classification",
            "Status": "Integrated",
        },
        {
            "Component": "SHAP Explainability",
            "Status": "Integrated",
        },
        {
            "Component": "Maintenance Recommendation",
            "Status": "Integrated",
        },
        {
            "Component": "NASA FD001 RUL Model",
            "Status": "Trained and Integrated",
        },
        {
            "Component": "FastAPI Live Integration",
            "Status": "Connected - Verified",
        },
    ])

    st.dataframe(
        status_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "AI4I 2020 is a synthetic dataset. Environmental variation is simulated "
        "and does not establish that weather causes machine failures. NASA "
        "C-MAPSS FD001 is used separately for Remaining Useful Life estimation."
    )


if __name__ == "__main__":
    main()