"""Smart Predictive Maintenance - Industrial AI Dashboard."""


import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src import config
from src.pipeline import run_pipeline
from src.recommend import recommend_maintenance
from src.rul import build_cycle_features, get_rul_status


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 25px;
}

.dashboard-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
}

.dashboard-subtitle {
    color: #91a9c0;
    font-size: 12px;
    margin-top: 5px;
}

.status-online {
    background: #063b32;
    border: 1px solid #00d68f;
    color: #00e6a0;
    padding: 9px 16px;
    border-radius: 30px;
    font-weight: 700;
}

.status-offline {
    background: #401827;
    border: 1px solid #ff4568;
    color: #ff8298;
    padding: 9px 16px;
    border-radius: 30px;
    font-weight: 700;
}

.section-heading {
    color: #eaf5ff;
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 12px;
}

.section-description {
    color: #91a9c0;
    font-size: 12px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #102b45, #091b2e);
    border: 1px solid #18547c;
    border-radius: 13px;
    padding: 16px;
    min-height: 100px;
}

div[data-testid="stMetricLabel"] {
    color: #a9c3da !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #0c2136, #08192b);
    border: 1px solid #174c70;
    border-radius: 13px;
    padding: 12px;
}

.stButton button,
.stFormSubmitButton button {
    border-radius: 9px;
    font-weight: 700;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #174c70;
    border-radius: 10px;
}

div[data-testid="stPlotlyChart"] {
    background: #091b2d;
    border: 1px solid #174b70;
    border-radius: 12px;
    padding: 5px;
}

div[data-testid="stExpander"] {
    background: #0c2035;
    border: 1px solid #174b70;
    border-radius: 10px;
}

hr {
    border-color: #173c58 !important;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PLOTLY THEME
# ============================================================

CHART_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": "Inter, sans-serif",
        "color": "#dceeff",
    },
    "margin": {
        "l": 35,
        "r": 20,
        "t": 45,
        "b": 35,
    },
    "legend": {
        "orientation": "h",
        "y": 1.12,
    },
}


# ============================================================
# FASTAPI CONNECTION
# ============================================================

def check_api_status() -> bool:
    """Check whether FastAPI is reachable."""

    try:
        response = requests.get(
            f"http://{config.API_HOST}:{config.API_PORT}/health",
            timeout=2,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def get_latest_api_predictions(limit: int = 10) -> list:
    """Retrieve the latest predictions from FastAPI."""

    response = requests.get(
        f"http://{config.API_HOST}:{config.API_PORT}/latest",
        params={"n": limit},
        timeout=5,
    )

    response.raise_for_status()

    records = response.json()

    if not isinstance(records, list):
        raise ValueError("Unexpected response from API.")

    return records


def get_api_prediction(
    machine_id,
    machine_type,
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear,
):
    """Send machine sensor inputs to FastAPI."""

    response = requests.post(
        f"http://{config.API_HOST}:{config.API_PORT}/predict",
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


def load_test_data():

    if not config.TEST_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(config.TEST_PATH)


# ============================================================
# NASA RUL MODEL
# ============================================================

@st.cache_resource
def load_rul_artifact():

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
            f"Missing RUL artifact keys: {missing}"
        )

    required_models = {"0.1", "0.5", "0.9"}

    if not required_models.issubset(artifact["models"]):
        raise ValueError("Required RUL quantile models are missing.")

    return artifact


@st.cache_data
def load_cmapss_test_data():

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
    data,
    rolling_window,
    feature_columns,
):

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
            f"Missing RUL features: {missing}"
        )

    return last_cycles[required]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

def render_sidebar():

    with st.sidebar:

        pages = {
            "🏠 Overview": "Overview",
            "📡 Live Predictions": "Live Predictions",
            "📈 Machine Monitoring": "Machine Monitoring",
            "⚙ Machine Analysis": "Machine Analysis",
            "🧠 SHAP Explainability": "SHAP Explainability",
            "🚀 NASA RUL Monitoring": "NASA RUL Monitoring",
            "📋 Project Status": "Project Status",
        }

        if "selected_page" not in st.session_state:
            st.session_state.selected_page = "Overview"

        # Brand occupies the full sidebar width.
        # The navigation dropdown is placed directly BELOW the brand,
        # rather than beside it, so menu items have enough horizontal space.
        st.markdown(
            """
            <div style="
                width:100%;
                box-sizing:border-box;
                padding:10px 10px 25px 10px;
                margin-bottom:8px;
                ">
                <div style="font-size:27px;font-weight:800;color:#08c8e8;">
                    ⚙ PREDICTIVE
                </div>
                <div style="font-size:19px;font-weight:700;color:white;">
                    MAINTENANCE
                </div>
                <div style="font-size:11px;color:#91a9c0;">
                    Explainable Industrial AI System
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Full-width expandable navigation directly below the brand.
        with st.expander("☰  Navigation Menu", expanded=False):
            for label, page in pages.items():
                if st.button(
                    label,
                    key=f"nav_{page}",
                    use_container_width=True,
                ):
                    st.session_state.selected_page = page
                    st.rerun()

        st.markdown("---")

        st.markdown(
            """
            <div style="
                background:#0c2941;
                border:1px solid #17547c;
                border-radius:12px;
                padding:15px;
                font-size:12px;
                color:#dceeff;
            ">
                <b>AI4I 2020 + NASA FD001</b>
                <br><br>
                Failure Prediction
                <br>
                Explainable AI
                <br>
                RUL Estimation
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.selected_page


# ============================================================
# DASHBOARD HEADER
# ============================================================

def render_header(api_online):
    """Render the dashboard title and API connection status."""

    status_class = "status-online" if api_online else "status-offline"
    status_text = "● API ONLINE" if api_online else "● API OFFLINE"

    header_html = f"""
<div class="dashboard-header">
    <div>
        <div class="dashboard-title">
            ⚙ Smart Predictive Maintenance Dashboard
        </div>
        <div class="dashboard-subtitle">
            AI4I Failure Prediction • Anomaly Detection •
            Explainable AI (SHAP) • NASA C-MAPSS RUL Monitoring
        </div>
    </div>
    <div class="{status_class}">
        {status_text}
    </div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================

def render_kpis(test_data, api_records):

    alert_count = 0
    anomaly_count = 0

    if api_records:

        df = pd.DataFrame(api_records)

        if "trigger_alert" in df:
            alert_count = int(
                df["trigger_alert"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        if "anomaly_flag" in df:
            anomaly_count = int(
                df["anomaly_flag"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

    mean_temp = (
        test_data["air_temp_k"].mean()
        if not test_data.empty and "air_temp_k" in test_data
        else 0
    )

    mean_speed = (
        test_data["rot_speed_rpm"].mean()
        if not test_data.empty and "rot_speed_rpm" in test_data
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "▣ Test Records",
        f"{len(test_data):,}",
    )

    c2.metric(
        "⚠ Failure Alerts",
        alert_count,
    )

    c3.metric(
        "∿ Anomalies Detected",
        anomaly_count,
    )

    c4.metric(
        "⚙ Mean Air Temperature",
        f"{mean_temp:.2f} K",
    )

    c5.metric(
        "◉ Mean Rotational Speed",
        f"{mean_speed:,.1f} RPM",
    )


# ============================================================
# LIVE API PREDICTIONS
# ============================================================

def render_live_predictions(api_online):

    st.markdown(
        '<div class="section-heading">📡 Live API Predictions</div>',
        unsafe_allow_html=True,
    )

    if not api_online:

        st.error("FastAPI is offline.")

        return

    try:

        refresh_col, info_col = st.columns([1, 3])

        with refresh_col:

            if st.button(
                "⟳ Refresh Predictions",
                use_container_width=True,
            ):
                st.cache_data.clear()

        with info_col:

            st.caption(
                "Latest machine predictions retrieved directly from FastAPI."
            )

        records = get_latest_api_predictions(limit=10)

        if not records:

            st.info("No API prediction records available yet.")

            return

        df = pd.DataFrame(records)

        display_columns = [
            "seq",
            "machine_id",
            "probability",
            "threshold",
            "trigger_alert",
            "anomaly_flag",
            "timestamp",
        ]

        available = [
            col
            for col in display_columns
            if col in df.columns
        ]

        display_df = df[available].copy()

        for column in ["probability", "threshold"]:

            if column in display_df.columns:

                display_df[column] = pd.to_numeric(
                    display_df[column],
                    errors="coerce",
                ).map(
                    lambda x: (
                        f"{x * 100:.4f}%"
                        if pd.notna(x)
                        else "N/A"
                    )
                )

        display_df.rename(
            columns={
                "seq": "#",
                "machine_id": "Machine ID",
                "probability": "Failure Prob.",
                "threshold": "Threshold",
                "trigger_alert": "Failure Alert",
                "anomaly_flag": "Anomaly",
                "timestamp": "Timestamp",
            },
            inplace=True,
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:

        st.error(
            f"Unable to load live predictions: {error}"
        )


# ============================================================
# MACHINE MONITORING
# ============================================================

def render_machine_monitoring(test_data):

    st.markdown(
        '<div class="section-heading">📈 Sensor Monitoring</div>',
        unsafe_allow_html=True,
    )

    if test_data.empty:

        st.warning("Test dataset unavailable.")

        return

    required = [
        "air_temp_k",
        "process_temp_k",
        "rot_speed_rpm",
        "torque_nm",
    ]

    missing = [
        column
        for column in required
        if column not in test_data.columns
    ]

    if missing:

        st.error(f"Missing sensor columns: {missing}")

        return

    chart_data = test_data.head(200).copy()

    if "temp_diff_k" not in chart_data:

        chart_data["temp_diff_k"] = (
            chart_data["process_temp_k"]
            - chart_data["air_temp_k"]
        )

    if "power_w" not in chart_data:

        chart_data["power_w"] = (
            chart_data["torque_nm"]
            * chart_data["rot_speed_rpm"]
            * 2
            * np.pi
            / 60
        )

    chart_data = chart_data.reset_index(drop=True)

    chart_data["Reading"] = chart_data.index + 1

    left, right = st.columns(2)

    with left:

        fig = px.line(
            chart_data,
            x="Reading",
            y="temp_diff_k",
            title="Temperature Difference",
        )

        fig.update_traces(
            line=dict(
                color="#08d9e8",
                width=2,
            )
        )

        fig.update_layout(
            **CHART_LAYOUT,
            height=280,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        fig = px.line(
            chart_data,
            x="Reading",
            y="power_w",
            title="Mechanical Power",
        )

        fig.update_traces(
            line=dict(
                color="#078cff",
                width=2,
            )
        )

        fig.update_layout(
            **CHART_LAYOUT,
            height=280,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MACHINE ANALYSIS FORM
# ============================================================

def render_machine_analysis(api_online):

    st.markdown(
        '<div class="section-heading">⚙ Machine Analysis</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Enter machine operating conditions to perform AI-based failure analysis."
    )

    with st.form("machine_analysis_form"):

        machine_id = st.text_input(
            "Machine ID",
            value="MACHINE-001",
        )

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            index=1,
        )

        col1, col2 = st.columns(2)

        with col1:

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

        with col2:

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
            "▶ Analyse Machine",
            use_container_width=True,
        )

    if submitted:

        if not machine_id.strip():

            st.error("Please enter a valid Machine ID.")

            return

        if process_temp <= air_temp:

            st.error(
                "Process temperature must be greater than air temperature."
            )

            return

        if not api_online:

            st.error(
                "FastAPI is offline. Start the API server first."
            )

            return

        try:

            with st.spinner("Running AI failure prediction..."):

                api_result = get_api_prediction(
                    machine_id=machine_id.strip(),
                    machine_type=machine_type,
                    air_temp=air_temp,
                    process_temp=process_temp,
                    rotational_speed=rotational_speed,
                    torque=torque,
                    tool_wear=tool_wear,
                )

            with st.spinner("Generating SHAP explanations..."):

                result = run_pipeline(
                    machine_id=machine_id.strip(),
                    type_=machine_type,
                    air_temp_k=air_temp,
                    process_temp_k=process_temp,
                    rot_speed_rpm=rotational_speed,
                    torque_nm=torque,
                    tool_wear_min=tool_wear,
                )

            probability = float(
                api_result["probability"]
            )

            threshold = float(
                api_result["threshold"]
            )

            anomaly_flag = bool(
                api_result["anomaly_flag"]
            )

            anomaly_score = float(
                api_result["anomaly_score"]
            )

            result["failure_prediction"] = {
                "probability": probability,
                "threshold": threshold,
                "trigger_alert": bool(
                    api_result["trigger_alert"]
                ),
            }

            result["anomaly_detection"] = {
                "anomaly_flag": anomaly_flag,
                "anomaly_score": anomaly_score,
            }

            result["maintenance"] = recommend_maintenance(
                probability=probability,
                anomaly_flag=anomaly_flag,
                failure_mode=result["failure_mode"]["prediction"],
                threshold=threshold,
            )

            result["api_response"] = api_result

            st.session_state["last_result"] = result

            st.success(
                "Prediction and explainability analysis completed."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# PREDICTION RESULTS
# ============================================================

def render_prediction_results():

    result = st.session_state.get("last_result")

    st.markdown(
        '<div class="section-heading">📊 Prediction Results</div>',
        unsafe_allow_html=True,
    )

    if not result:

        st.info(
            "Run Machine Analysis to see prediction results here."
        )

        return

    failure = result["failure_prediction"]

    anomaly = result["anomaly_detection"]

    maintenance = result["maintenance"]

    probability = failure["probability"] * 100

    threshold = failure["threshold"] * 100

    col1, col2 = st.columns([1, 1])

    with col1:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 35,
                        "color": "white",
                    },
                },
                title={
                    "text": "Failure Probability",
                    "font": {
                        "color": "#dceeff",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "white",
                    },
                    "bar": {
                        "color": "#08c8e8",
                    },
                    "bgcolor": "#0c2035",
                    "bordercolor": "#174b70",
                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#073b35",
                        },
                        {
                            "range": [30, 70],
                            "color": "#51451b",
                        },
                        {
                            "range": [70, 100],
                            "color": "#542037",
                        },
                    ],
                },
            )
        )

        gauge.update_layout(
            **CHART_LAYOUT,
            height=250,
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

        st.metric(
            "Dynamic Alert Threshold",
            f"{threshold:.4f}%",
        )

        if failure["trigger_alert"]:

            st.error("⚠ FAILURE ALERT TRIGGERED")

        else:

            st.success("✓ NORMAL OPERATION")

        if anomaly["anomaly_flag"]:

            st.warning("Anomalous behaviour detected.")

        else:

            st.success("No anomaly detected.")

        st.metric(
            "Anomaly Score",
            f"{anomaly['anomaly_score']:.4f}",
        )

    with col2:

        mode = result["failure_mode"]

        mode_df = pd.DataFrame({
            "Failure Mode": list(
                mode["probabilities"].keys()
            ),
            "Probability": list(
                mode["probabilities"].values()
            ),
        })

        fig = px.bar(
            mode_df,
            x="Probability",
            y="Failure Mode",
            orientation="h",
            title="Failure Mode Classification",
        )

        fig.update_traces(
            marker_color="#08bfe8"
        )

        fig.update_layout(
            **CHART_LAYOUT,
            height=330,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.metric(
            "Predicted Failure Mode",
            mode["prediction"],
        )

    st.markdown("### 🔧 Maintenance Recommendation")

    st.metric(
        "Maintenance Priority",
        maintenance["priority"],
    )

    st.info(
        maintenance["recommendation"]
    )

    with st.expander("View complete prediction output"):

        st.json(result)


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def render_shap_explainability():
    """Show concise, sample-specific fault evidence and inspection guidance."""
    st.markdown(
        '<div class="section-heading">🧠 SHAP Explainability</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.get("last_result")
    if not result:
        st.info("Run Machine Analysis to generate a SHAP explanation.")
        return

    explanation = result.get("explanation") or {}
    top_features = explanation.get("top_features") or []
    if not top_features:
        st.warning("SHAP contributions are unavailable for the latest prediction.")
        return

    labels = {
        "tool_wear_min": "Tool Wear",
        "torque_nm": "Torque",
        "rot_speed_rpm": "Rotational Speed",
        "air_temp_k": "Air Temperature",
        "process_temp_k": "Process Temperature",
        "temp_diff_k": "Temperature Difference",
        "power_w": "Mechanical Power",
        "type_encoded": "Machine Type",
    }
    # These mappings identify inspection areas, not confirmed failed parts.
    feature_inspection = {
        "tool_wear_min": ("Cutting tool / tool holder", "Inspect the cutting edge and tool holder; compare measured wear with the manufacturer's service limit."),
        "temp_diff_k": ("Cooling and heat-dissipation system", "Check coolant level and circulation, filters, pump, ventilation and heat exchanger. Verify both temperature sensors."),
        "air_temp_k": ("Cooling and ventilation system", "Check ambient conditions, ventilation openings and the air-temperature sensor."),
        "process_temp_k": ("Process cooling system", "Compare process temperature with the approved range; inspect coolant flow, pump, filters and temperature-sensor wiring."),
        "rot_speed_rpm": ("Motor drive / spindle speed system", "Compare actual RPM with the commanded setting; check drive settings, speed feedback, belts/couplings and spindle vibration."),
        "torque_nm": ("Mechanical load path / spindle", "Compare torque with the normal operating range; check for jamming, excessive cutting load, misalignment and abnormal friction."),
        "power_w": ("Motor and drive system", "Compare mechanical power with the normal value for the same operation; inspect overload, drive, bearings and electrical connections."),
        "type_encoded": ("Machine configuration", "Verify that the selected machine type matches the actual machine. This is not a physical component fault."),
    }
    units = {
        "air_temp_k": "K", "process_temp_k": "K", "temp_diff_k": "K",
        "rot_speed_rpm": "RPM", "torque_nm": "Nm", "tool_wear_min": "min",
        "power_w": "W",
    }

    clean_rows = []
    for item in top_features:
        try:
            raw = str(item["feature"])
            value = float(item["shap_value"])
        except (KeyError, TypeError, ValueError):
            continue
        clean_rows.append({
            "feature": raw,
            "label": labels.get(raw, raw.replace("_", " ").title()),
            "value": value,
        })
    if not clean_rows:
        st.warning("The SHAP result contains no valid feature contributions.")
        return

    # Compact contribution chart; mechanic guidance below is limited to one priority area.
    chart_df = pd.DataFrame([
        {"Feature": row["label"], "SHAP Value": row["value"]}
        for row in clean_rows
    ]).sort_values("SHAP Value", ascending=True)
    st.markdown("### Feature contribution analysis")
    fig = px.bar(
        chart_df, x="SHAP Value", y="Feature", orientation="h",
        color="SHAP Value", color_continuous_scale=["#078cff", "#6f8ca8", "#ff4568"],
    )
    fig.update_layout(
        **CHART_LAYOUT, height=300, coloraxis_showscale=False,
        xaxis_title="SHAP value (direction and strength of influence)",
        yaxis_title="Machine feature",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Positive SHAP values push the explained model output higher; negative values push it lower. "
        "They describe model influence, not physical proof of a fault."
    )

    failure = result.get("failure_prediction") or {}
    mode_result = result.get("failure_mode") or {}
    predicted_mode = str(mode_result.get("prediction", "UNKNOWN")).upper()
    probabilities = mode_result.get("probabilities") or {}
    try:
        mode_confidence = float(probabilities.get(predicted_mode))
    except (TypeError, ValueError):
        mode_confidence = None
    try:
        risk_probability = float(failure.get("probability"))
    except (TypeError, ValueError):
        risk_probability = None
    alert = bool(failure.get("trigger_alert", False))

    st.markdown("### 🛠️ Fault assessment and priority inspection")
    c1, c2, c3 = st.columns(3)
    c1.metric("Binary failure risk", f"{risk_probability * 100:.1f}%" if risk_probability is not None else "N/A")
    c2.metric("Failure alert", "TRIGGERED" if alert else "NOT TRIGGERED")
    c3.metric("Failure-mode prediction", predicted_mode)

    mode_names = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
        "NONE": "No specific failure mode identified",
    }
    mode_parts = {
        "TWF": ("Cutting tool / tool holder", "Inspect cutting-edge wear, chipping and tool-holder condition."),
        "HDF": ("Cooling and heat-dissipation system", "Inspect coolant flow, filters, pump, ventilation and heat exchanger."),
        "PWF": ("Motor, drive and power-transmission system", "Inspect motor/drive operation, power supply, torque and speed feedback."),
        "OSF": ("Tooling and mechanical load path", "Inspect tool wear, excessive load, jamming, spindle and bearings."),
        "RNF": ("No single component can be inferred", "Review machine history, sensor quality, wiring and physical condition."),
    }

    # If the mode model says NONE while the independent binary model raises an
    # alert, do not call the machine healthy: report high risk with unknown type.
    if alert and predicted_mode == "NONE":
        st.error(
            "HIGH FAILURE RISK — the binary model triggered an alert, but the "
            "failure-mode classifier could not identify a specific failure type."
        )
    elif alert:
        st.warning("Failure alert triggered. The failure-mode label is a model estimate and needs physical verification.")
    elif predicted_mode != "NONE":
        st.info("A failure mode was suggested, but the binary model did not trigger its failure alert.")
    else:
        st.success("No binary failure alert was triggered and no specific failure mode was identified.")

    # Prefer the classifier's named component when it has a non-NONE class.
    # Otherwise, use the strongest positive SHAP feature only as a check priority.
    if predicted_mode in mode_parts:
        component, checks = mode_parts[predicted_mode]
        evidence_rows = [r for r in clean_rows if r["feature"] in {
            "tool_wear_min" if predicted_mode == "TWF" else
            "temp_diff_k" if predicted_mode == "HDF" else
            "power_w" if predicted_mode == "PWF" else
            "torque_nm" if predicted_mode == "OSF" else "__none__",
            "air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min",
        }]
        basis = f"Failure-mode classifier: {mode_names.get(predicted_mode, predicted_mode)}"
    else:
        candidates = [r for r in clean_rows if r["value"] > 0 and r["feature"] in feature_inspection]
        if not candidates:
            candidates = [r for r in clean_rows if r["feature"] in feature_inspection]
        strongest = max(candidates, key=lambda r: abs(r["value"])) if candidates else None
        if strongest:
            component, checks = feature_inspection[strongest["feature"]]
            evidence_rows = [strongest]
            basis = "Priority selected from the strongest available SHAP feature; this is an inspection lead, not a diagnosed fault."
        else:
            component, checks, evidence_rows = "Not identifiable from available model output", "Review the machine's sensor readings and service history.", []
            basis = "Insufficient model evidence to identify a subsystem."

    st.markdown(f"**Predicted fault category:** {mode_names.get(predicted_mode, predicted_mode)} (`{predicted_mode}`)")
    if mode_confidence is not None:
        st.caption(f"Failure-mode classifier confidence: {mode_confidence * 100:.1f}%")
    st.markdown(f"**Priority inspection area:** {component}")
    st.caption(basis)
    st.markdown(f"**Mechanic should check:** {checks}")

    if evidence_rows:
        st.markdown("**Relevant evidence from this machine:**")
        for row in evidence_rows[:2]:
            raw = row["feature"]
            reading = (result.get("inputs") or {}).get(raw)
            if reading is None:
                reading = (result.get("features") or {}).get(raw)
            reading_text = f"; current reading: {reading} {units.get(raw, '')}" if reading is not None else ""
            direction = "increased the explained model output" if row["value"] > 0 else "decreased the explained model output"
            st.write(f"- **{row['label']}** — SHAP {row['value']:.4f} ({direction}){reading_text}")

    st.caption(
        "The binary failure model and multiclass failure-mode model are separate models. "
        "A high risk score with NONE means the system detects risk but cannot name the failure type. "
        "Confirm any suspected issue using sensor validation, physical inspection and the manufacturer's procedure."
    )


# ============================================================
# NASA RUL MONITORING
# ============================================================

def render_rul_monitoring():

    st.markdown(
        '<div class="section-heading">🚀 NASA C-MAPSS RUL Monitoring</div>',
        unsafe_allow_html=True,
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

            st.warning("No engine records available.")

            return

        selected_engine = st.selectbox(
            "Select Engine ID",
            engines,
            format_func=lambda value: f"Engine {int(value)}",
            key="rul_engine_selector",
        )

        selected_data = engine_features.loc[
            engine_features["unit"] == selected_engine
        ]

        if selected_data.empty:

            st.warning("No sensor records available.")

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

        lower = float(
            np.clip(
                models["0.1"].predict(X_engine)[0],
                0,
                rul_cap,
            )
        )

        median = float(
            np.clip(
                models["0.5"].predict(X_engine)[0],
                0,
                rul_cap,
            )
        )

        upper = float(
            np.clip(
                models["0.9"].predict(X_engine)[0],
                0,
                rul_cap,
            )
        )

        status = get_rul_status(median)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Engine ID",
            f"Engine {int(selected_engine)}",
        )

        c2.metric(
            "Predicted RUL",
            f"{median:.1f} cycles",
        )

        c3.metric(
            "RUL Interval",
            f"{lower:.1f} - {upper:.1f}",
        )

        c4.metric(
            "Current Cycle",
            current_cycle,
        )

        if status == "URGENT":

            st.error(
                "URGENT: Immediate maintenance attention required."
            )

        elif status == "PLAN_MAINTENANCE":

            st.warning(
                "PLAN MAINTENANCE: Schedule maintenance."
            )

        else:

            st.success(
                "NORMAL: Continue routine monitoring."
            )

        st.caption(
            "Estimates are produced by NASA C-MAPSS FD001 "
            "quantile models. They are not guaranteed failure dates."
        )

    except Exception as error:

        st.error(
            f"NASA RUL monitoring error: {error}"
        )


# ============================================================
# PROJECT STATUS
# ============================================================

def render_project_status():

    st.markdown(
        '<div class="section-heading">📋 Project Development Status</div>',
        unsafe_allow_html=True,
    )

    status_data = pd.DataFrame([
        {
            "Component": "AI4I Dataset",
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
            "Status": "Integrated",
        },
        {
            "Component": "FastAPI Integration",
            "Status": "Connected",
        },
    ])

    st.dataframe(
        status_data,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    selected_page = render_sidebar()

    api_online = check_api_status()

    test_data = load_test_data()

    try:

        api_records = (
            get_latest_api_predictions(limit=10)
            if api_online
            else []
        )

    except Exception:

        api_records = []

    # Header

    render_header(api_online)

    # ========================================================
    # OVERVIEW
    # ========================================================

    if selected_page == "Overview":

        render_kpis(
            test_data,
            api_records,
        )

        st.divider()

        left, right = st.columns(
            [1, 1.15]
        )

        with left:

            with st.container(border=True):

                render_live_predictions(api_online)

        with right:

            with st.container(border=True):

                render_machine_monitoring(test_data)

        st.divider()

        col1, col2 = st.columns(
            [0.85, 1.15]
        )

        with col1:

            with st.container(border=True):

                render_machine_analysis(api_online)

        with col2:

            with st.container(border=True):

                render_prediction_results()

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                render_shap_explainability()

        with col2:

            with st.container(border=True):

                render_rul_monitoring()

    # ========================================================
    # LIVE PREDICTIONS
    # ========================================================

    elif selected_page == "Live Predictions":

        render_live_predictions(api_online)

    # ========================================================
    # MACHINE MONITORING
    # ========================================================

    elif selected_page == "Machine Monitoring":

        render_machine_monitoring(test_data)

    # ========================================================
    # MACHINE ANALYSIS
    # ========================================================

    elif selected_page == "Machine Analysis":

        render_machine_analysis(api_online)

        st.divider()

        render_prediction_results()

    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================

    elif selected_page == "SHAP Explainability":

        render_shap_explainability()

    # ========================================================
    # NASA RUL MONITORING
    # ========================================================

    elif selected_page == "NASA RUL Monitoring":

        render_rul_monitoring()

    # ========================================================
    # PROJECT STATUS
    # ========================================================

    elif selected_page == "Project Status":

        render_project_status()

    # Footer

    st.divider()

    st.caption(
        "Smart Predictive Maintenance | AI4I 2020 | NASA C-MAPSS FD001 "
        "| Explainable Artificial Intelligence"
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()