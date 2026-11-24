
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="TRANSFORMER IQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# INDUSTRIAL DARK THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0B1220;
    color: #E5EDF7;
}
[data-testid="stSidebar"] {
    background: #111D2E;
    border-right: 1px solid #26364B;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #F0F6FC !important;
}
p, label {
    color: #B8C7D9;
}
[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #263C57;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricValue"] {
    color: #35D6D0;
}
.stButton button {
    background: #087F8C;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 600;
}
.stButton button:hover {
    background: #0AA6B5;
    color: white;
}
.panel {
    background: #142238;
    border: 1px solid #263C57;
    border-radius: 12px;
    padding: 18px;
}
.small-muted {
    color: #91A7BF;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL LOADING
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "transformer_random_forest.pkl"
INFO_PATH = BASE_DIR / "models" / "model_info.pkl"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    info = joblib.load(INFO_PATH)
    return model, info

try:
    model, info = load_assets()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

# =====================================================
# FEATURE ENGINEERING
# Must match the training feature definitions
# =====================================================

LOAD_FEATURES = [
    "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"
]

REQUIRED_RAW_COLUMNS = ["date"] + LOAD_FEATURES + ["OT"]

def prepare_features(raw_df, expected_features):

    data = raw_df.copy()

    missing = [
        col for col in REQUIRED_RAW_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    for col in LOAD_FEATURES + ["OT"]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna(
        subset=REQUIRED_RAW_COLUMNS
    )

    data = data.sort_values("date").reset_index(drop=True)

    if len(data) < 49:
        raise ValueError(
            "At least 49 hourly records are required "
            "to calculate the 48-hour lag features."
        )

    # Oil temperature lags
    for lag in [1, 6, 12, 24, 48]:
        data[f"OT_lag_{lag}"] = data["OT"].shift(lag)

    # Load feature lags
    for col in LOAD_FEATURES:
        data[f"{col}_lag_24"] = data[col].shift(24)

    # Verify exact model inputs
    missing_features = [
        col for col in expected_features
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Feature mismatch. Missing: {missing_features}"
        )

    valid = data.dropna(
        subset=expected_features
    )

    if valid.empty:
        raise ValueError(
            "Not enough valid rows to generate features."
        )

    X_latest = valid[expected_features].iloc[[-1]]

    return data, X_latest

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# ⚡ TRANSFORMER IQ")
st.sidebar.caption("INDUSTRIAL INTELLIGENCE CENTER")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Command Center",
        "Thermal Analytics",
        "Load Intelligence",
        "AI Forecasting",
        "Model Lab",
        "Data Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### MODEL STATUS")
st.sidebar.success("Model artifacts loaded")

st.sidebar.markdown(
    '<p class="small-muted">Forecast horizon: 24 hours'
    '<br>Dataset: ETTh1'
    '<br>Algorithm: Random Forest</p>',
    unsafe_allow_html=True
)

# =====================================================
# MAIN HEADER
# =====================================================

st.markdown(
    '<p class="small-muted">POWER SYSTEMS / MACHINE LEARNING / MONITORING</p>',
    unsafe_allow_html=True
)

st.title("TRANSFORMER INTELLIGENCE CENTER")

st.markdown(
    "AI-powered oil temperature forecasting and load analytics"
)

st.divider()

# =====================================================
# DATA UPLOAD
# =====================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload historical transformer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        raw_df = pd.read_csv(uploaded_file)

        data, X_latest = prepare_features(
            raw_df,
            info["features"]
        )

        prediction = float(model.predict(X_latest)[0])

        st.sidebar.success("CSV processed successfully")

    except Exception as e:
        st.error(str(e))
        st.stop()

else:
    data = None
    X_latest = None
    prediction = None

# =====================================================
# REUSABLE CHART SETTINGS
# =====================================================

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#142238",
    plot_bgcolor="#142238",
    font=dict(color="#DCE8F5"),
    margin=dict(l=20, r=20, t=45, b=20),
    height=380
)

def show_upload_message():
    st.info(
        "Upload a historical CSV file from the sidebar "
        "to activate transformer analytics and forecasting."
    )

# =====================================================
# COMMAND CENTER
# =====================================================

if page == "Command Center":

    st.subheader("01 / COMMAND CENTER")

    if data is None:
        show_upload_message()

    else:
        current_ot = float(data["OT"].iloc[-1])
        forecast_time = (
            data["date"].iloc[-1] + pd.Timedelta(hours=24)
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "CURRENT OIL TEMP",
            f"{current_ot:.3f}"
        )

        c2.metric(
            "24H FORECAST",
            f"{prediction:.3f}",
            delta=f"{prediction-current_ot:+.3f}"
        )

        c3.metric(
            "MODEL R²",
            f"{info['r2']:.3f}"
        )

        c4.metric(
            "FORECAST HORIZON",
            "24 HOURS"
        )

        st.divider()

        st.subheader("Oil Temperature Overview")

        history = data.tail(168)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history["date"],
            y=history["OT"],
            mode="lines",
            name="Historical OT",
            line=dict(color="#35D6D0", width=2.5)
        ))

        fig.add_trace(go.Scatter(
            x=[history["date"].iloc[-1], forecast_time],
            y=[current_ot, prediction],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#FFB547", width=3, dash="dash")
        ))

        fig.update_layout(
            **CHART_LAYOUT,
            xaxis_title="Time",
            yaxis_title="Oil Temperature"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Forecast timestamp: {forecast_time}"
        )

# =====================================================
# THERMAL ANALYTICS
# =====================================================

elif page == "Thermal Analytics":

    st.subheader("02 / THERMAL ANALYTICS")

    if data is None:
        show_upload_message()

    else:
        history = data.tail(500)

        fig = px.line(
            history,
            x="date",
            y="OT",
            title="Historical Oil Temperature"
        )

        fig.update_traces(
            line_color="#35D6D0",
            line_width=2
        )

        fig.update_layout(
            **CHART_LAYOUT,
            xaxis_title="Date",
            yaxis_title="Oil Temperature"
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average OT",
            f"{data['OT'].mean():.3f}"
        )

        col2.metric(
            "Minimum OT",
            f"{data['OT'].min():.3f}"
        )

        col3.metric(
            "Maximum OT",
            f"{data['OT'].max():.3f}"
        )

        st.subheader("Temperature Statistics")
        st.dataframe(
            data[["OT"]].describe(),
            use_container_width=True
        )

# =====================================================
# LOAD INTELLIGENCE
# =====================================================

elif page == "Load Intelligence":

    st.subheader("03 / LOAD INTELLIGENCE")

    if data is None:
        show_upload_message()

    else:
        selected_load = st.selectbox(
            "Select load feature",
            LOAD_FEATURES
        )

        fig = px.line(
            data.tail(500),
            x="date",
            y=selected_load,
            title=f"{selected_load} Trend"
        )

        fig.update_traces(
            line_color="#FFB547",
            line_width=2
        )

        fig.update_layout(
            **CHART_LAYOUT,
            xaxis_title="Date",
            yaxis_title="Load feature value"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Load Feature Statistics")

        st.dataframe(
            data[LOAD_FEATURES].describe(),
            use_container_width=True
        )

# =====================================================
# AI FORECASTING
# =====================================================


# =====================================================
# AI FORECASTING - ENHANCED
# =====================================================

elif page == "AI Forecasting":

    st.subheader("04 / AI FORECASTING")

    if data is None:
        show_upload_message()

    else:
        current_ot = float(data["OT"].iloc[-1])

        forecast_time = (
            data["date"].iloc[-1] + pd.Timedelta(hours=24)
        )

        change = prediction - current_ot

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "CURRENT OIL TEMP",
            f"{current_ot:.3f}"
        )

        col2.metric(
            "24H PREDICTION",
            f"{prediction:.3f}",
            delta=f"{change:+.3f}"
        )

        col3.metric(
            "FORECAST HORIZON",
            "24 Hours"
        )

        st.divider()

        st.subheader("Temperature Forecast Visualization")

        history = data.tail(168)

        fig = go.Figure()

        # Historical temperature
        fig.add_trace(go.Scatter(
            x=history["date"],
            y=history["OT"],
            name="Historical Temperature",
            mode="lines",
            line=dict(color="#35D6D0", width=2.5)
        ))

        # Forecast endpoint
        fig.add_trace(go.Scatter(
            x=[forecast_time],
            y=[prediction],
            name="24H Forecast",
            mode="markers",
            marker=dict(
                color="#FFB547",
                size=15,
                symbol="diamond"
            )
        ))

        # Dashed connector - not an hourly forecast
        fig.add_trace(go.Scatter(
            x=[history["date"].iloc[-1], forecast_time],
            y=[current_ot, prediction],
            mode="lines",
            name="Endpoint connector",
            line=dict(
                color="#FFB547",
                width=2,
                dash="dash"
            )
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#142238",
            plot_bgcolor="#142238",
            height=480,
            xaxis_title="Date and Time",
            yaxis_title="Oil Temperature",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.12
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            f"Predicted oil temperature at {forecast_time}: "
            f"{prediction:.4f}"
        )

        st.caption(
            "The dashed line is only a visual connector between "
            "the latest observation and the forecast endpoint. "
            "It does not represent hourly predictions."
        )

        st.divider()

        st.subheader("Model Input Features")

        st.dataframe(
            X_latest.T.rename(
                columns={X_latest.index[0]: "Latest Input"}
            ),
            use_container_width=True
        )

# =====================================================
# DATA EXPLORER
# =====================================================

elif page == "Data Explorer":

    st.subheader("06 / DATA EXPLORER")

    if data is None:
        show_upload_message()

    else:
        st.metric("Total Records", f"{len(data):,}")

        st.write("Dataset Preview")

        st.dataframe(
            data.tail(100),
            use_container_width=True
        )

        st.write("Dataset Information")

        st.write("Columns:", len(data.columns))
        st.write("Date range:")
        st.write(
            f"{data['date'].min()} to {data['date'].max()}"
        )

        csv_data = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Processed Dataset",
            data=csv_data,
            file_name="transformer_processed_data.csv",
            mime="text/csv"
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    '<p class="small-muted">'
    'TRANSFORMER IQ | AI-BASED OIL TEMPERATURE FORECASTING'
    '<br>Research prototype — not a certified protection or fault diagnosis system.'
    '</p>',
    unsafe_allow_html=True
)
