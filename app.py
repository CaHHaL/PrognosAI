import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Predictive Dashboard", layout="wide")

# ===============================
# LOAD DATA
# ===============================
alert_df = pd.read_csv("outputs/alert_data.csv")
results = pd.read_csv("outputs/model_results.csv")

# ===============================
# DARK STYLE
# ===============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Dashboard Controls")

status_filter = st.sidebar.selectbox(
    "Filter Status",
    ["All", "Healthy", "Warning", "Critical"]
)

engine_index = st.sidebar.selectbox(
    "Select Engine",
    alert_df.index
)

# Filter data
if status_filter != "All":
    df = alert_df[alert_df["Status"] == status_filter]
else:
    df = alert_df

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align:center;'>🚀 AI Predictive Maintenance Dashboard</h1>", unsafe_allow_html=True)

# ===============================
# METRIC CARDS
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div class='card'>Total Engines<br><h2>{len(alert_df)}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'>Filtered<br><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'>Best RMSE<br><h2>{round(results['RMSE'].min(),4)}</h2></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='card'>Avg RUL<br><h2>{round(alert_df['Actual_RUL'].mean(),3)}</h2></div>", unsafe_allow_html=True)

# ===============================
# MODEL PERFORMANCE
# ===============================
st.markdown("---")
st.subheader("📊 Model Performance")

fig = px.bar(results, x="Model", y="RMSE", color="Model", title="RMSE Comparison")
st.plotly_chart(fig, use_container_width=True)

# ===============================
# ALERT DISTRIBUTION
# ===============================
st.subheader("🚨 Alert Distribution")

fig = px.pie(alert_df, names="Status", title="Engine Health Distribution")
st.plotly_chart(fig, use_container_width=True)

# ===============================
# ENGINE DETAILS
# ===============================
st.markdown("---")
st.subheader("🔍 Engine Details")

row = alert_df.loc[engine_index]

status_color = {
    "Healthy": "green",
    "Warning": "orange",
    "Critical": "red"
}

pred_val = row.get("Ensemble", row.get("Final_Prediction"))

col1, col2 = st.columns(2)

with col1:
    st.metric("Actual RUL", round(row["Actual_RUL"], 3))
    st.metric("Predicted RUL", round(pred_val, 3))
    st.markdown(f"### Status: <span style='color:{status_color[row['Status']]}'>{row['Status']}</span>", unsafe_allow_html=True)

with col2:
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_val,
        title={'text': "RUL Gauge"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 0.3], 'color': "red"},
                {'range': [0.3, 0.6], 'color': "orange"},
                {'range': [0.6, 1], 'color': "green"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# SMART ALERT
# ===============================
if row["Status"] == "Critical":
    st.error("⚠️ Immediate failure risk! Take action now.")
elif row["Status"] == "Warning":
    st.warning("⚠️ Degradation detected. Plan maintenance.")
else:
    st.success("✅ System operating normally.")

# ===============================
# RUL TREND
# ===============================
st.markdown("---")
st.subheader("📈 RUL Trends")

fig = go.Figure()

fig.add_trace(go.Scatter(y=alert_df["Actual_RUL"], name="Actual"))

if "LSTM" in alert_df.columns:
    fig.add_trace(go.Scatter(y=alert_df["LSTM"], name="LSTM"))

if "GRU" in alert_df.columns:
    fig.add_trace(go.Scatter(y=alert_df["GRU"], name="GRU"))

if "Ensemble" in alert_df.columns:
    fig.add_trace(go.Scatter(y=alert_df["Ensemble"], name="Ensemble"))

fig.update_layout(title="RUL Predictions")
st.plotly_chart(fig, use_container_width=True)

# ===============================
# ERROR ANALYSIS
# ===============================
st.subheader("📉 Error Distribution")

if "Ensemble" in alert_df.columns:
    errors = alert_df["Actual_RUL"] - alert_df["Ensemble"]
else:
    errors = alert_df["Actual_RUL"] - alert_df["Final_Prediction"]

fig = px.histogram(errors, nbins=50, title="Prediction Errors")
st.plotly_chart(fig, use_container_width=True)

# ===============================
# CRITICAL ENGINES
# ===============================
st.markdown("---")
st.subheader("🔥 Critical Engines")

critical = alert_df[alert_df["Status"] == "Critical"]

st.dataframe(critical.head(10), use_container_width=True)

# ===============================
# FULL DATA
# ===============================
st.markdown("---")
st.subheader("📋 Dataset Overview")

st.dataframe(df, use_container_width=True)