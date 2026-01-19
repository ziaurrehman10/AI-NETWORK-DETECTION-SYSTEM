import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.stream_loader import StreamLoader
# Preprocessor class definition must be available or loaded from pickle correctly
# Ideally we import the class definition to ensure pickling works
from src.processing.preprocessor import Preprocessor
from src.models.baseline import IsolationForestModel
from src.models.deep_learning import AutoEncoderModel, LSTMModel

st.set_page_config(page_title="Network Anomaly Detection", layout="wide")

st.title("🛡️ Network Anomaly Detection System")
st.markdown("### Real-time Traffic Monitoring & Threat Intelligence")

# Sidebar Controls
st.sidebar.header("Simulation Control")
data_speed = st.sidebar.slider("Simulation Speed (s)", 0.1, 2.0, 0.5)
chunk_size = st.sidebar.slider("Batch Size", 1, 100, 10)
run_simulation = st.sidebar.checkbox("Start Monitoring", value=False)

# Metrics Placeholders
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_total_pkts = st.empty()
with col2:
    metric_anomalies = st.empty()
with col3:
    metric_status = st.empty()
with col4:
    metric_threat_level = st.empty()

# Charts Placeholders
st.subheader("Live Traffic Analysis")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("**Traffic Volume (Bytes)**")
    chart_bytes = st.empty()
with chart_col2:
    st.markdown("**Anomaly Scores**")
    chart_scores = st.empty()

# Alert Area
alert_area = st.empty()

# Load Resources (Cache to prevent reload)
@st.cache_resource
def load_resources():
    try:
        # Paths relative to where streamlit is run (root)
        preprocessor = joblib.load("src/processing/preprocessor.pkl")
        iso_forest = IsolationForestModel()
        iso_forest.load("src/models/isolation_forest.pkl")
        
        autoencoder = AutoEncoderModel(input_dim=1) # dim placeholder, loads from file
        autoencoder.load("src/models/autoencoder")
        
        # lstm = LSTMModel(...) # metrics/loading complex for LSTM in this demo, skipping for dashboard simplicity or load if ready
        # For this dashboard we focus on IF and AE as they are easier to visualize in real-time stream 1-by-1
        
        return preprocessor, iso_forest, autoencoder
    except Exception as e:
        return None, None, None

preprocessor, iso_forest, autoencoder = load_resources()

if not preprocessor:
    st.error("Models not found! Please run 'main.py' to train models first.")
    st.stop()

# Data Handling
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = pd.DataFrame()
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = []

def process_stream():
    # Use testing set for simulation
    data_path = r"D:\DS\Anomaly Detection\DATA\UNSW_NB15_testing-set.csv"
    loader = StreamLoader(data_path, chunk_size=chunk_size, delay=data_speed)
    
    total_anomalies = 0
    total_processed = 0
    
    for batch in loader.stream():
        if not run_simulation:
            break
            
        # Preprocess
        try:
            X, _ = preprocessor.transform(batch)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            break
            
        # Predict
        if_preds = iso_forest.predict(X)
        ae_preds = autoencoder.predict(X)
        
        # Ensemble Logic (OR gate for safety, AND for high confidence)
        final_preds = np.maximum(if_preds, ae_preds) # 1 if either deteects
        
        anomalies_batch = np.sum(final_preds)
        total_anomalies += anomalies_batch
        total_processed += len(batch)
        
        # Update Dashboard
        metric_total_pkts.metric("Total Packets", total_processed)
        metric_anomalies.metric("Detected Anomalies", int(total_anomalies), delta_color="inverse")
        
        status = "Normal" if anomalies_batch == 0 else "Anomalous"
        metric_status.metric("Network Status", status)
        
        threat_color = "red" if anomalies_batch > 0 else "green"
        metric_threat_level.markdown(f"<h3 style='color:{threat_color}'>{'CRITICAL' if anomalies_batch > 0 else 'LOW'}</h3>", unsafe_allow_html=True)
        
        if anomalies_batch > 0:
            alert_area.error(f"🚨 ALERT: {anomalies_batch} anomalies detected in current batch! Investigating IPs...")
            # Display anomalous records
            st.dataframe(batch[final_preds == 1].head(3))
        else:
            alert_area.success("System Normal. Monitoring active.")
            
        # Charts Update
        # Add to history
        new_row = {"Time": time.strftime("%H:%M:%S"), "Bytes": batch['sbytes'].mean() if 'sbytes' in batch else 0, "Anomalies": anomalies_batch}
        st.session_state.anomaly_history.append(new_row)
        if len(st.session_state.anomaly_history) > 50:
            st.session_state.anomaly_history.pop(0)
            
        hist_df = pd.DataFrame(st.session_state.anomaly_history)
        
        if not hist_df.empty:
            line = alt.Chart(hist_df).mark_line().encode(
                x='Time',
                y='Bytes'
            ).properties(title="Average Bytes/Batch")
            chart_bytes.altair_chart(line, use_container_width=True)
            
            bar = alt.Chart(hist_df).mark_bar(color='red').encode(
                x='Time',
                y='Anomalies'
            ).properties(title="Anomalies Found")
            chart_scores.altair_chart(bar, use_container_width=True)

        time.sleep(data_speed)

if run_simulation:
    process_stream()
else:
    st.info("Click 'Start Monitoring' to begin real-time analysis.")
