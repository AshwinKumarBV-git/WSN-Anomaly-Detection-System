import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Constants ---
API_URL = "http://127.0.0.1:8000"
ANOMALY_TYPES = {
    0: "normal", 1: "DoS", 2: "Jamming", 3: "Tampering",
    4: "HardwareFault", 5: "EnvironmentalNoise"
}
ANOMALY_DISPLAY_NAMES = {
    "normal": "Normal", "DoS": "Denial of Service (DoS)", "Jamming": "Jamming",
    "Tampering": "Tampering", "HardwareFault": "Hardware Fault",
    "EnvironmentalNoise": "Environmental Noise", "unknown": "Unknown",
    "anomaly": "Anomaly (Unspecified)", "error": "Error"
}
ANOMALY_COLORS = {
    "normal": "#2ECC71", "DoS": "#E74C3C", "Jamming": "#F39C12",
    "Tampering": "#9B59B6", "HardwareFault": "#3498DB",
    "EnvironmentalNoise": "#1ABC9C", "unknown": "#95A5A6",
    "anomaly": "#E74C3C", "error": "#F1C40F"
}

# --- API Helper Functions ---

def check_api_status():
    """Check if the backend API is running and reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_system_status():
    """Get system status from the backend API."""
    try:
        response = requests.get(f"{API_URL}/status")
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

def get_models_info():
    """Get loaded model information from the backend API."""
    try:
        response = requests.get(f"{API_URL}/models")
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

def predict_single(sensor_data):
    """Send a single data point for prediction."""
    payload = {
        "data": sensor_data, "model_type": "ensemble",
        "return_probabilities": True, "return_features": True
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None

def predict_batch(sensor_data_list):
    """Send a batch of data points for prediction."""
    payload = {"data": sensor_data_list}
    try:
        response = requests.post(f"{API_URL}/predict/batch", json=payload)
        if response.status_code == 200:
            return response.json()
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None

def generate_simulation_data(num_samples=100, include_anomalies=True):
    """Request simulated data from the backend API."""
    params = {"num_samples": num_samples, "include_anomalies": include_anomalies}
    try:
        response = requests.get(f"{API_URL}/simulate", params=params)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None

# --- Plotting Functions ---

def plot_sensor_data(df, anomalies_df=None):
    """Plot sensor data with anomalies highlighted."""
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Temperature", "Motion", "Pulse"), shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'], mode='lines', name='Temperature'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['motion'], mode='lines', name='Motion', line_shape='hv'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['pulse'], mode='lines', name='Pulse'), row=3, col=1)

    if anomalies_df is not None and not anomalies_df.empty:
        for anomaly_type in anomalies_df['label'].unique():
            if anomaly_type == 'normal': continue
            type_anomalies = anomalies_df[anomalies_df['label'] == anomaly_type]
            color = ANOMALY_COLORS.get(anomaly_type, "#95A5A6")
            display_name = ANOMALY_DISPLAY_NAMES.get(anomaly_type, anomaly_type)
            fig.add_trace(go.Scatter(x=type_anomalies['timestamp'], y=type_anomalies['temperature'], mode='markers', name=display_name, marker=dict(color=color, size=8, symbol='x')), row=1, col=1)
            fig.add_trace(go.Scatter(x=type_anomalies['timestamp'], y=type_anomalies['motion'], mode='markers', name=display_name, marker=dict(color=color, size=8, symbol='x'), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=type_anomalies['timestamp'], y=type_anomalies['pulse'], mode='markers', name=display_name, marker=dict(color=color, size=8, symbol='x'), showlegend=False), row=3, col=1)

    fig.update_layout(height=600, title_text="Sensor Data with Ground Truth Labels", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="°C", row=1, col=1); fig.update_yaxes(title_text="State", row=2, col=1, type='category'); fig.update_yaxes(title_text="BPM", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    return fig

def plot_prediction_results(df, predictions):
    """Plot sensor data with model predictions highlighted."""
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Temperature", "Motion", "Pulse"), shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'], mode='lines', name='Temperature', line=dict(color='grey')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['motion'], mode='lines', name='Motion', line=dict(color='grey'), line_shape='hv'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['pulse'], mode='lines', name='Pulse', line=dict(color='grey')), row=3, col=1)

    pred_df = pd.DataFrame(predictions)
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
    
    anomalous_preds = pred_df[pred_df['prediction'] != 'normal']
    merged_anomalies = pd.merge_asof(anomalous_preds.sort_values('timestamp'), df.sort_values('timestamp'), on='timestamp', direction='nearest', tolerance=pd.Timedelta('1min'))

    for pred_type in merged_anomalies['prediction'].unique():
        type_preds = merged_anomalies[merged_anomalies['prediction'] == pred_type]
        color = ANOMALY_COLORS.get(pred_type, "#95A5A6")
        display_name = ANOMALY_DISPLAY_NAMES.get(pred_type, pred_type)
        fig.add_trace(go.Scatter(x=type_preds['timestamp'], y=type_preds['temperature'], mode='markers', name=display_name, marker=dict(color=color, size=8, symbol='circle')), row=1, col=1)
        fig.add_trace(go.Scatter(x=type_preds['timestamp'], y=type_preds['motion'], mode='markers', name=display_name, marker=dict(color=color, size=8, symbol='circle'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=type_preds['timestamp'], y=type_preds['pulse'], mode='markers', name=display_name, marker=dict(color=color, size=8, symbol='circle'), showlegend=False), row=3, col=1)

    fig.update_layout(height=600, title_text="Sensor Data with Model Predictions", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Streamlit Page Functions ---

def display_api_status():
    """Check and display the API status at the top of the page."""
    if not check_api_status():
        st.error("⚠️ Backend API is not running. Please start the server by running `python backend/main.py` in your terminal.", icon="🚨")
        st.stop()
    else:
        st.success("✅ Backend API is running.", icon="🚀")

def create_dashboard_page():
    """Create the main dashboard page for live monitoring."""
    st.title("📡 WSN Anomaly Detection Dashboard")
    display_api_status()

    with st.expander("System & Model Status"):
        system_status = get_system_status()
        if system_status:
            col1, col2, col3 = st.columns(3)
            uptime_s = system_status.get('uptime_seconds', 0)
            uptime_str = f"{int(uptime_s // 3600)}h {int((uptime_s % 3600) // 60)}m"
            col1.metric("Uptime", uptime_str)
            col2.metric("CPU Usage", f"{system_status.get('cpu_usage_percent', 0):.1f}%")
            col3.metric("Memory Usage", f"{system_status.get('memory_usage_mb', 0):.1f} MB")
        
        models_info = get_models_info()
        if models_info:
            for model in models_info:
                st.write(f"**Model:** `{model.get('model_name')}` | **Type:** `{model.get('model_type')}`")
        else:
            st.warning("No models loaded. Please ensure they are trained.")

    st.header("Live Simulation & Analysis")
    col1, col2, col3 = st.columns([1, 1, 2])
    num_samples = col1.number_input("Number of samples", 50, 2000, 200, 50)
    include_anomalies = col2.checkbox("Inject anomalies", True)
    if col3.button("▶️ Run Simulation & Analysis", use_container_width=True):
        with st.spinner("Generating simulation data..."):
            sim_response = generate_simulation_data(num_samples, include_anomalies)
        if sim_response and 'data' in sim_response:
            df = pd.DataFrame(sim_response['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.session_state['simulation_data'] = df
            
            with st.spinner("Analyzing data..."):
                batch_response = predict_batch(df.to_dict('records'))
            if batch_response and 'predictions' in batch_response:
                st.session_state['predictions'] = batch_response['predictions']
                anomaly_count = sum(1 for p in batch_response['predictions'] if p['prediction'] != 'normal')
                st.success(f"Analysis complete. Detected {anomaly_count} anomalous windows.")

    if 'simulation_data' in st.session_state:
        df = st.session_state['simulation_data']
        st.subheader("Ground Truth Data")
        anomalies_df = df[df['label'] != 'normal']
        st.plotly_chart(plot_sensor_data(df, anomalies_df), use_container_width=True)

    if 'predictions' in st.session_state:
        st.subheader("Model Prediction Results")
        predictions = st.session_state['predictions']
        st.plotly_chart(plot_prediction_results(st.session_state['simulation_data'], predictions), use_container_width=True)
        
        st.subheader("Detected Anomaly Details")
        anomalies = [p for p in predictions if p['prediction'] != 'normal']
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(anomalies_df[['timestamp', 'prediction', 'confidence']], use_container_width=True)

def create_manual_test_page():
    """Create the page for manually testing single data points."""
    st.title("🔬 Manual Anomaly Test")
    display_api_status()

    st.subheader("Sensor Data Input")
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        temperature = col1.slider("Temperature (°C)", -20.0, 60.0, 25.0, 0.1)
        motion = col2.selectbox("Motion Detected", [0, 1])
        pulse = col3.slider("Pulse (BPM)", 30.0, 220.0, 70.0, 0.1)
        submitted = st.form_submit_button("Predict Anomaly")

    if submitted:
        sensor_data = {
            "temperature": temperature, "motion": motion, "pulse": pulse,
            "timestamp": datetime.now().isoformat(), "sensor_id": "manual_test"
        }
        with st.spinner("Predicting..."):
            result = predict_single(sensor_data)
        
        if result:
            st.subheader("Prediction Result")
            prediction = result.get('prediction', 'error')
            confidence = result.get('confidence', 0)
            
            col1, col2 = st.columns(2)
            display_name = ANOMALY_DISPLAY_NAMES.get(prediction, prediction)
            if prediction != 'normal':
                col1.error(f"Anomaly Detected: {display_name}", icon="⚠️")
            else:
                col1.success(f"Normal: {display_name}", icon="✅")
            
            col2.metric("Confidence Score", f"{confidence:.4f}")

            if 'probabilities' in result:
                st.subheader("Model Prediction Probabilities")
                probs = result['probabilities']
                if isinstance(probs, dict) and probs:
                    # **DEFINITIVE FIX**: This logic correctly processes the numeric keys from the backend
                    # and prepares the data and color map for Plotly in a robust way.
                    prob_data = []
                    for k, v in probs.items():
                        # The key from the backend is a string like '0', '1', etc.
                        if isinstance(k, str) and k.isdigit():
                            internal_name = ANOMALY_TYPES.get(int(k))
                            if internal_name:
                                prob_data.append({
                                    "Class": ANOMALY_DISPLAY_NAMES.get(internal_name),
                                    "Probability": v,
                                    "Color": ANOMALY_COLORS.get(internal_name)
                                })
                    
                    if prob_data:
                        probs_df = pd.DataFrame(prob_data)
                        
                        fig = px.bar(
                            probs_df, 
                            x='Class', 
                            y='Probability',
                            color='Class',
                            color_discrete_map={row['Class']: row['Color'] for _, row in probs_df.iterrows()},
                            title="Anomaly Classification Probabilities"
                        )
                        fig.update_layout(
                            xaxis_title="Anomaly Class",
                            yaxis_title="Probability",
                            yaxis_range=[0,1], # Set a fixed y-axis range
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No valid probability data to display.")
                else:
                    st.warning("Invalid probability data format received from API.")

# --- Main Application ---
def main():
    st.set_page_config(page_title="WSN Anomaly Detection", page_icon="📡", layout="wide")

    st.sidebar.title("WSN Anomaly Detection")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Manual Test"])

    if page == "Dashboard":
        create_dashboard_page()
    elif page == "Manual Test":
        create_manual_test_page()

    st.sidebar.markdown("---")
    st.sidebar.info("Version 1.3 | © 2025")

if __name__ == "__main__":
    main()
