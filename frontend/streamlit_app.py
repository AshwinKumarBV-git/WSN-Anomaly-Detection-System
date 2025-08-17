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

# --- Constants ---
# Make sure the API_URL matches the address of your running FastAPI backend
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

def check_api_root():
    """Check the root endpoint of the API."""
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def check_api_health():
    """Check the /health endpoint of the API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return True, response.json()
        return False, {"status": "unhealthy", "detail": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"status": "unhealthy", "error": str(e)}

def get_system_status():
    """Get system status from the /status endpoint."""
    try:
        response = requests.get(f"{API_URL}/status")
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

def get_models_info():
    """Get loaded model information from the /models endpoint."""
    try:
        response = requests.get(f"{API_URL}/models")
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException:
        return []

def predict_single(sensor_data):
    """Send a single data point to the /predict endpoint."""
    payload = {
        "data": sensor_data, "model_type": "ensemble",
        "return_probabilities": True, "return_features": True
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        st.error(f"API Error on /predict: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None

def predict_batch(sensor_data_list):
    """Send a batch of data to the /predict/batch endpoint."""
    payload = {"data": sensor_data_list, "model_type": "ensemble", "return_probabilities": True}
    try:
        response = requests.post(f"{API_URL}/predict/batch", json=payload)
        if response.status_code == 200:
            return response.json()
        st.error(f"API Error on /predict/batch: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None

def generate_simulation_data(num_samples=100, include_anomalies=True):
    """Request simulated data from the /simulate endpoint."""
    params = {"num_samples": num_samples, "include_anomalies": include_anomalies}
    try:
        response = requests.get(f"{API_URL}/simulate", params=params)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None

# --- Plotting Functions ---

def plot_sensor_data(df, anomalies_df=None, title="Sensor Data with Ground Truth Labels"):
    """Plot sensor data with optional anomalies highlighted."""
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

    fig.update_layout(height=600, title_text=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="°C", row=1, col=1); fig.update_yaxes(title_text="State", row=2, col=1, type='category'); fig.update_yaxes(title_text="BPM", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    return fig

def plot_prediction_results(df, predictions):
    """Plot sensor data with model predictions highlighted."""
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Temperature", "Motion", "Pulse"), shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'], mode='lines', name='Temperature', line=dict(color='grey')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['motion'], mode='lines', name='Motion', line=dict(color='grey'), line_shape='hv'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['pulse'], mode='lines', name='Pulse', line=dict(color='grey')), row=3, col=1)

    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
        
        anomalous_preds = pred_df[pred_df['prediction'] != 'normal']
        merged_anomalies = pd.merge_asof(anomalous_preds.sort_values('timestamp'), df.sort_values('timestamp'), on='timestamp', direction='nearest', tolerance=pd.Timedelta('1min'))

        for pred_type in merged_anomalies['prediction'].unique():
            type_preds = merged_anomalies[merged_anomalies['prediction'] == pred_type]
            color = ANOMALY_COLORS.get(pred_type, "#95A5A6")
            display_name = ANOMALY_DISPLAY_NAMES.get(pred_type, pred_type)
            fig.add_trace(go.Scatter(x=type_preds['timestamp'], y=type_preds['temperature'], mode='markers', name=f"Pred: {display_name}", marker=dict(color=color, size=8, symbol='circle')), row=1, col=1)
            fig.add_trace(go.Scatter(x=type_preds['timestamp'], y=type_preds['motion'], mode='markers', name=f"Pred: {display_name}", marker=dict(color=color, size=8, symbol='circle'), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=type_preds['timestamp'], y=type_preds['pulse'], mode='markers', name=f"Pred: {display_name}", marker=dict(color=color, size=8, symbol='circle'), showlegend=False), row=3, col=1)

    fig.update_layout(height=600, title_text="Sensor Data with Model Predictions", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Streamlit Page Functions ---

def display_api_status_header():
    """Check and display the API status at the top of the page."""
    is_healthy, _ = check_api_health()
    if not is_healthy:
        st.error("⚠️ Backend API is not running. Please start the server by running `python backend/main.py` in your terminal.", icon="🚨")
        st.stop()
    else:
        st.success("✅ Backend API is running and healthy.", icon="🚀")

def create_dashboard_page():
    """Create the main dashboard page for live monitoring and batch processing."""
    st.title("📡 WSN Anomaly Detection Dashboard")
    
    st.header("Live Simulation & Analysis")
    st.markdown("Use the `/simulate` endpoint to generate data and `/predict/batch` to analyze it.")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    num_samples = col1.number_input("Number of samples", 50, 2000, 200, 50)
    include_anomalies = col2.checkbox("Inject anomalies", True)
    if col3.button("▶️ Run Simulation & Analysis", use_container_width=True):
        with st.spinner("Generating simulation data via `/simulate`..."):
            sim_response = generate_simulation_data(num_samples, include_anomalies)
        if sim_response and 'data' in sim_response:
            df = pd.DataFrame(sim_response['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.session_state['simulation_data'] = df
            
            with st.spinner("Analyzing data via `/predict/batch`..."):
                batch_response = predict_batch(df.to_dict('records'))
            if batch_response and 'predictions' in batch_response:
                st.session_state['predictions'] = batch_response['predictions']
                st.session_state['batch_summary'] = batch_response.get('summary', {})
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
        
        with st.expander("View Batch Prediction Summary & Details"):
            summary = st.session_state.get('batch_summary', {})
            if summary:
                st.write("#### Prediction Summary")
                st.json(summary)
            
            st.write("#### Detected Anomaly Details")
            anomalies = [p for p in predictions if p['prediction'] != 'normal']
            if anomalies:
                anomalies_df = pd.DataFrame(anomalies)
                anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(anomalies_df[['timestamp', 'prediction', 'confidence']], use_container_width=True)
            else:
                st.info("No anomalies were detected in this batch.")

    st.header("Batch Processing from File")
    uploaded_file = st.file_uploader("Upload a CSV file with sensor data", type="csv")
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df_upload.head())
        if st.button("Analyze Uploaded File"):
            required_cols = ['temperature', 'motion', 'pulse']
            if all(col in df_upload.columns for col in required_cols):
                if 'timestamp' not in df_upload.columns:
                    df_upload['timestamp'] = [datetime.now().isoformat() for _ in range(len(df_upload))]
                
                with st.spinner("Analyzing uploaded file..."):
                    batch_response = predict_batch(df_upload.to_dict('records'))
                if batch_response:
                    st.success("Analysis complete!")
                    st.plotly_chart(plot_prediction_results(df_upload, batch_response['predictions']), use_container_width=True)
                    st.json(batch_response.get('summary'))
            else:
                st.error(f"CSV must contain the columns: {', '.join(required_cols)}")

def create_manual_test_page():
    """Create the page for manually testing single data points via /predict."""
    st.title("🔬 Manual Anomaly Test")
    st.markdown("Use the `/predict` endpoint to analyze a single sensor reading.")

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
            color = ANOMALY_COLORS.get(prediction, "#95A5A6").lstrip('#')
            
            # Simple colored box for the result
            st.markdown(f"""
            <div style="background-color:#{color}22; padding: 20px; border-radius: 10px; border-left: 6px solid #{color};">
                <h3 style="color:#{color}; margin:0;">Result: {display_name}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Confidence Score", f"{confidence:.4f}")

            with st.expander("View Detailed Results"):
                if 'probabilities' in result:
                    st.write("#### Model Prediction Probabilities")
                    probs = result['probabilities']
                    if isinstance(probs, dict) and probs:
                        prob_data = []
                        for k, v in probs.items():
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
                                probs_df, x='Class', y='Probability', color='Class',
                                color_discrete_map={row['Class']: row['Color'] for _, row in probs_df.iterrows()},
                                title="Anomaly Classification Probabilities"
                            )
                            fig.update_layout(yaxis_range=[0,1])
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No valid probability data to display.")
                    else:
                        st.warning("Invalid probability data format received from API.")
                
                if 'features' in result:
                    st.write("#### Extracted Features")
                    st.json(result['features'])

def create_system_info_page():
    """Create a page to display system status and model info."""
    st.title("ℹ️ System Information")
    st.markdown("This page uses the `/`, `/health`, `/status`, and `/models` endpoints to provide information about the backend.")

    # Check root endpoint
    st.subheader("Root Endpoint (`/`)")
    is_root_ok, root_data = check_api_root()
    if is_root_ok:
        st.success("Root endpoint is accessible.")
        st.json(root_data)
    else:
        st.error("Could not reach the root endpoint.")

    # Check health endpoint
    st.subheader("Health Check (`/health`)")
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("API is healthy.")
        st.json(health_data)
    else:
        st.error("API is unhealthy or unreachable.")
        st.json(health_data)

    # Get system status
    st.subheader("System Status (`/status`)")
    system_status = get_system_status()
    if system_status:
        col1, col2, col3 = st.columns(3)
        uptime_s = system_status.get('uptime_seconds', 0)
        uptime_str = f"{int(uptime_s // 3600)}h {int((uptime_s % 3600) // 60)}m {int(uptime_s % 60)}s"
        col1.metric("API Uptime", uptime_str)
        col2.metric("CPU Usage", f"{system_status.get('cpu_usage_percent', 0):.1f}%")
        col3.metric("Memory Usage", f"{system_status.get('memory_usage_mb', 0):.1f} MB")
        
        st.write("##### Model Loading Status")
        st.json(system_status.get('models_loaded', {}))
    else:
        st.warning("Could not retrieve system status.")
    
    # Get model info
    st.subheader("Loaded Models (`/models`)")
    models_info = get_models_info()
    if models_info:
        for model in models_info:
            with st.container(border=True):
                st.write(f"**Model Name:** `{model.get('model_name')}`")
                st.write(f"**Model Type:** `{model.get('model_type')}`")
                st.write("**Features Used:**")
                st.write(model.get('features_used', []))
    else:
        st.warning("No model information available. Please ensure models are trained and loaded.")

# --- Main Application ---
def main():
    st.set_page_config(page_title="WSN Anomaly Detection", page_icon="📡", layout="wide")

    st.sidebar.title("WSN Anomaly Detection")
    page_options = {
        "Dashboard": create_dashboard_page,
        "Manual Test": create_manual_test_page,
        "System Info": create_system_info_page
    }
    page_selection = st.sidebar.radio("Navigation", list(page_options.keys()))

    st.sidebar.markdown("---")
    st.sidebar.info("This application provides a frontend for the WSN Anomaly Detection API.")
    st.sidebar.info("Version 2.0 | © 2025")

    # Display API status on every page
    display_api_status_header()

    # Run the selected page function
    page_options[page_selection]()

if __name__ == "__main__":
    main()
