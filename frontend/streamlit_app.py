import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
API_URL = "http://localhost:8000"
ANOMALY_TYPES = {
    "normal": "Normal",
    "dos": "Denial of Service (DoS)",
    "jamming": "Jamming",
    "tampering": "Tampering",
    "hardware_fault": "Hardware Fault",
    "environmental_noise": "Environmental Noise"
}
ANOMALY_COLORS = {
    "normal": "#2ECC71",  # Green
    "dos": "#E74C3C",  # Red
    "jamming": "#F39C12",  # Orange
    "tampering": "#9B59B6",  # Purple
    "hardware_fault": "#3498DB",  # Blue
    "environmental_noise": "#1ABC9C"  # Teal
}


# Helper functions
def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_system_status():
    """Get system status from API"""
    try:
        response = requests.get(f"{API_URL}/status")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def predict_single(sensor_data):
    """Send a single prediction request to the API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=sensor_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None


def predict_batch(sensor_data_list):
    """Send a batch prediction request to the API"""
    try:
        response = requests.post(f"{API_URL}/predict-batch", json={"data": sensor_data_list})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None


def generate_simulation_data(num_samples=100, include_anomalies=True):
    """Generate simulation data from the API"""
    try:
        params = {
            "num_samples": num_samples,
            "include_anomalies": include_anomalies
        }
        response = requests.get(f"{API_URL}/simulate", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None


def load_local_data(file_path):
    """Load data from a local CSV file"""
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def plot_sensor_data(df, anomalies=None):
    """Plot sensor data with anomalies highlighted"""
    # Create subplots
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=("Temperature", "Motion", "Pulse"),
                        shared_xaxes=True,
                        vertical_spacing=0.1)
    
    # Add temperature trace
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['temperature'], mode='lines', name='Temperature'),
        row=1, col=1
    )
    
    # Add motion trace
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['motion'], mode='lines', name='Motion'),
        row=2, col=1
    )
    
    # Add pulse trace
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['pulse'], mode='lines', name='Pulse'),
        row=3, col=1
    )
    
    # If anomalies are provided, highlight them
    if anomalies is not None and not anomalies.empty:
        # Group anomalies by type
        for anomaly_type in anomalies['anomaly_type'].unique():
            type_anomalies = anomalies[anomalies['anomaly_type'] == anomaly_type]
            
            # Get color for this anomaly type
            color = ANOMALY_COLORS.get(anomaly_type, "#95A5A6")  # Default to gray if type not found
            
            # Add temperature anomalies
            fig.add_trace(
                go.Scatter(x=type_anomalies['timestamp'], y=type_anomalies['temperature'], 
                           mode='markers', name=f'{ANOMALY_TYPES.get(anomaly_type, anomaly_type)} (Temp)', 
                           marker=dict(color=color, size=10, symbol='x')),
                row=1, col=1
            )
            
            # Add motion anomalies
            fig.add_trace(
                go.Scatter(x=type_anomalies['timestamp'], y=type_anomalies['motion'], 
                           mode='markers', name=f'{ANOMALY_TYPES.get(anomaly_type, anomaly_type)} (Motion)', 
                           marker=dict(color=color, size=10, symbol='x'),
                           showlegend=False),  # Don't show duplicate in legend
                row=2, col=1
            )
            
            # Add pulse anomalies
            fig.add_trace(
                go.Scatter(x=type_anomalies['timestamp'], y=type_anomalies['pulse'], 
                           mode='markers', name=f'{ANOMALY_TYPES.get(anomaly_type, anomaly_type)} (Pulse)', 
                           marker=dict(color=color, size=10, symbol='x'),
                           showlegend=False),  # Don't show duplicate in legend
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Sensor Data with Anomalies",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="Level", row=2, col=1)
    fig.update_yaxes(title_text="BPM", row=3, col=1)
    
    # Update x-axis label
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    return fig


def plot_anomaly_distribution(anomalies):
    """Plot distribution of anomaly types"""
    if anomalies is None or anomalies.empty:
        return None
    
    # Count anomalies by type
    anomaly_counts = anomalies['anomaly_type'].value_counts().reset_index()
    anomaly_counts.columns = ['anomaly_type', 'count']
    
    # Map anomaly types to display names
    anomaly_counts['display_name'] = anomaly_counts['anomaly_type'].map(ANOMALY_TYPES)
    
    # Map anomaly types to colors
    anomaly_counts['color'] = anomaly_counts['anomaly_type'].map(ANOMALY_COLORS)
    
    # Create bar chart
    fig = px.bar(anomaly_counts, x='display_name', y='count', 
                 title='Anomaly Type Distribution',
                 labels={'display_name': 'Anomaly Type', 'count': 'Count'},
                 color='anomaly_type',
                 color_discrete_map=ANOMALY_COLORS)
    
    # Update layout
    fig.update_layout(
        xaxis_title="Anomaly Type",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig


def plot_anomaly_scores(predictions):
    """Plot anomaly scores over time"""
    if predictions is None or len(predictions) == 0:
        return None
    
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions)
    
    # Create line chart
    fig = px.line(df, x='timestamp', y='anomaly_score', 
                  title='Anomaly Score Over Time',
                  labels={'timestamp': 'Time', 'anomaly_score': 'Anomaly Score'})
    
    # Add threshold line if available
    if 'threshold' in df.columns and not df['threshold'].isna().all():
        threshold = df['threshold'].iloc[0]
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Threshold: {threshold:.3f}",
                     annotation_position="bottom right")
    
    # Color points based on anomaly status
    fig.add_trace(
        go.Scatter(x=df[df['is_anomaly']]['timestamp'], 
                   y=df[df['is_anomaly']]['anomaly_score'],
                   mode='markers',
                   marker=dict(color='red', size=10),
                   name='Anomaly')
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Anomaly Score"
    )
    
    return fig


def create_live_monitoring_page():
    """Create the live monitoring page"""
    st.title("Live WSN Monitoring")
    
    # Check API status
    api_running = check_api_status()
    if not api_running:
        st.error("⚠️ API is not running. Please start the backend server.")
        st.info("Run the following command in a terminal:")
        st.code("cd backend && python main.py")
        return
    
    st.success("✅ Connected to API")
    
    # Get system status
    system_status = get_system_status()
    if system_status:
        # Display system metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU Usage", f"{system_status['cpu_percent']}%")
        with col2:
            st.metric("Memory Usage", f"{system_status['memory_percent']}%")
        with col3:
            st.metric("Uptime", f"{system_status['uptime_seconds'] // 3600}h {(system_status['uptime_seconds'] % 3600) // 60}m")
    
    # Get model info
    model_info = get_model_info()
    if model_info:
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Sklearn Model**: {model_info['sklearn_model_type']}")
            st.info(f"**Sklearn Accuracy**: {model_info['sklearn_metrics'].get('accuracy', 'N/A'):.4f}")
        with col2:
            st.info(f"**Autoencoder Type**: {model_info['autoencoder_type']}")
            st.info(f"**Autoencoder Accuracy**: {model_info['autoencoder_metrics'].get('accuracy', 'N/A'):.4f}")
    
    # Simulation controls
    st.subheader("Simulation Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_samples = st.number_input("Number of samples", min_value=10, max_value=1000, value=100, step=10)
    with col2:
        include_anomalies = st.checkbox("Include anomalies", value=True)
    with col3:
        if st.button("Generate Data"):
            with st.spinner("Generating simulation data..."):
                simulation_data = generate_simulation_data(num_samples, include_anomalies)
                if simulation_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(simulation_data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Store in session state
                    st.session_state['simulation_data'] = df
                    
                    # If anomalies are included, extract them
                    if include_anomalies and 'anomalies' in simulation_data:
                        anomalies_df = pd.DataFrame(simulation_data['anomalies'])
                        anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp'])
                        st.session_state['anomalies'] = anomalies_df
                    else:
                        st.session_state['anomalies'] = pd.DataFrame()
                    
                    st.success(f"Generated {len(df)} samples")
    
    # Display simulation data if available
    if 'simulation_data' in st.session_state:
        st.subheader("Sensor Data Visualization")
        
        # Plot sensor data
        fig = plot_sensor_data(st.session_state['simulation_data'], 
                              st.session_state.get('anomalies', None))
        st.plotly_chart(fig, use_container_width=True)
        
        # If anomalies are available, show distribution
        if 'anomalies' in st.session_state and not st.session_state['anomalies'].empty:
            st.subheader("Anomaly Distribution")
            fig = plot_anomaly_distribution(st.session_state['anomalies'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Run anomaly detection
        st.subheader("Anomaly Detection")
        if st.button("Run Anomaly Detection"):
            with st.spinner("Running anomaly detection..."):
                # Prepare data for batch prediction
                sensor_data_list = st.session_state['simulation_data'].to_dict('records')
                
                # Send batch prediction request
                predictions = predict_batch(sensor_data_list)
                
                if predictions and 'results' in predictions:
                    # Store predictions in session state
                    st.session_state['predictions'] = predictions['results']
                    
                    # Count anomalies
                    anomaly_count = sum(1 for p in predictions['results'] if p['is_anomaly'])
                    
                    st.success(f"Detected {anomaly_count} anomalies out of {len(predictions['results'])} samples")
        
        # Display predictions if available
        if 'predictions' in st.session_state:
            # Plot anomaly scores
            fig = plot_anomaly_scores(st.session_state['predictions'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomaly details
            st.subheader("Anomaly Details")
            
            # Filter only anomalies
            anomalies = [p for p in st.session_state['predictions'] if p['is_anomaly']]
            
            if anomalies:
                # Convert to DataFrame for display
                anomalies_df = pd.DataFrame(anomalies)
                
                # Format timestamp
                if 'timestamp' in anomalies_df.columns:
                    anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Select columns to display
                display_cols = ['timestamp', 'anomaly_type', 'anomaly_score', 'temperature', 'motion', 'pulse']
                display_cols = [col for col in display_cols if col in anomalies_df.columns]
                
                # Display table
                st.dataframe(anomalies_df[display_cols], use_container_width=True)
            else:
                st.info("No anomalies detected")


def create_manual_testing_page():
    """Create the manual testing page"""
    st.title("Manual Anomaly Testing")
    
    # Check API status
    api_running = check_api_status()
    if not api_running:
        st.error("⚠️ API is not running. Please start the backend server.")
        st.info("Run the following command in a terminal:")
        st.code("cd backend && python main.py")
        return
    
    st.success("✅ Connected to API")
    
    # Manual input form
    st.subheader("Sensor Data Input")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    with col2:
        motion = st.slider("Motion (0-100)", min_value=0, max_value=100, value=10)
    with col3:
        pulse = st.slider("Pulse (BPM)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    
    # Add timestamp
    timestamp = datetime.now().isoformat()
    
    # Create sensor data
    sensor_data = {
        "timestamp": timestamp,
        "temperature": temperature,
        "motion": motion,
        "pulse": pulse
    }
    
    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction = predict_single(sensor_data)
            
            if prediction:
                # Display result
                st.subheader("Prediction Result")
                
                # Create columns for display
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display anomaly status with color
                    if prediction['is_anomaly']:
                        st.error("⚠️ Anomaly Detected")
                    else:
                        st.success("✅ Normal")
                    
                    # Display anomaly type if available
                    if prediction['is_anomaly'] and 'anomaly_type' in prediction:
                        anomaly_type = prediction['anomaly_type']
                        display_type = ANOMALY_TYPES.get(anomaly_type, anomaly_type)
                        st.info(f"Type: {display_type}")
                
                with col2:
                    # Display anomaly score
                    st.metric("Anomaly Score", f"{prediction['anomaly_score']:.4f}")
                    
                    # Display threshold if available
                    if 'threshold' in prediction:
                        st.metric("Threshold", f"{prediction['threshold']:.4f}")
                
                # Display model contributions if available
                if 'model_contributions' in prediction:
                    st.subheader("Model Contributions")
                    
                    contribs = prediction['model_contributions']
                    
                    # Create bar chart
                    contrib_df = pd.DataFrame({
                        'Model': list(contribs.keys()),
                        'Score': list(contribs.values())
                    })
                    
                    fig = px.bar(contrib_df, x='Model', y='Score', 
                                 title='Model Contribution to Anomaly Score',
                                 labels={'Model': 'Model', 'Score': 'Contribution'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store prediction in history
                if 'prediction_history' not in st.session_state:
                    st.session_state['prediction_history'] = []
                
                # Add timestamp to prediction for history
                prediction['timestamp'] = timestamp
                
                # Add input values to prediction for history
                prediction['temperature'] = temperature
                prediction['motion'] = motion
                prediction['pulse'] = pulse
                
                # Add to history (at the beginning)
                st.session_state['prediction_history'].insert(0, prediction)
    
    # Display prediction history
    if 'prediction_history' in st.session_state and st.session_state['prediction_history']:
        st.subheader("Prediction History")
        
        # Convert to DataFrame for display
        history_df = pd.DataFrame(st.session_state['prediction_history'])
        
        # Format timestamp
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Select columns to display
        display_cols = ['timestamp', 'is_anomaly', 'anomaly_type', 'anomaly_score', 
                       'temperature', 'motion', 'pulse']
        display_cols = [col for col in display_cols if col in history_df.columns]
        
        # Display table
        st.dataframe(history_df[display_cols], use_container_width=True)
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state['prediction_history'] = []
            st.experimental_rerun()


def create_batch_upload_page():
    """Create the batch upload page"""
    st.title("Batch Data Upload")
    
    # Check API status
    api_running = check_api_status()
    if not api_running:
        st.error("⚠️ API is not running. Please start the backend server.")
        st.info("Run the following command in a terminal:")
        st.code("cd backend && python main.py")
        return
    
    st.success("✅ Connected to API")
    
    # File upload
    st.subheader("Upload Data File")
    uploaded_file = st.file_uploader("Upload a CSV file with sensor data", type="csv")
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['timestamp', 'temperature', 'motion', 'pulse']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("The CSV file must contain the following columns: timestamp, temperature, motion, pulse")
                return
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store in session state
            st.session_state['uploaded_data'] = df
            
            st.success(f"Loaded {len(df)} samples")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Plot sensor data
            st.subheader("Sensor Data Visualization")
            fig = plot_sensor_data(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Run anomaly detection
            st.subheader("Anomaly Detection")
            if st.button("Run Anomaly Detection"):
                with st.spinner("Running anomaly detection..."):
                    # Prepare data for batch prediction
                    sensor_data_list = df.to_dict('records')
                    
                    # Send batch prediction request
                    predictions = predict_batch(sensor_data_list)
                    
                    if predictions and 'results' in predictions:
                        # Store predictions in session state
                        st.session_state['batch_predictions'] = predictions['results']
                        
                        # Count anomalies
                        anomaly_count = sum(1 for p in predictions['results'] if p['is_anomaly'])
                        
                        st.success(f"Detected {anomaly_count} anomalies out of {len(predictions['results'])} samples")
            
            # Display predictions if available
            if 'batch_predictions' in st.session_state:
                # Create DataFrame with predictions
                pred_df = pd.DataFrame(st.session_state['batch_predictions'])
                pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
                
                # Plot anomaly scores
                fig = plot_anomaly_scores(st.session_state['batch_predictions'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Extract anomalies
                anomalies = pred_df[pred_df['is_anomaly']].copy()
                
                if not anomalies.empty:
                    # Plot sensor data with anomalies
                    st.subheader("Sensor Data with Anomalies")
                    
                    # Merge with original data to get sensor values
                    merged_df = pd.merge(anomalies, df, on='timestamp')
                    
                    # Plot
                    fig = plot_sensor_data(df, merged_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show anomaly details
                    st.subheader("Anomaly Details")
                    
                    # Format timestamp for display
                    display_df = anomalies.copy()
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Select columns to display
                    display_cols = ['timestamp', 'anomaly_type', 'anomaly_score']
                    display_cols = [col for col in display_cols if col in display_df.columns]
                    
                    # Display table
                    st.dataframe(display_df[display_cols], use_container_width=True)
                    
                    # Option to download results
                    st.download_button(
                        label="Download Anomaly Results",
                        data=display_df.to_csv(index=False),
                        file_name="anomaly_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No anomalies detected")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def create_about_page():
    """Create the about page"""
    st.title("About WSN Anomaly Detection System")
    
    st.markdown("""
    ## System Overview
    
    This system provides real-time anomaly detection for Wireless Sensor Networks (WSNs) using 
    machine learning techniques. It can detect various types of anomalies in sensor data, including:
    
    - **Denial of Service (DoS)**: Attacks that prevent legitimate users from accessing the network
    - **Jamming**: Interference with the wireless communication channel
    - **Tampering**: Physical or logical manipulation of sensor nodes
    - **Hardware Faults**: Malfunctions in sensor hardware
    - **Environmental Noise**: Unusual environmental conditions affecting sensor readings
    
    ## Architecture
    
    The system consists of the following components:
    
    1. **Simulation Module**: Generates realistic sensor data and injects anomalies
    2. **Backend API**: FastAPI-based service for anomaly detection
    3. **Machine Learning Models**: Ensemble of traditional ML and deep learning models
    4. **Frontend**: This Streamlit application for visualization and interaction
    
    ## Models
    
    The system uses an ensemble of models for anomaly detection:
    
    - **Traditional ML Model**: Isolation Forest, One-Class SVM, or Random Forest
    - **Deep Learning Model**: Autoencoder for reconstruction-based anomaly detection
    
    ## How to Use
    
    1. **Live Monitoring**: Simulate sensor data and detect anomalies in real-time
    2. **Manual Testing**: Test specific sensor values for anomalies
    3. **Batch Upload**: Upload CSV files with sensor data for batch processing
    
    ## Data Format
    
    The system expects sensor data with the following fields:
    
    - `timestamp`: Date and time of the reading
    - `temperature`: Temperature reading in Celsius
    - `motion`: Motion level (0-100)
    - `pulse`: Pulse reading in BPM
    
    ## API Endpoints
    
    - `/predict`: Single prediction
    - `/predict-batch`: Batch prediction
    - `/simulate`: Generate simulation data
    - `/health`: API health check
    - `/status`: System status
    - `/model-info`: Model information
    """)


def main():
    # Set page config
    st.set_page_config(
        page_title="WSN Anomaly Detection",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("WSN Anomaly Detection")
    page = st.sidebar.radio("Navigation", ["Live Monitoring", "Manual Testing", "Batch Upload", "About"])
    
    # Display selected page
    if page == "Live Monitoring":
        create_live_monitoring_page()
    elif page == "Manual Testing":
        create_manual_testing_page()
    elif page == "Batch Upload":
        create_batch_upload_page()
    elif page == "About":
        create_about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "WSN Anomaly Detection System - v1.0\n\n"
        "© 2023 - All Rights Reserved"
    )


if __name__ == "__main__":
    main()