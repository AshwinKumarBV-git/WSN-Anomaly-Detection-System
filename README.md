# WSN Anomaly Detection System

A comprehensive simulation-based anomaly detection system for Wireless Sensor Networks (WSNs) using temperature, motion, and pulse sensors. The system detects 5 types of anomalies (DoS, jamming, tampering, hardware faults, environmental noise) and serves predictions via a FastAPI backend with a Streamlit frontend.

## 🧱 Project Structure

```
wsn_anomaly_project/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── models/                 # Trained models
│   │   ├── sklearn_model.pkl   # Traditional ML model
│   │   └── tf_autoencoder.h5   # Deep learning model
│   ├── preprocessing/
│   │   ├── scaler.pkl          # Feature scaler
│   │   └── feature_extractor.py # Feature extraction
│   ├── services/
│   │   ├── predictor.py        # Prediction service
│   │   └── data_validator.py   # Data validation
│   └── utils/
│       └── logger.py           # Logging utility
│
├── frontend/
│   └── streamlit_app.py       # Streamlit UI
│
├── simulation/
│   ├── simulate_data.py        # Data simulation
│   └── inject_anomalies.py     # Anomaly injection
│
├── training/
│   ├── train_sklearn.py        # Train traditional ML
│   ├── train_autoencoder.py    # Train deep learning
│   └── evaluate_models.py      # Model evaluation
│
├── data/                       # Data storage
│   └── simulated_sensor_data.csv
├── notebooks/                  # Analysis notebooks
│   └── analysis.ipynb
└── requirements.txt            # Dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository or download the source code

2. Navigate to the project directory

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the System

#### 1. Generate Simulated Data

First, generate the simulated sensor data:

```bash
python simulation/simulate_data.py
```

Then, inject anomalies into the data:

```bash
python simulation/inject_anomalies.py
```

#### 2. Train the Models

Train the traditional machine learning model:

```bash
python training/train_sklearn.py
```

Train the deep learning autoencoder model:

```bash
python training/train_autoencoder.py
```

Evaluate the models:

```bash
python training/evaluate_models.py
```

#### 3. Start the Backend Server

Start the FastAPI backend server:

```bash
cd backend
python main.py
```

The API will be available at http://localhost:8000

#### 4. Launch the Frontend

Start the Streamlit frontend application:

```bash
cd frontend
streamlit run streamlit_app.py
```

The UI will be available at http://localhost:8501

## 📊 System Features

### Anomaly Types

The system can detect the following types of anomalies:

1. **Denial of Service (DoS)**: Attacks that prevent legitimate users from accessing the network
2. **Jamming**: Interference with the wireless communication channel
3. **Tampering**: Physical or logical manipulation of sensor nodes
4. **Hardware Faults**: Malfunctions in sensor hardware
5. **Environmental Noise**: Unusual environmental conditions affecting sensor readings

### Machine Learning Models

The system uses an ensemble of models for anomaly detection:

- **Traditional ML Model**: Isolation Forest, One-Class SVM, or Random Forest
- **Deep Learning Model**: Autoencoder for reconstruction-based anomaly detection

### API Endpoints

- `/predict`: Single prediction
- `/predict-batch`: Batch prediction
- `/simulate`: Generate simulation data
- `/health`: API health check
- `/status`: System status
- `/model-info`: Model information

### Frontend Features

- **Live Monitoring**: Simulate sensor data and detect anomalies in real-time
- **Manual Testing**: Test specific sensor values for anomalies
- **Batch Upload**: Upload CSV files with sensor data for batch processing
- **Visualization**: Interactive plots of sensor data and anomaly scores

## 🔧 Technical Details

### Data Simulation

The simulation module generates realistic time-series data for 3 sensors using sinusoids, noise, and event-driven pulses. It can inject 5 types of labeled anomalies with configurable parameters.

### Feature Extraction

The system uses sliding window feature extraction to compute statistical features (mean, std, min, max), frequency domain features (FFT), and cross-sensor features (correlations).

### Anomaly Detection

The anomaly detection is performed using an ensemble approach:

1. **Sklearn Model**: Traditional machine learning for anomaly detection
2. **Autoencoder**: Deep learning model that learns to reconstruct normal data
3. **Ensemble**: Combination of both models for robust detection

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This project was created as a demonstration of anomaly detection in WSNs
- Inspired by real-world applications in IoT security and monitoring