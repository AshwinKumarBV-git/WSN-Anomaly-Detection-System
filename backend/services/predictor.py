import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from typing import Dict, List, Union, Any, Tuple

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.feature_extractor import FeatureExtractor

def sanitize_for_json(data):
    """
    Recursively clean a dictionary or list of NaN/inf values
    that are not compliant with the JSON standard.
    """
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(i) for i in data]
    if isinstance(data, (np.float32, np.float64, float)):
        if np.isnan(data) or np.isinf(data):
            return None  # Replace with null in JSON
        return float(data)
    if isinstance(data, (np.int32, np.int64, int)):
        return int(data)
    return data

class AnomalyPredictor:
    """
    Anomaly predictor for WSN sensor data.
    Loads trained models and makes predictions on new data.
    """
    
    def __init__(self, models_dir="../models", window_size=30):
        """Initialize the predictor."""
        self.models_dir = models_dir
        self.window_size = window_size
        self.sklearn_model = None
        self.sklearn_model_type = 'unknown'
        self.autoencoder_model = None
        self.feature_extractor = None
        self.anomaly_thresholds = {}
        self.class_mapping = {
            0: "normal", 1: "DoS", 2: "Jamming", 3: "Tampering",
            4: "HardwareFault", 5: "EnvironmentalNoise"
        }
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk."""
        os.makedirs(self.models_dir, exist_ok=True)
        
        sklearn_path = os.path.join(self.models_dir, "sklearn_model.pkl")
        if os.path.exists(sklearn_path):
            try:
                with open(sklearn_path, 'rb') as f:
                    model_info = pickle.load(f)
                
                # Robustly load the model whether it's in a dict or saved directly.
                if isinstance(model_info, dict):
                    self.sklearn_model = model_info['model']
                    self.sklearn_model_type = model_info.get('model_type', 'unknown')
                else:
                    self.sklearn_model = model_info
                    self.sklearn_model_type = 'random_forest' if hasattr(self.sklearn_model, 'predict_proba') else 'isolation_forest'

                print(f"Loaded sklearn model ({self.sklearn_model_type}) from {sklearn_path}")
            except Exception as e:
                print(f"Error loading sklearn model: {e}")
        else:
            print(f"Sklearn model not found at {sklearn_path}")
        
        autoencoder_path = os.path.join(self.models_dir, "tf_autoencoder.h5")
        if os.path.exists(autoencoder_path):
            try:
                self.autoencoder_model = tf.keras.models.load_model(autoencoder_path, compile=False)
                print(f"Loaded autoencoder model from {autoencoder_path}")
            except Exception as e:
                print(f"Error loading autoencoder model: {e}")
        else:
            print(f"Autoencoder model not found at {autoencoder_path}")
        
        thresholds_path = os.path.join(self.models_dir, "anomaly_thresholds.pkl")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'rb') as f: self.anomaly_thresholds = pickle.load(f)
            print(f"Loaded anomaly thresholds from {thresholds_path}")
        
        if 'autoencoder' not in self.anomaly_thresholds: self.anomaly_thresholds['autoencoder'] = 0.1
        
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.feature_extractor = FeatureExtractor(
            window_size=self.window_size,
            overlap=0.5,
            scaler_path=scaler_path if os.path.exists(scaler_path) else None
        )

    def prepare_data(self, data: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """Prepare input data for prediction."""
        df = pd.DataFrame([data] if isinstance(data, dict) else data)
        if 'timestamp' not in df.columns: df['timestamp'] = datetime.now()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df

    def extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features from a window of data."""
        if len(data) < self.window_size:
            padding = pd.concat([data.iloc[[0]]] * (self.window_size - len(data)), ignore_index=True)
            data = pd.concat([padding, data], ignore_index=True)
        return self.feature_extractor.extract_features_from_single_window(data.tail(self.window_size), scale=True)

    def predict_with_sklearn(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction using the sklearn model."""
        if self.sklearn_model is None or self.feature_extractor.scaler is None:
            return "unknown", 0.0, {}
            
        # Reindex the incoming features to match the exact order the model was trained on.
        feature_names = self.feature_extractor.scaler.get_feature_names_out()
        feature_df = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
        
        try:
            if self.sklearn_model_type == 'random_forest':
                probas = self.sklearn_model.predict_proba(feature_df)[0]
                pred_idx = np.argmax(probas)
                # **FIX**: The probabilities dictionary should have integer keys for the plot to work correctly.
                # The frontend will map these integers to names.
                probabilities = {i: p for i, p in enumerate(probas)}
                return self.class_mapping.get(pred_idx, "unknown"), probas[pred_idx], probabilities
            else: # Anomaly detection models like IsolationForest
                pred = self.sklearn_model.predict(feature_df)[0]
                prediction = "normal" if pred == 1 else "anomaly"
                score = self.sklearn_model.decision_function(feature_df)[0]
                confidence = 1.0 / (1.0 + np.exp(-score)) # Normalize score
                return prediction, confidence, {"normal": confidence, "anomaly": 1 - confidence}
        except Exception as e:
            print(f"Error during sklearn prediction: {e}")
            return "error", 0.0, {"error": str(e)}

    def predict_with_autoencoder(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction using the autoencoder model."""
        if self.autoencoder_model is None or self.feature_extractor.scaler is None:
            return "unknown", 0.0, {}

        feature_names = self.feature_extractor.scaler.get_feature_names_out()
        feature_vector = np.array([[features.get(k, 0) for k in feature_names]])

        try:
            recon = self.autoencoder_model.predict(feature_vector, verbose=0)
            mse = np.mean(np.square(feature_vector - recon))
            threshold = self.anomaly_thresholds.get("autoencoder", 0.1)
            is_anomaly = mse > threshold
            score = min(1.0, mse / (2 * threshold))
            prediction = "anomaly" if is_anomaly else "normal"
            confidence = score if is_anomaly else 1.0 - score
            return prediction, confidence, {"reconstruction_error": mse, "anomaly_score": score}
        except Exception as e:
            return "error", 0.0, {"error": str(e)}

    def ensemble_prediction(self, sklearn_res, autoencoder_res):
        """Combine predictions from multiple models."""
        sk_pred, sk_conf, sk_probs = sklearn_res
        ae_pred, ae_conf, _ = autoencoder_res
        
        if self.sklearn_model_type == 'random_forest' and sk_conf > 0.7:
            return sk_pred, sk_conf, sk_probs
        if (sk_pred != 'normal' and ae_pred == 'anomaly'):
            return sk_pred, (sk_conf + ae_conf) / 2, sk_probs
        if ae_pred == 'anomaly' and ae_conf > 0.9:
            return 'anomaly', ae_conf, {'anomaly': ae_conf}
        return sk_pred, sk_conf, sk_probs

    def predict(self, data: Union[Dict, List, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """Make prediction on raw input data."""
        df = self.prepare_data(data)
        features = self.extract_features(df)
        
        sk_res = self.predict_with_sklearn(features)
        ae_res = self.predict_with_autoencoder(features)
        
        pred, conf, probs = self.ensemble_prediction(sk_res, ae_res)
        
        result = {"prediction": pred, "confidence": conf, "timestamp": datetime.now()}
        if kwargs.get('return_probabilities'): result["probabilities"] = probs
        if kwargs.get('return_features'):
            metadata = ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']
            result["features"] = {k: v for k, v in features.items() if k not in metadata}
        
        return sanitize_for_json(result)

    def batch_predict(self, data: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make predictions on a batch of data."""
        df = self.prepare_data(data)
        predictions = []
        for i in range(len(df)):
            window = df.iloc[max(0, i - self.window_size + 1):i + 1]
            pred = self.predict(window, **kwargs)
            pred['timestamp'] = df['timestamp'].iloc[i]
            predictions.append(pred)

        summary = {
            "total_predictions": len(predictions),
            "class_distribution": pd.Series([p['prediction'] for p in predictions]).value_counts().to_dict(),
            "average_confidence": np.mean([p['confidence'] for p in predictions if p['confidence'] is not None])
        }
        return sanitize_for_json({"predictions": predictions, "summary": summary})

# Global predictor instance for FastAPI
predictor = AnomalyPredictor()

def predict_anomaly(sensor_data, **kwargs):
    """Wrapper function for the FastAPI endpoint."""
    return predictor.predict(sensor_data.dict(), **kwargs)
