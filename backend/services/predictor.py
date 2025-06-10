import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from typing import Dict, List, Union, Optional, Tuple, Any

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.feature_extractor import FeatureExtractor


class AnomalyPredictor:
    """
    Anomaly predictor for WSN sensor data
    Loads trained models and makes predictions on new data
    """
    
    def __init__(self, models_dir="../models", window_size=30):
        """
        Initialize the predictor
        
        Args:
            models_dir: Directory containing trained models
            window_size: Size of the sliding window for feature extraction
        """
        self.models_dir = models_dir
        self.window_size = window_size
        self.sklearn_model = None
        self.autoencoder_model = None
        self.feature_extractor = None
        self.anomaly_thresholds = None
        self.class_mapping = {
            0: "normal",
            1: "DoS",
            2: "Jamming",
            3: "Tampering",
            4: "HardwareFault",
            5: "EnvironmentalNoise"
        }
        self.inverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        
        # Load models and feature extractor
        self.load_models()
    
    def load_models(self):
        """
        Load trained models from disk
        """
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load sklearn model (if exists)
        sklearn_path = os.path.join(self.models_dir, "sklearn_model.pkl")
        if os.path.exists(sklearn_path):
            try:
                with open(sklearn_path, 'rb') as f:
                    self.sklearn_model = pickle.load(f)
                print(f"Loaded sklearn model from {sklearn_path}")
            except Exception as e:
                print(f"Error loading sklearn model: {e}")
        else:
            print(f"Sklearn model not found at {sklearn_path}")
        
        # Load autoencoder model (if exists)
        autoencoder_path = os.path.join(self.models_dir, "tf_autoencoder.h5")
        if os.path.exists(autoencoder_path):
            try:
                self.autoencoder_model = tf.keras.models.load_model(autoencoder_path)
                print(f"Loaded autoencoder model from {autoencoder_path}")
            except Exception as e:
                print(f"Error loading autoencoder model: {e}")
        else:
            print(f"Autoencoder model not found at {autoencoder_path}")
        
        # Load anomaly thresholds (if exists)
        thresholds_path = os.path.join(self.models_dir, "anomaly_thresholds.pkl")
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'rb') as f:
                    self.anomaly_thresholds = pickle.load(f)
                print(f"Loaded anomaly thresholds from {thresholds_path}")
            except Exception as e:
                print(f"Error loading anomaly thresholds: {e}")
        else:
            print(f"Anomaly thresholds not found at {thresholds_path}")
            # Set default thresholds
            self.anomaly_thresholds = {
                "autoencoder": 0.1,  # Reconstruction error threshold
                "confidence": 0.7    # Confidence threshold for classification
            }
        
        # Initialize feature extractor
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.feature_extractor = FeatureExtractor(
            window_size=self.window_size,
            overlap=0.5,
            scaler_path=scaler_path if os.path.exists(scaler_path) else None
        )
    
    def prepare_data(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare input data for prediction
        
        Args:
            data: Input data (dict, list of dicts, or DataFrame)
            
        Returns:
            DataFrame with prepared data
        """
        # Convert to DataFrame if not already
        if isinstance(data, dict):
            # Single reading
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # List of readings
            df = pd.DataFrame(data)
        else:
            # Already a DataFrame
            df = data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'timestamp' not in df.columns:
            # Add timestamp if not present
            df['timestamp'] = datetime.now()
        
        # Add sensor_id if not present
        if 'sensor_id' not in df.columns:
            df['sensor_id'] = 'unknown'
        
        # Add label column if not present (for feature extraction)
        if 'label' not in df.columns:
            df['label'] = 'unknown'
        
        return df
    
    def extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from input data
        
        Args:
            data: DataFrame with sensor data
            
        Returns:
            Dictionary of extracted features
        """
        # Check if we have enough data for a window
        if len(data) < self.window_size:
            # Pad with repeated data if needed
            padding_needed = self.window_size - len(data)
            padding = pd.concat([data.iloc[[0]]] * padding_needed, ignore_index=True)
            data = pd.concat([padding, data], ignore_index=True)
            print(f"Warning: Input data padded with {padding_needed} repeated entries to reach window size {self.window_size}")
        
        # Use the most recent window if we have more data than needed
        if len(data) > self.window_size:
            data = data.iloc[-self.window_size:].reset_index(drop=True)
        
        # Extract features using the feature extractor
        features = self.feature_extractor.extract_features_from_single_window(data, scale=True)
        
        return features
    
    def predict_with_sklearn(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Make prediction using the sklearn model
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        if self.sklearn_model is None:
            return "unknown", 0.0, {}
        
        # Prepare feature vector (exclude metadata)
        metadata_keys = ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']
        feature_keys = [k for k in features.keys() if k not in metadata_keys]
        feature_vector = np.array([[features[k] for k in feature_keys]])
        
        # Get prediction and probabilities
        try:
            # Check if model has predict_proba method
            if hasattr(self.sklearn_model, 'predict_proba'):
                probas = self.sklearn_model.predict_proba(feature_vector)[0]
                prediction_idx = np.argmax(probas)
                confidence = probas[prediction_idx]
                prediction = self.class_mapping.get(prediction_idx, "unknown")
                
                # Create probabilities dictionary
                probabilities = {self.class_mapping.get(i, f"class_{i}"): float(p) 
                                for i, p in enumerate(probas)}
            else:
                # For models without probabilities (like isolation forest)
                prediction_idx = self.sklearn_model.predict(feature_vector)[0]
                prediction = self.class_mapping.get(prediction_idx, "unknown")
                
                # Use decision function if available for confidence
                if hasattr(self.sklearn_model, 'decision_function'):
                    decision_scores = self.sklearn_model.decision_function(feature_vector)[0]
                    if isinstance(decision_scores, np.ndarray):
                        confidence = float(decision_scores[prediction_idx])
                    else:
                        confidence = float(decision_scores)
                else:
                    confidence = 1.0  # Default confidence
                
                # No probabilities available
                probabilities = {prediction: confidence}
            
            return prediction, confidence, probabilities
            
        except Exception as e:
            print(f"Error in sklearn prediction: {e}")
            return "error", 0.0, {"error": 1.0}
    
    def predict_with_autoencoder(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Make prediction using the autoencoder model
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (prediction, confidence, anomaly_scores)
        """
        if self.autoencoder_model is None:
            return "unknown", 0.0, {}
        
        # Prepare feature vector (exclude metadata)
        metadata_keys = ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']
        feature_keys = [k for k in features.keys() if k not in metadata_keys]
        feature_vector = np.array([[features[k] for k in feature_keys]])
        
        # Get reconstruction
        try:
            reconstruction = self.autoencoder_model.predict(feature_vector)
            
            # Calculate reconstruction error (MSE)
            mse = np.mean(np.square(feature_vector - reconstruction))
            
            # Determine if it's an anomaly based on threshold
            threshold = self.anomaly_thresholds.get("autoencoder", 0.1)
            is_anomaly = mse > threshold
            
            # Calculate normalized anomaly score (0 to 1)
            anomaly_score = min(1.0, mse / (2 * threshold))  # Cap at 1.0
            
            # For autoencoders, we only detect anomaly vs normal, not the specific type
            prediction = "anomaly" if is_anomaly else "normal"
            confidence = anomaly_score if is_anomaly else 1.0 - anomaly_score
            
            # Create anomaly scores dictionary
            anomaly_scores = {
                "reconstruction_error": float(mse),
                "anomaly_score": float(anomaly_score),
                "threshold": float(threshold),
                "normal": float(1.0 - anomaly_score),
                "anomaly": float(anomaly_score)
            }
            
            return prediction, confidence, anomaly_scores
            
        except Exception as e:
            print(f"Error in autoencoder prediction: {e}")
            return "error", 0.0, {"error": 1.0}
    
    def ensemble_prediction(self, sklearn_result: Tuple[str, float, Dict[str, float]],
                           autoencoder_result: Tuple[str, float, Dict[str, float]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Combine predictions from multiple models
        
        Args:
            sklearn_result: Result from sklearn model (prediction, confidence, probabilities)
            autoencoder_result: Result from autoencoder model (prediction, confidence, anomaly_scores)
            
        Returns:
            Tuple of (prediction, confidence, combined_probabilities)
        """
        sklearn_pred, sklearn_conf, sklearn_probs = sklearn_result
        autoencoder_pred, autoencoder_conf, autoencoder_scores = autoencoder_result
        
        # Initialize combined probabilities
        combined_probs = {}
        
        # If both models are available, combine their predictions
        if sklearn_pred != "unknown" and autoencoder_pred != "unknown":
            # Start with sklearn probabilities
            combined_probs.update(sklearn_probs)
            
            # Adjust normal/anomaly probabilities based on autoencoder
            if "normal" in combined_probs:
                combined_probs["normal"] = (combined_probs["normal"] + autoencoder_scores.get("normal", 0.5)) / 2
            else:
                combined_probs["normal"] = autoencoder_scores.get("normal", 0.5)
            
            # If autoencoder detects anomaly with high confidence, boost anomaly classes
            if autoencoder_pred == "anomaly" and autoencoder_conf > 0.8:
                # Reduce normal probability
                combined_probs["normal"] *= 0.5
                
                # Boost anomaly class probabilities
                anomaly_classes = [c for c in combined_probs.keys() if c != "normal" and c != "error"]
                for cls in anomaly_classes:
                    combined_probs[cls] *= 1.5
                
                # Normalize probabilities to sum to 1
                total = sum(combined_probs.values())
                if total > 0:
                    combined_probs = {k: v / total for k, v in combined_probs.items()}
            
            # Get final prediction and confidence
            final_pred = max(combined_probs.items(), key=lambda x: x[1])[0]
            final_conf = combined_probs[final_pred]
            
        # If only sklearn model is available
        elif sklearn_pred != "unknown":
            final_pred = sklearn_pred
            final_conf = sklearn_conf
            combined_probs = sklearn_probs
            
        # If only autoencoder model is available
        elif autoencoder_pred != "unknown":
            final_pred = autoencoder_pred
            final_conf = autoencoder_conf
            combined_probs = autoencoder_scores
            
        # If no models are available
        else:
            final_pred = "unknown"
            final_conf = 0.0
            combined_probs = {"unknown": 1.0}
        
        return final_pred, final_conf, combined_probs
    
    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame], 
                model_type: str = "ensemble",
                return_probabilities: bool = False,
                return_features: bool = False) -> Dict[str, Any]:
        """
        Make prediction on input data
        
        Args:
            data: Input data (dict, list of dicts, or DataFrame)
            model_type: Type of model to use (sklearn, autoencoder, ensemble)
            return_probabilities: Whether to return class probabilities
            return_features: Whether to return extracted features
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare data
        df = self.prepare_data(data)
        
        # Extract features
        features = self.extract_features(df)
        
        # Get predictions from models based on requested type
        if model_type == "sklearn" or model_type == "ensemble":
            sklearn_result = self.predict_with_sklearn(features)
        else:
            sklearn_result = ("unknown", 0.0, {})
        
        if model_type == "autoencoder" or model_type == "ensemble":
            autoencoder_result = self.predict_with_autoencoder(features)
        else:
            autoencoder_result = ("unknown", 0.0, {})
        
        # Combine predictions if using ensemble
        if model_type == "ensemble":
            prediction, confidence, probabilities = self.ensemble_prediction(sklearn_result, autoencoder_result)
        elif model_type == "sklearn":
            prediction, confidence, probabilities = sklearn_result
        else:  # autoencoder
            prediction, confidence, probabilities = autoencoder_result
        
        # Create result dictionary
        result = {
            "prediction": prediction,
            "confidence": float(confidence),
            "timestamp": datetime.now(),
            "sensor_id": df["sensor_id"].iloc[0] if "sensor_id" in df.columns else "unknown"
        }
        
        # Add probabilities if requested
        if return_probabilities:
            result["probabilities"] = {k: float(v) for k, v in probabilities.items()}
        
        # Add features if requested
        if return_features:
            # Filter out metadata
            metadata_keys = ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']
            feature_dict = {k: float(v) for k, v in features.items() if k not in metadata_keys}
            result["features"] = feature_dict
        
        return result
    
    def batch_predict(self, data: List[Dict], model_type: str = "ensemble",
                     return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions on a batch of data
        
        Args:
            data: List of data points
            model_type: Type of model to use
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with batch prediction results and summary
        """
        # Make predictions for each data point
        predictions = []
        for item in data:
            pred = self.predict(item, model_type, return_probabilities)
            predictions.append(pred)
        
        # Calculate summary statistics
        pred_classes = [p["prediction"] for p in predictions]
        class_counts = {cls: pred_classes.count(cls) for cls in set(pred_classes)}
        
        # Calculate average confidence
        avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions) if predictions else 0
        
        # Create summary
        summary = {
            "total_predictions": len(predictions),
            "class_distribution": class_counts,
            "average_confidence": avg_confidence,
            "timestamp": datetime.now()
        }
        
        return {
            "predictions": predictions,
            "summary": summary
        }


# Function to use in FastAPI endpoint
def predict_anomaly(sensor_data, model_type="ensemble", return_probabilities=False, return_features=False):
    """
    Predict anomalies in sensor data (for FastAPI endpoint)
    
    Args:
        sensor_data: Input sensor data
        model_type: Type of model to use
        return_probabilities: Whether to return class probabilities
        return_features: Whether to return extracted features
        
    Returns:
        Dictionary with prediction results
    """
    # Initialize predictor (lazy loading of models)
    predictor = AnomalyPredictor()
    
    # Make prediction
    result = predictor.predict(
        data=sensor_data.dict(),
        model_type=model_type,
        return_probabilities=return_probabilities,
        return_features=return_features
    )
    
    return result


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "temperature": 25.5,
        "motion": 1,
        "pulse": 72.0,
        "timestamp": datetime.now(),
        "sensor_id": "WSN001"
    }
    
    # Initialize predictor
    predictor = AnomalyPredictor()
    
    # Make prediction
    result = predictor.predict(
        data=sample_data,
        model_type="ensemble",
        return_probabilities=True,
        return_features=True
    )
    
    # Print result
    print("Prediction result:")
    for key, value in result.items():
        if key not in ["probabilities", "features"]:
            print(f"{key}: {value}")
    
    if "probabilities" in result:
        print("\nClass probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"{cls}: {prob:.4f}")
    
    if "features" in result:
        print("\nExtracted features (sample):")
        feature_items = list(result["features"].items())[:5]  # Show first 5 features
        for feature, value in feature_items:
            print(f"{feature}: {value:.4f}")
        print(f"... and {len(result['features']) - 5} more features")