import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
import pickle
from sklearn.preprocessing import StandardScaler
import os


class FeatureExtractor:
    """
    Feature extractor for WSN sensor data
    Extracts statistical and frequency domain features from time series data
    """
    
    def __init__(self, window_size=30, overlap=0.5, scaler_path=None):
        """
        Initialize the feature extractor
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.scaler = None
        self.scaler_path = scaler_path
        
        if scaler_path and os.path.exists(scaler_path):
            self.load_scaler(scaler_path)
    
    def load_scaler(self, scaler_path):
        """Load a pre-trained scaler from disk."""
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
    
    def save_scaler(self, scaler_path=None):
        """Save the trained scaler to disk."""
        if scaler_path is None: scaler_path = self.scaler_path
        if scaler_path and self.scaler is not None:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Saved scaler to {scaler_path}")
    
    def extract_statistical_features(self, window):
        """Extract statistical features from a window of data."""
        features = {}
        for column in ['temperature', 'motion', 'pulse']:
            if column in window.columns:
                values = window[column].values
                features[f"{column}_mean"] = np.mean(values)
                features[f"{column}_std"] = np.std(values)
                features[f"{column}_min"] = np.min(values)
                features[f"{column}_max"] = np.max(values)
                features[f"{column}_range"] = np.max(values) - np.min(values)
                features[f"{column}_skew"] = stats.skew(values)
                features[f"{column}_kurtosis"] = stats.kurtosis(values)
        return features
    
    def extract_frequency_features(self, window):
        """Extract frequency domain features from a window of data."""
        features = {}
        for column in ['temperature', 'pulse']:
            if column in window.columns:
                values = window[column].values
                fft_magnitude = np.abs(fft(values))[1:len(values)//2]
                if len(fft_magnitude) > 0:
                    features[f"{column}_fft_mean"] = np.mean(fft_magnitude)
                    features[f"{column}_fft_std"] = np.std(fft_magnitude)
                    features[f"{column}_fft_max"] = np.max(fft_magnitude)
        return features

    def extract_cross_sensor_features(self, window):
        """Extract features that capture relationships between sensors."""
        features = {}
        required = ['temperature', 'motion', 'pulse']
        if all(col in window.columns for col in required):
            corr, _ = stats.pearsonr(window['temperature'], window['pulse'])
            features["temp_pulse_corr"] = corr if not np.isnan(corr) else 0
            features["motion_activity_ratio"] = np.mean(window['motion'])
        return features

    def extract_features_from_window(self, window):
        """Extract all features from a window of data."""
        all_features = {}
        all_features.update(self.extract_statistical_features(window))
        all_features.update(self.extract_frequency_features(window))
        all_features.update(self.extract_cross_sensor_features(window))
        
        if 'timestamp' in window.columns:
            all_features['window_start_time'] = window['timestamp'].iloc[0]
        if 'label' in window.columns:
            all_features['window_label'] = window['label'].mode().iloc[0]
        return all_features
    
    def extract_features(self, df, fit_scaler=False):
        """Extract features from the entire dataset using sliding windows."""
        windows = [df.iloc[i:i + self.window_size] for i in range(0, len(df) - self.window_size + 1, self.step_size)]
        feature_dicts = [self.extract_features_from_window(w) for w in windows]
        features_df = pd.DataFrame(feature_dicts).fillna(0)
        
        metadata_cols = [col for col in ['window_start_time', 'window_label'] if col in features_df.columns]
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        
        if fit_scaler:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features_df[feature_cols])
            self.save_scaler()
        elif self.scaler:
            # **FIX**: Ensure columns are in the correct order before transforming
            feature_cols_ordered = self.scaler.get_feature_names_out()
            scaled_features = self.scaler.transform(features_df[feature_cols_ordered])
        else:
            scaled_features = features_df[feature_cols].values

        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
        for col in metadata_cols:
            scaled_df[col] = features_df[col].values
        
        return scaled_df

    def extract_features_from_single_window(self, window_data, scale=True):
        """
        Extract and scale features from a single window, ensuring correct feature order.
        """
        features = self.extract_features_from_window(window_data)
        
        metadata_keys = ['window_start_time', 'window_label']
        metadata = {k: features[k] for k in metadata_keys if k in features}
        feature_dict = {k: features[k] for k in features if k not in metadata_keys}

        if scale and self.scaler:
            # **FIX**: Use the scaler to define the feature order, creating a robust pipeline.
            feature_names = self.scaler.get_feature_names_out()
            
            # Create a DataFrame with the exact order and columns the scaler expects.
            feature_df = pd.DataFrame([feature_dict]).reindex(columns=feature_names, fill_value=0)
            
            # Transform the correctly ordered DataFrame
            scaled_values = self.scaler.transform(feature_df)
            
            # Create the final dictionary of scaled features
            scaled_features = dict(zip(feature_names, scaled_values[0]))
        else:
            scaled_features = feature_dict
        
        scaled_features.update(metadata)
        return scaled_features
