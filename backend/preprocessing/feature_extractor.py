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
        
        Args:
            window_size: Size of the sliding window (in data points)
            overlap: Overlap between consecutive windows (0 to 1)
            scaler_path: Path to saved StandardScaler (if None, a new one will be created)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.scaler = None
        self.scaler_path = scaler_path
        
        # Load scaler if path is provided and file exists
        if scaler_path and os.path.exists(scaler_path):
            self.load_scaler(scaler_path)
    
    def load_scaler(self, scaler_path):
        """
        Load a pre-trained scaler from disk
        
        Args:
            scaler_path: Path to the saved scaler
        """
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
    
    def save_scaler(self, scaler_path=None):
        """
        Save the trained scaler to disk
        
        Args:
            scaler_path: Path to save the scaler (uses self.scaler_path if None)
        """
        if scaler_path is None:
            scaler_path = self.scaler_path
        
        if scaler_path and self.scaler is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Saved scaler to {scaler_path}")
    
    def extract_statistical_features(self, window):
        """
        Extract statistical features from a window of data
        
        Args:
            window: DataFrame containing a window of sensor data
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Process each sensor column (excluding timestamp and labels)
        for column in ['temperature', 'motion', 'pulse']:
            if column in window.columns:
                values = window[column].values
                
                # Basic statistics
                features[f"{column}_mean"] = np.mean(values)
                features[f"{column}_std"] = np.std(values)
                features[f"{column}_min"] = np.min(values)
                features[f"{column}_max"] = np.max(values)
                features[f"{column}_range"] = np.max(values) - np.min(values)
                
                # Percentiles
                features[f"{column}_25p"] = np.percentile(values, 25)
                features[f"{column}_50p"] = np.percentile(values, 50)  # median
                features[f"{column}_75p"] = np.percentile(values, 75)
                
                # Higher order statistics
                features[f"{column}_skew"] = stats.skew(values)
                features[f"{column}_kurtosis"] = stats.kurtosis(values)
                
                # Rate of change statistics
                if len(values) > 1:
                    diff = np.diff(values)
                    features[f"{column}_diff_mean"] = np.mean(diff)
                    features[f"{column}_diff_std"] = np.std(diff)
                    features[f"{column}_diff_max"] = np.max(np.abs(diff))
                else:
                    features[f"{column}_diff_mean"] = 0
                    features[f"{column}_diff_std"] = 0
                    features[f"{column}_diff_max"] = 0
        
        return features
    
    def extract_frequency_features(self, window):
        """
        Extract frequency domain features from a window of data
        
        Args:
            window: DataFrame containing a window of sensor data
            
        Returns:
            Dictionary of frequency domain features
        """
        features = {}
        
        # Process each sensor column (excluding timestamp, motion, and labels)
        # Motion is binary so FFT doesn't make sense for it
        for column in ['temperature', 'pulse']:
            if column in window.columns:
                values = window[column].values
                
                # Apply FFT
                fft_values = fft(values)
                fft_magnitude = np.abs(fft_values)
                
                # Use only the first half (positive frequencies)
                n = len(fft_magnitude)
                half_n = n // 2
                pos_freq_magnitudes = fft_magnitude[1:half_n]  # Skip DC component
                
                if len(pos_freq_magnitudes) > 0:
                    # Frequency domain features
                    features[f"{column}_fft_mean"] = np.mean(pos_freq_magnitudes)
                    features[f"{column}_fft_std"] = np.std(pos_freq_magnitudes)
                    features[f"{column}_fft_max"] = np.max(pos_freq_magnitudes)
                    
                    # Dominant frequency
                    if len(pos_freq_magnitudes) > 0:
                        dom_freq_idx = np.argmax(pos_freq_magnitudes) + 1  # +1 because we skipped DC
                        features[f"{column}_dominant_freq"] = dom_freq_idx / n
                    else:
                        features[f"{column}_dominant_freq"] = 0
                    
                    # Energy in different frequency bands
                    if len(pos_freq_magnitudes) >= 3:
                        band_size = len(pos_freq_magnitudes) // 3
                        features[f"{column}_low_freq_energy"] = np.sum(pos_freq_magnitudes[:band_size])
                        features[f"{column}_mid_freq_energy"] = np.sum(pos_freq_magnitudes[band_size:2*band_size])
                        features[f"{column}_high_freq_energy"] = np.sum(pos_freq_magnitudes[2*band_size:])
                    else:
                        features[f"{column}_low_freq_energy"] = np.sum(pos_freq_magnitudes)
                        features[f"{column}_mid_freq_energy"] = 0
                        features[f"{column}_high_freq_energy"] = 0
                else:
                    # Default values if window is too small
                    features[f"{column}_fft_mean"] = 0
                    features[f"{column}_fft_std"] = 0
                    features[f"{column}_fft_max"] = 0
                    features[f"{column}_dominant_freq"] = 0
                    features[f"{column}_low_freq_energy"] = 0
                    features[f"{column}_mid_freq_energy"] = 0
                    features[f"{column}_high_freq_energy"] = 0
        
        return features
    
    def extract_cross_sensor_features(self, window):
        """
        Extract features that capture relationships between different sensors
        
        Args:
            window: DataFrame containing a window of sensor data
            
        Returns:
            Dictionary of cross-sensor features
        """
        features = {}
        
        # Check if we have all required columns
        required_columns = ['temperature', 'motion', 'pulse']
        if all(col in window.columns for col in required_columns):
            # Correlations between sensors
            temp_values = window['temperature'].values
            pulse_values = window['pulse'].values
            
            # Correlation between temperature and pulse
            if len(temp_values) > 1 and len(pulse_values) > 1:
                corr, _ = stats.pearsonr(temp_values, pulse_values)
                features["temp_pulse_corr"] = corr if not np.isnan(corr) else 0
            else:
                features["temp_pulse_corr"] = 0
            
            # Motion activity ratio
            motion_values = window['motion'].values
            features["motion_activity_ratio"] = np.mean(motion_values)
            
            # Conditional statistics
            if np.any(motion_values > 0):
                # Statistics when motion is detected
                motion_mask = motion_values > 0
                features["temp_when_motion_mean"] = np.mean(temp_values[motion_mask]) if np.any(motion_mask) else np.mean(temp_values)
                features["pulse_when_motion_mean"] = np.mean(pulse_values[motion_mask]) if np.any(motion_mask) else np.mean(pulse_values)
            else:
                features["temp_when_motion_mean"] = np.mean(temp_values)
                features["pulse_when_motion_mean"] = np.mean(pulse_values)
        
        return features
    
    def extract_features_from_window(self, window):
        """
        Extract all features from a window of data
        
        Args:
            window: DataFrame containing a window of sensor data
            
        Returns:
            Dictionary of all features
        """
        # Extract all feature types
        statistical_features = self.extract_statistical_features(window)
        frequency_features = self.extract_frequency_features(window)
        cross_sensor_features = self.extract_cross_sensor_features(window)
        
        # Combine all features
        all_features = {}
        all_features.update(statistical_features)
        all_features.update(frequency_features)
        all_features.update(cross_sensor_features)
        
        # Add metadata
        if 'timestamp' in window.columns:
            all_features['window_start_time'] = window['timestamp'].iloc[0]
            all_features['window_end_time'] = window['timestamp'].iloc[-1]
        
        if 'sensor_id' in window.columns:
            all_features['sensor_id'] = window['sensor_id'].iloc[0]
        
        # Determine window label (majority vote)
        if 'label' in window.columns:
            label_counts = window['label'].value_counts()
            all_features['window_label'] = label_counts.index[0]  # Most common label
            
            # Calculate label purity (percentage of majority label)
            all_features['label_purity'] = label_counts.iloc[0] / len(window)
        
        return all_features
    
    def extract_features(self, df, fit_scaler=False):
        """
        Extract features from the entire dataset using sliding windows
        
        Args:
            df: DataFrame containing sensor data
            fit_scaler: Whether to fit a new scaler on the extracted features
            
        Returns:
            DataFrame of extracted features
        """
        # Make sure timestamp column is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create sliding windows
        windows = []
        for i in range(0, len(df) - self.window_size + 1, self.step_size):
            window = df.iloc[i:i + self.window_size]
            windows.append(window)
        
        # Extract features from each window
        feature_dicts = []
        for window in windows:
            features = self.extract_features_from_window(window)
            feature_dicts.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_dicts)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Separate metadata columns
        metadata_cols = ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        
        # Scale the features
        if fit_scaler:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features_df[feature_cols])
            
            # Save the scaler if path is provided
            if self.scaler_path:
                self.save_scaler()
        elif self.scaler is not None:
            scaled_features = self.scaler.transform(features_df[feature_cols])
        else:
            # If no scaler is available and not fitting, just standardize manually
            scaled_features = (features_df[feature_cols] - features_df[feature_cols].mean()) / features_df[feature_cols].std()
        
        # Create DataFrame with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
        
        # Add back metadata columns
        for col in metadata_cols:
            if col in features_df.columns:
                scaled_df[col] = features_df[col]
        
        return scaled_df
    
    def extract_features_from_single_window(self, window_data, scale=True):
        """
        Extract features from a single window of data (for real-time prediction)
        
        Args:
            window_data: DataFrame containing a window of sensor data
            scale: Whether to scale the features using the pre-trained scaler
            
        Returns:
            Dictionary of extracted features (scaled if requested)
        """
        # Extract features
        features = self.extract_features_from_window(window_data)
        
        # Separate metadata
        metadata_keys = ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']
        metadata = {k: features[k] for k in metadata_keys if k in features}
        
        # Get feature values (excluding metadata)
        feature_keys = [k for k in features.keys() if k not in metadata_keys]
        feature_values = [features[k] for k in feature_keys]
        
        # Scale features if requested and scaler is available
        if scale and self.scaler is not None:
            scaled_values = self.scaler.transform([feature_values])[0]
            scaled_features = dict(zip(feature_keys, scaled_values))
        else:
            scaled_features = {k: features[k] for k in feature_keys}
        
        # Add back metadata
        scaled_features.update(metadata)
        
        return scaled_features


# Example usage
if __name__ == "__main__":
    # Load sample data
    data_path = "../../data/simulated_sensor_data_with_anomalies.csv"
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Create feature extractor
        scaler_path = "../models/scaler.pkl"
        extractor = FeatureExtractor(window_size=30, overlap=0.5, scaler_path=scaler_path)
        
        # Extract features
        print("Extracting features...")
        features_df = extractor.extract_features(df, fit_scaler=True)
        
        # Save features
        features_path = "../../data/extracted_features.csv"
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        features_df.to_csv(features_path, index=False)
        
        print(f"Extracted {len(features_df)} feature windows from {len(df)} data points")
        print(f"Features saved to {features_path}")
        
        # Display sample of the features
        print("\nSample of extracted features:")
        print(features_df.head())
        
        # Display feature statistics
        print("\nFeature statistics:")
        feature_cols = [col for col in features_df.columns 
                       if col not in ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']]
        print(features_df[feature_cols].describe().T)
    else:
        print(f"Data file {data_path} not found. Please run the simulation scripts first.")