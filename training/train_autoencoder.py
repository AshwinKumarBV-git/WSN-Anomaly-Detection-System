import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.feature_extractor import FeatureExtractor


def load_and_preprocess_data(data_path, feature_extractor=None, test_size=0.2, random_state=42):
    """
    Load and preprocess the data for training
    
    Args:
        data_path: Path to the data file
        feature_extractor: FeatureExtractor instance (if None, a new one will be created)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, normal_indices, anomaly_indices
    """
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Create feature extractor if not provided
    if feature_extractor is None:
        scaler_path = "../backend/models/scaler.pkl"
        feature_extractor = FeatureExtractor(
            window_size=30,
            overlap=0.5,
            scaler_path=scaler_path
        )
    
    # Extract features
    print("Extracting features...")
    features_df = feature_extractor.extract_features(df, fit_scaler=True)
    
    # Save scaler
    feature_extractor.save_scaler()
    
    # Separate features and labels
    X = features_df.drop(['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity'], 
                        axis=1, errors='ignore')
    
    # Get label if available
    if 'window_label' in features_df.columns:
        labels = features_df['window_label']
        
        # Identify normal and anomaly indices
        normal_indices = labels == 'normal'
        anomaly_indices = ~normal_indices
        
        # For autoencoder training, we'll use only normal data
        X_normal = X[normal_indices]
        
        # Split normal data into train and validation
        X_train, X_val = train_test_split(
            X_normal, test_size=test_size, random_state=random_state
        )
        
        # Use all data for testing (to evaluate anomaly detection)
        X_test = X
        y_test = pd.Series(~normal_indices).astype(int)  # 0 for normal, 1 for anomaly
    else:
        # If no labels, assume all data is normal and split
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        X_val = X_test.copy()
        y_test = pd.Series(np.zeros(len(X_test)))  # Assume all normal
        normal_indices = pd.Series(np.ones(len(X), dtype=bool))
        anomaly_indices = pd.Series(np.zeros(len(X), dtype=bool))
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X_train, X_test, X_val, y_test, feature_names, normal_indices, anomaly_indices


def build_dense_autoencoder(input_dim, encoding_dim=32, activation='relu'):
    """
    Build a dense autoencoder model
    
    Args:
        input_dim: Dimension of input features
        encoding_dim: Dimension of the encoded representation
        activation: Activation function to use
        
    Returns:
        Autoencoder model
    """
    print(f"Building dense autoencoder with input_dim={input_dim}, encoding_dim={encoding_dim}")
    
    # Define encoder
    encoder = Sequential([
        Dense(input_dim // 2, activation=activation, input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(input_dim // 4, activation=activation),
        Dense(encoding_dim, activation=activation)
    ])
    
    # Define decoder
    decoder = Sequential([
        Dense(input_dim // 4, activation=activation, input_shape=(encoding_dim,)),
        Dropout(0.2),
        Dense(input_dim // 2, activation=activation),
        Dense(input_dim, activation='linear')
    ])
    
    # Define autoencoder
    autoencoder = Sequential([
        encoder,
        decoder
    ])
    
    return autoencoder


def build_lstm_autoencoder(input_dim, timesteps=1, encoding_dim=32):
    """
    Build an LSTM autoencoder model
    
    Args:
        input_dim: Dimension of input features
        timesteps: Number of time steps (1 for non-sequential data)
        encoding_dim: Dimension of the encoded representation
        
    Returns:
        LSTM Autoencoder model
    """
    print(f"Building LSTM autoencoder with input_dim={input_dim}, timesteps={timesteps}, encoding_dim={encoding_dim}")
    
    # Reshape input for LSTM
    inputs = Input(shape=(timesteps, input_dim))
    
    # Encoder
    encoded = LSTM(input_dim // 2, return_sequences=True)(inputs)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(encoding_dim, return_sequences=False)(encoded)
    
    # Decoder
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(encoding_dim, return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(input_dim // 2, return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)
    
    # Autoencoder model
    autoencoder = Model(inputs, decoded)
    
    return autoencoder


def train_autoencoder(model, X_train, X_val, epochs=100, batch_size=32, patience=10):
    """
    Train the autoencoder model
    
    Args:
        model: Autoencoder model
        X_train: Training data
        X_val: Validation data
        epochs: Maximum number of epochs
        batch_size: Batch size
        patience: Patience for early stopping
        
    Returns:
        Trained model and training history
    """
    print("Training autoencoder model...")
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    # Add model checkpoint if directory exists
    checkpoint_dir = "../backend/models/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "autoencoder_checkpoint.h5")
    callbacks.append(ModelCheckpoint(checkpoint_path, save_best_only=True))
    
    # Train model
    history = model.fit(
        X_train, X_train,  # Autoencoder tries to reconstruct the input
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_autoencoder(model, X_test, y_test, normal_indices, anomaly_indices):
    """
    Evaluate the autoencoder model for anomaly detection
    
    Args:
        model: Trained autoencoder model
        X_test: Test data
        y_test: Test labels (0 for normal, 1 for anomaly)
        normal_indices: Boolean mask for normal samples in the original dataset
        anomaly_indices: Boolean mask for anomaly samples in the original dataset
        
    Returns:
        Dictionary of evaluation metrics and threshold
    """
    print("Evaluating autoencoder model...")
    
    # Get reconstructions
    X_test_pred = model.predict(X_test)
    
    # Calculate reconstruction error (MSE)
    mse = np.mean(np.square(X_test - X_test_pred), axis=1)
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(12, 6))
    
    if np.any(anomaly_indices):
        # If we have labeled anomalies
        plt.hist(mse[~y_test.astype(bool)], bins=50, alpha=0.5, label='Normal')
        plt.hist(mse[y_test.astype(bool)], bins=50, alpha=0.5, label='Anomaly')
        plt.legend()
    else:
        # If no labeled anomalies
        plt.hist(mse, bins=50)
    
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.title('Reconstruction Error Distribution')
    
    # Save plot
    os.makedirs("../data/plots", exist_ok=True)
    plt.savefig("../data/plots/reconstruction_error_distribution.png")
    plt.close()
    
    # Determine threshold for anomaly detection
    # Use the 95th percentile of reconstruction errors on normal data
    if np.any(normal_indices):
        threshold = np.percentile(mse[~y_test.astype(bool)], 95)
    else:
        threshold = np.percentile(mse, 95)
    
    print(f"Anomaly threshold: {threshold:.6f}")
    
    # Classify as anomaly if reconstruction error > threshold
    predictions = (mse > threshold).astype(int)
    
    # Calculate metrics if we have labeled anomalies
    if np.any(anomaly_indices):
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save plot
        plt.savefig("../data/plots/autoencoder_confusion_matrix.png")
        plt.close()
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, mse)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.savefig("../data/plots/autoencoder_roc_curve.png")
        plt.close()
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "roc_auc": roc_auc,
            "threshold": float(threshold)
        }
    else:
        # If no labeled anomalies, just return basic stats
        anomaly_ratio = np.mean(predictions)
        metrics = {
            "anomaly_ratio": anomaly_ratio,
            "num_anomalies": int(np.sum(predictions)),
            "num_normal": int(np.sum(1 - predictions)),
            "threshold": float(threshold)
        }
    
    return metrics


def save_model_and_threshold(model, threshold, model_path, threshold_path):
    """
    Save the trained model and anomaly threshold
    
    Args:
        model: Trained autoencoder model
        threshold: Anomaly detection threshold
        model_path: Path to save the model
        threshold_path: Path to save the threshold
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save the threshold
    thresholds = {"autoencoder": threshold}
    with open(threshold_path, 'wb') as f:
        pickle.dump(thresholds, f)
    
    print(f"Threshold saved to {threshold_path}")


def plot_training_history(history):
    """
    Plot the training history
    
    Args:
        history: Training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("../data/plots", exist_ok=True)
    plt.savefig("../data/plots/autoencoder_training_history.png")
    plt.close()


def main():
    # Set paths
    data_path = "../data/simulated_sensor_data_with_anomalies.csv"
    model_path = "../backend/models/tf_autoencoder.h5"
    threshold_path = "../backend/models/anomaly_thresholds.pkl"
    metrics_path = "../data/autoencoder_metrics.json"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please run the simulation scripts first.")
        return
    
    # Load and preprocess data
    X_train, X_test, X_val, y_test, feature_names, normal_indices, anomaly_indices = \
        load_and_preprocess_data(data_path)
    
    # Print data info
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    if np.any(anomaly_indices):
        print("\nClass distribution in test data:")
        print(pd.Series(y_test).value_counts())
    
    # Build model
    input_dim = X_train.shape[1]
    encoding_dim = min(32, input_dim // 2)  # Adjust encoding dimension based on input size
    
    # Choose model type (dense or LSTM)
    model_type = "dense"  # or "lstm"
    
    if model_type == "dense":
        model = build_dense_autoencoder(input_dim, encoding_dim)
    else:
        # Reshape data for LSTM
        timesteps = 1
        X_train_reshaped = X_train.values.reshape(X_train.shape[0], timesteps, input_dim)
        X_val_reshaped = X_val.values.reshape(X_val.shape[0], timesteps, input_dim)
        X_test_reshaped = X_test.values.reshape(X_test.shape[0], timesteps, input_dim)
        
        model = build_lstm_autoencoder(input_dim, timesteps, encoding_dim)
        
        # Update variables for LSTM
        X_train, X_val, X_test = X_train_reshaped, X_val_reshaped, X_test_reshaped
    
    # Print model summary
    model.summary()
    
    # Train model
    model, history = train_autoencoder(model, X_train, X_val, epochs=100, batch_size=32, patience=10)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    metrics = evaluate_autoencoder(model, X_test, y_test, normal_indices, anomaly_indices)
    
    # Print evaluation results
    print("\nEvaluation results:")
    for key, value in metrics.items():
        if key not in ["classification_report", "confusion_matrix"]:
            print(f"{key}: {value}")
    
    # Save model and threshold
    save_model_and_threshold(model, metrics["threshold"], model_path, threshold_path)
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        import json
        
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if key in ["classification_report", "confusion_matrix"]:
                metrics_json[key] = value
            elif isinstance(value, (np.float32, np.float64)):
                metrics_json[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                metrics_json[key] = int(value)
            else:
                metrics_json[key] = value
        
        json.dump(metrics_json, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()