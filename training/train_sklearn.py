import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
        X_train, X_test, y_train, y_test, feature_names
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
    y = features_df['window_label'] if 'window_label' in features_df.columns else None
    
    # Convert labels to numeric
    if y is not None:
        # Create mapping of labels to integers
        label_mapping = {
            'normal': 0,
            'DoS': 1,
            'Jamming': 2,
            'Tampering': 3,
            'HardwareFault': 4,
            'EnvironmentalNoise': 5
        }
        
        # Map labels to integers
        y = y.map(label_mapping)
    
    # Split data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # If no labels, just split X
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        y_train, y_test = None, None
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names


def train_isolation_forest(X_train, contamination=0.1, random_state=42):
    """
    Train an Isolation Forest model for anomaly detection
    
    Args:
        X_train: Training data
        contamination: Expected proportion of anomalies
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    print("Training Isolation Forest model...")
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train)
    return model


def train_one_class_svm(X_train, nu=0.1):
    """
    Train a One-Class SVM model for anomaly detection
    
    Args:
        X_train: Training data
        nu: An upper bound on the fraction of training errors
        
    Returns:
        Trained model
    """
    print("Training One-Class SVM model...")
    model = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=nu
    )
    
    model.fit(X_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier for multi-class classification
    
    Args:
        X_train: Training data
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test=None, model_type="classifier", label_mapping=None):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels (optional for anomaly detection)
        model_type: Type of model ("classifier" or "anomaly_detector")
        label_mapping: Mapping from numeric labels to string labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating {model_type} model...")
    
    if model_type == "classifier" and y_test is not None:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_mapping.keys() if label_mapping else None,
                   yticklabels=label_mapping.keys() if label_mapping else None)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save plot
        os.makedirs("../data/plots", exist_ok=True)
        plt.savefig("../data/plots/confusion_matrix.png")
        plt.close()
        
        # Feature importance for Random Forest
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            feature_names = X_test.columns
            
            # Sort features by importance
            indices = np.argsort(feature_importances)[::-1]
            top_n = 20  # Show top 20 features
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances')
            plt.bar(range(top_n), feature_importances[indices[:top_n]], align='center')
            plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=90)
            plt.tight_layout()
            
            # Save plot
            plt.savefig("../data/plots/feature_importances.png")
            plt.close()
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }
    
    elif model_type == "anomaly_detector":
        # For anomaly detectors, we predict and then compare with y_test if available
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            
            # Convert predictions to binary (1 for normal, -1 for anomaly)
            # Isolation Forest and One-Class SVM use -1 for anomalies, 1 for normal
            y_pred_binary = np.where(y_pred == 1, 0, 1)  # Convert to 0 for normal, 1 for anomaly
            
            if y_test is not None:
                # Convert y_test to binary (0 for normal, 1 for anomaly)
                y_test_binary = np.where(y_test == 0, 0, 1)  # 0 is normal in our mapping
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_binary, y_pred_binary)
                report = classification_report(y_test_binary, y_pred_binary, output_dict=True)
                conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Normal', 'Anomaly'],
                           yticklabels=['Normal', 'Anomaly'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix (Binary)')
                
                # Save plot
                os.makedirs("../data/plots", exist_ok=True)
                plt.savefig("../data/plots/anomaly_confusion_matrix.png")
                plt.close()
                
                return {
                    "accuracy": accuracy,
                    "classification_report": report,
                    "confusion_matrix": conf_matrix.tolist()
                }
            else:
                # If no labels, just return the predictions
                anomaly_ratio = np.mean(y_pred_binary)
                return {
                    "anomaly_ratio": anomaly_ratio,
                    "num_anomalies": int(np.sum(y_pred_binary)),
                    "num_normal": int(np.sum(1 - y_pred_binary))
                }
        else:
            return {"error": "Model does not have predict method"}
    
    else:
        return {"error": "Invalid model type or missing test labels"}


def save_model(model, model_path):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")


def main():
    # Set paths
    data_path = "../data/simulated_sensor_data_with_anomalies.csv"
    model_path = "../backend/models/sklearn_model.pkl"
    metrics_path = "../data/model_metrics.json"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please run the simulation scripts first.")
        return
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_path)
    print("test data: ", X_test.head())
    print("test labels: ", y_test.head())
    print("feature names: ", feature_names)
    print("test data shape: ", X_test.shape) 
    # Print data info
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    if y_train is not None:
        print("\nClass distribution in training data:")
        print(pd.Series(y_train).value_counts())
    
    # Define label mapping
    label_mapping = {
        0: "normal",
        1: "DoS",
        2: "Jamming",
        3: "Tampering",
        4: "HardwareFault",
        5: "EnvironmentalNoise"
    }
    
    # Train model based on available labels
    if y_train is not None and len(np.unique(y_train)) > 2:
        # Multi-class classification
        model = train_random_forest(X_train, y_train)
        model_type = "classifier"
    else:
        # Anomaly detection (binary or unsupervised)
        # Use Isolation Forest for better performance
        model = train_isolation_forest(X_train)
        model_type = "anomaly_detector"
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, model_type, label_mapping)
    
    # Print evaluation results
    print("\nEvaluation results:")
    if "accuracy" in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    if "classification_report" in metrics:
        print("\nClassification Report:")
        for cls, values in metrics["classification_report"].items():
            if isinstance(values, dict):
                print(f"{cls}:")
                for metric, value in values.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"{cls}: {values:.4f}")
    
    # Save model
    save_model(model, model_path)
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()