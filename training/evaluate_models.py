import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.feature_extractor import FeatureExtractor
from backend.services.predictor import AnomalyPredictor


def load_data_and_models():
    """
    Load the test data, trained models, and thresholds.
    """
    # Set paths
    data_path = "../data/simulated_sensor_data_with_anomalies.csv"
    sklearn_model_path = "../backend/models/sklearn_model.pkl"
    autoencoder_model_path = "../backend/models/tf_autoencoder.h5"
    threshold_path = "../backend/models/anomaly_thresholds.pkl"
    scaler_path = "../backend/models/scaler.pkl"
    
    # Check if essential files exist
    missing_files = []
    for path in [data_path, sklearn_model_path, autoencoder_model_path, scaler_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the simulation and training scripts first (simulate_data.py, inject_anomalies.py, train_sklearn.py, train_autoencoder.py).")
        return None
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Load feature extractor with scaler
    feature_extractor = FeatureExtractor(window_size=30, overlap=0.5, scaler_path=scaler_path)
    
    # Extract features
    print("Extracting features...")
    features_df = feature_extractor.extract_features(df, fit_scaler=False)
    
    # Separate features and labels
    feature_cols = [col for col in features_df.columns if col not in ['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity']]
    X = features_df[feature_cols]
    
    if 'window_label' in features_df.columns:
        labels = features_df['window_label']
        y = (labels != 'normal').astype(int)
        anomaly_types = labels.copy()
    else:
        y = pd.Series(np.zeros(len(X)), index=X.index)
        anomaly_types = pd.Series(['normal'] * len(X), index=X.index)
    
    # Load sklearn model
    print(f"Loading sklearn model from {sklearn_model_path}")
    with open(sklearn_model_path, 'rb') as f:
        sklearn_model_info = pickle.load(f)
        # **FIX**: Ensure we handle both dictionary and direct model pickles for backward compatibility
        if isinstance(sklearn_model_info, dict):
            sklearn_model = sklearn_model_info['model']
            sklearn_model_type = sklearn_model_info.get('model_type', 'unknown')
        else:
            # Handle the old format where the model was saved directly
            sklearn_model = sklearn_model_info
            sklearn_model_type = 'random_forest' if hasattr(sklearn_model, 'predict_proba') else 'isolation_forest'

    # Load autoencoder model
    print(f"Loading autoencoder model from {autoencoder_model_path}")
    autoencoder_model = tf.keras.models.load_model(autoencoder_model_path)
    
    # Load thresholds if they exist
    thresholds = {}
    if os.path.exists(threshold_path):
        print(f"Loading thresholds from {threshold_path}")
        with open(threshold_path, 'rb') as f:
            thresholds = pickle.load(f)
    else:
        print(f"Threshold file not found at {threshold_path}. Using default.")
        thresholds['autoencoder'] = 0.1 # Default value

    # Create AnomalyPredictor instance to use its logic
    predictor = AnomalyPredictor()
    
    return {
        'features': X,
        'labels': y,
        'anomaly_types': anomaly_types,
        'sklearn_model': sklearn_model,
        'sklearn_model_type': sklearn_model_type,
        'autoencoder_model': autoencoder_model,
        'thresholds': thresholds,
        'predictor': predictor,
    }

def evaluate_sklearn_model(model, X, y, model_type):
    """Evaluate the sklearn model."""
    print(f"Evaluating sklearn model ({model_type})...")
    
    if model_type in ['isolation_forest', 'one_class_svm']:
        raw_preds = model.predict(X)
        predictions = (raw_preds == -1).astype(int)
        scores = -model.decision_function(X) if hasattr(model, 'decision_function') else -model.score_samples(X)
    elif model_type == 'random_forest':
        predictions = model.predict(X)
        scores = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        print(f"Unknown model type for evaluation: {model_type}")
        return None

    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y, predictions)
    
    fpr, tpr, roc_auc = (None, None, None)
    precision_curve, recall_curve, pr_auc = (None, None, None)

    if scores is not None:
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(y, scores)
        pr_auc = auc(recall_curve, precision_curve)

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "classification_report": report, "confusion_matrix": conf_matrix,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "precision_curve": precision_curve, "recall_curve": recall_curve, "pr_auc": pr_auc
    }

def evaluate_autoencoder(model, X, y, threshold):
    """Evaluate the autoencoder model."""
    print("Evaluating autoencoder model...")
    X_pred = model.predict(X)
    mse = np.mean(np.square(X.values - X_pred), axis=1)
    predictions = (mse > threshold).astype(int)
    
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y, predictions)
    
    fpr, tpr, _ = roc_curve(y, mse)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y, mse)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "classification_report": report, "confusion_matrix": conf_matrix,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "precision_curve": precision_curve, "recall_curve": recall_curve, "pr_auc": pr_auc
    }

def evaluate_ensemble(predictor, X, y):
    """Evaluate the ensemble model using the AnomalyPredictor."""
    print("Evaluating ensemble model...")
    
    results = []
    for _, row in X.iterrows():
        features_dict = row.to_dict()
        # **FIX**: Use a method that predicts from features, which we will add to the predictor
        prediction = predictor.predict_from_features(features_dict)
        results.append(prediction)
    
    predictions = np.array([p['prediction'] != 'normal' for p in results]).astype(int)
    scores = np.array([p['confidence'] for p in results])
    
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y, predictions)
    
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y, scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "classification_report": report, "confusion_matrix": conf_matrix,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "precision_curve": precision_curve, "recall_curve": recall_curve, "pr_auc": pr_auc
    }

def evaluate_by_anomaly_type(predictor, X, anomaly_types):
    """Evaluate performance by anomaly type."""
    print("Evaluating performance by anomaly type...")
    
    results = []
    for _, row in X.iterrows():
        features_dict = row.to_dict()
        prediction = predictor.predict_from_features(features_dict)
        results.append(prediction)
        
    pred_df = pd.DataFrame(results)
    pred_df['true_label'] = anomaly_types.values
    
    metrics_by_type = {}
    for anomaly_type in pred_df['true_label'].unique():
        type_df = pred_df[pred_df['true_label'] == anomaly_type]
        
        if anomaly_type == 'normal':
            # Correct if prediction is 'normal'
            detection_rate = np.mean(type_df['prediction'] == 'normal')
        else:
            # Correct if prediction matches the anomaly type
            detection_rate = np.mean(type_df['prediction'] == anomaly_type)
            
        metrics_by_type[anomaly_type] = {
            "count": len(type_df),
            "detection_rate": float(detection_rate)
        }
        
    return metrics_by_type

def plot_roc_curves(metrics_dict):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    for name, metrics in metrics_dict.items():
        if metrics and metrics.get('fpr') is not None and metrics.get('tpr') is not None:
            plt.plot(metrics['fpr'], metrics['tpr'], label=f"{name.capitalize()} (AUC = {metrics.get('roc_auc', 0):.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves'); plt.legend(loc="lower right"); plt.grid(True)
    
    os.makedirs("../data/plots", exist_ok=True)
    plt.savefig("../data/plots/roc_curves_comparison.png")
    plt.close()

def plot_pr_curves(metrics_dict):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 8))
    for name, metrics in metrics_dict.items():
        if metrics and metrics.get('precision_curve') is not None and metrics.get('recall_curve') is not None:
            plt.plot(metrics['recall_curve'], metrics['precision_curve'], label=f"{name.capitalize()} (AUC = {metrics.get('pr_auc', 0):.3f})")

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curves'); plt.legend(loc="lower left"); plt.grid(True)
    
    os.makedirs("../data/plots", exist_ok=True)
    plt.savefig("../data/plots/pr_curves_comparison.png")
    plt.close()

def plot_confusion_matrices(metrics_dict):
    """Plot confusion matrices for all models."""
    num_models = len([m for m in metrics_dict.values() if m])
    if num_models == 0: return
    
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1: axes = [axes] # Make it iterable
    
    model_names = list(metrics_dict.keys())
    for i, ax in enumerate(axes):
        model_name = model_names[i]
        metrics = metrics_dict[model_name]
        if metrics and 'confusion_matrix' in metrics:
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(f'{model_name.capitalize()} Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig("../data/plots/confusion_matrices_comparison.png")
    plt.close()

def save_evaluation_results(metrics_dict, metrics_by_type):
    """Save evaluation results to a JSON file."""
    results = {}
    serializable_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    for model_name, metrics in metrics_dict.items():
        if not metrics: continue
        results[model_name] = {key: float(metrics[key]) for key in serializable_metrics if key in metrics and metrics[key] is not None}
        if 'classification_report' in metrics:
            results[model_name]['classification_report'] = metrics['classification_report']
        if 'confusion_matrix' in metrics:
            results[model_name]['confusion_matrix'] = metrics['confusion_matrix'].tolist()

    results['by_anomaly_type'] = metrics_by_type
    
    os.makedirs("../data", exist_ok=True)
    with open("../data/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Evaluation results saved to ../data/evaluation_results.json")

def main():
    """Main function to run the evaluation pipeline."""
    loaded_assets = load_data_and_models()
    if loaded_assets is None:
        return
    
    # Evaluate individual models
    sklearn_metrics = evaluate_sklearn_model(
        loaded_assets['sklearn_model'], loaded_assets['features'], loaded_assets['labels'], loaded_assets['sklearn_model_type']
    )
    autoencoder_metrics = evaluate_autoencoder(
        loaded_assets['autoencoder_model'], loaded_assets['features'], loaded_assets['labels'], loaded_assets['thresholds'].get('autoencoder', 0.1)
    )
    ensemble_metrics = evaluate_ensemble(
        loaded_assets['predictor'], loaded_assets['features'], loaded_assets['labels']
    )
    
    metrics_by_type = evaluate_by_anomaly_type(
        loaded_assets['predictor'], loaded_assets['features'], loaded_assets['anomaly_types']
    )
    
    all_metrics = {
        "sklearn": sklearn_metrics,
        "autoencoder": autoencoder_metrics,
        "ensemble": ensemble_metrics
    }
    
    # Generate plots
    plot_roc_curves(all_metrics)
    plot_pr_curves(all_metrics)
    plot_confusion_matrices(all_metrics)
    
    # Save results
    save_evaluation_results(all_metrics, metrics_by_type)
    
    print("\n--- Evaluation Summary ---")
    for name, metrics in all_metrics.items():
        if metrics:
            print(f"{name.capitalize()} Model: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC AUC={metrics.get('roc_auc', 0):.4f}")
    
    print("\n--- Performance by Anomaly Type (Ensemble) ---")
    for anomaly_type, metrics in metrics_by_type.items():
        print(f"{anomaly_type}: {metrics['count']} samples, Detection Rate={metrics['detection_rate']:.4f}")
    
    print("\nAll plots and results saved to the '../data' directory.")

if __name__ == "__main__":
    main()
