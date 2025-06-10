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
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.feature_extractor import FeatureExtractor
from backend.services.predictor import AnomalyPredictor


def load_data_and_models():
    """
    Load the test data, trained models, and thresholds
    
    Returns:
        Dictionary containing data, models, and thresholds
    """
    # Set paths
    data_path = "../data/simulated_sensor_data_with_anomalies.csv"
    sklearn_model_path = "../backend/models/sklearn_model.pkl"
    autoencoder_model_path = "../backend/models/tf_autoencoder.h5"
    threshold_path = "../backend/models/anomaly_thresholds.pkl"
    scaler_path = "../backend/models/scaler.pkl"
    
    # Check if files exist
    missing_files = []
    for path in [data_path, sklearn_model_path, autoencoder_model_path, threshold_path, scaler_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the simulation and training scripts first.")
        return None
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Load feature extractor with scaler
    feature_extractor = FeatureExtractor(
        window_size=30,
        overlap=0.5,
        scaler_path=scaler_path
    )
    
    # Extract features
    print("Extracting features...")
    features_df = feature_extractor.extract_features(df, fit_scaler=False)
    
    # Separate features and labels
    X = features_df.drop(['window_start_time', 'window_end_time', 'sensor_id', 'window_label', 'label_purity'], 
                        axis=1, errors='ignore')
    
    # Get label if available
    if 'window_label' in features_df.columns:
        labels = features_df['window_label']
        
        # Convert labels to binary (0 for normal, 1 for anomaly)
        y = (labels != 'normal').astype(int)
        
        # Get anomaly types
        anomaly_types = labels.copy()
        anomaly_types[anomaly_types == 'normal'] = 'normal'
    else:
        # If no labels, assume all data is normal
        y = pd.Series(np.zeros(len(X)))
        anomaly_types = pd.Series(['normal'] * len(X))
    
    # Load sklearn model
    print(f"Loading sklearn model from {sklearn_model_path}")
    with open(sklearn_model_path, 'rb') as f:
        sklearn_model_info = pickle.load(f)
        
    sklearn_model = sklearn_model_info['model']
    sklearn_model_type = sklearn_model_info.get('model_type', 'unknown')
    
    # Load autoencoder model
    print(f"Loading autoencoder model from {autoencoder_model_path}")
    autoencoder_model = tf.keras.models.load_model(autoencoder_model_path)
    
    # Load thresholds
    print(f"Loading thresholds from {threshold_path}")
    with open(threshold_path, 'rb') as f:
        thresholds = pickle.load(f)
    
    # Create AnomalyPredictor instance
    predictor = AnomalyPredictor(
        sklearn_model_path=sklearn_model_path,
        autoencoder_model_path=autoencoder_model_path,
        threshold_path=threshold_path,
        scaler_path=scaler_path
    )
    
    return {
        'data': df,
        'features': X,
        'labels': y,
        'anomaly_types': anomaly_types,
        'sklearn_model': sklearn_model,
        'sklearn_model_type': sklearn_model_type,
        'autoencoder_model': autoencoder_model,
        'thresholds': thresholds,
        'predictor': predictor,
        'feature_extractor': feature_extractor
    }


def evaluate_sklearn_model(model, X, y, model_type):
    """
    Evaluate the sklearn model
    
    Args:
        model: Trained sklearn model
        X: Features
        y: Labels (0 for normal, 1 for anomaly)
        model_type: Type of sklearn model
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating sklearn model ({model_type})...")
    
    # Get predictions
    if model_type in ['isolation_forest', 'one_class_svm']:
        # For anomaly detection models, -1 is anomaly, 1 is normal
        # Convert to 0 for normal, 1 for anomaly
        raw_preds = model.predict(X)
        predictions = (raw_preds == -1).astype(int)
        
        # Get decision scores
        if hasattr(model, 'decision_function'):
            scores = -model.decision_function(X)  # Negate so higher = more anomalous
        elif hasattr(model, 'score_samples'):
            scores = -model.score_samples(X)  # Negate so higher = more anomalous
        else:
            scores = None
    elif model_type == 'random_forest':
        # For classification models
        predictions = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            # Get probability of anomaly class
            probs = model.predict_proba(X)
            if probs.shape[1] > 1:  # Binary classification
                scores = probs[:, 1]  # Probability of anomaly class
            else:  # One-class classification
                scores = probs[:, 0]
        else:
            scores = None
    else:
        # Unknown model type
        print(f"Unknown model type: {model_type}")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y, predictions)
    
    # Calculate ROC and AUC if scores are available
    if scores is not None:
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y, scores)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        fpr, tpr, roc_auc = None, None, None
        precision_curve, recall_curve, pr_auc = None, None, None
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "pr_auc": pr_auc
    }


def evaluate_autoencoder(model, X, y, threshold):
    """
    Evaluate the autoencoder model
    
    Args:
        model: Trained autoencoder model
        X: Features
        y: Labels (0 for normal, 1 for anomaly)
        threshold: Anomaly detection threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating autoencoder model...")
    
    # Get reconstructions
    X_pred = model.predict(X)
    
    # Calculate reconstruction error (MSE)
    mse = np.mean(np.square(X - X_pred), axis=1)
    
    # Classify as anomaly if reconstruction error > threshold
    predictions = (mse > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y, predictions)
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y, mse)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y, mse)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "pr_auc": pr_auc,
        "reconstruction_errors": mse,
        "threshold": threshold
    }


def evaluate_ensemble(predictor, X, y):
    """
    Evaluate the ensemble model using the AnomalyPredictor
    
    Args:
        predictor: AnomalyPredictor instance
        X: Features
        y: Labels (0 for normal, 1 for anomaly)
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating ensemble model...")
    
    # Get predictions
    results = []
    for _, row in X.iterrows():
        # Convert row to dict for predictor
        features_dict = row.to_dict()
        
        # Get prediction
        prediction = predictor.predict_with_features(features_dict)
        results.append(prediction)
    
    # Extract predictions and scores
    predictions = np.array([r['is_anomaly'] for r in results]).astype(int)
    scores = np.array([r['anomaly_score'] for r in results])
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y, predictions)
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y, scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "pr_auc": pr_auc,
        "anomaly_scores": scores
    }


def evaluate_by_anomaly_type(predictor, X, y, anomaly_types):
    """
    Evaluate the ensemble model performance by anomaly type
    
    Args:
        predictor: AnomalyPredictor instance
        X: Features
        y: Labels (0 for normal, 1 for anomaly)
        anomaly_types: Series with anomaly type labels
        
    Returns:
        Dictionary of evaluation metrics by anomaly type
    """
    print("Evaluating performance by anomaly type...")
    
    # Get predictions
    results = []
    for _, row in X.iterrows():
        # Convert row to dict for predictor
        features_dict = row.to_dict()
        
        # Get prediction
        prediction = predictor.predict_with_features(features_dict)
        results.append(prediction)
    
    # Extract predictions
    predictions = np.array([r['is_anomaly'] for r in results]).astype(int)
    
    # Get unique anomaly types
    unique_types = anomaly_types.unique()
    
    # Calculate metrics for each anomaly type
    metrics_by_type = {}
    
    for anomaly_type in unique_types:
        # Get indices for this anomaly type
        type_indices = (anomaly_types == anomaly_type)
        
        # Skip if no samples of this type
        if not np.any(type_indices):
            continue
        
        # Get predictions and true labels for this type
        type_preds = predictions[type_indices]
        type_y = y[type_indices]
        
        # For normal type, true label should be 0, for anomaly types, true label should be 1
        expected_label = 0 if anomaly_type == 'normal' else 1
        
        # Calculate accuracy for this type
        if expected_label == 0:
            # For normal type, correct prediction is 0
            accuracy = np.mean(type_preds == 0)
        else:
            # For anomaly types, correct prediction is 1
            accuracy = np.mean(type_preds == 1)
        
        # Store metrics
        metrics_by_type[anomaly_type] = {
            "count": int(np.sum(type_indices)),
            "accuracy": float(accuracy),
            "detection_rate": float(np.mean(type_preds == expected_label))
        }
    
    return metrics_by_type


def plot_roc_curves(sklearn_metrics, autoencoder_metrics, ensemble_metrics):
    """
    Plot ROC curves for all models
    
    Args:
        sklearn_metrics: Metrics for sklearn model
        autoencoder_metrics: Metrics for autoencoder model
        ensemble_metrics: Metrics for ensemble model
    """
    plt.figure(figsize=(10, 8))
    
    # Plot sklearn ROC curve
    if sklearn_metrics['fpr'] is not None and sklearn_metrics['tpr'] is not None:
        plt.plot(sklearn_metrics['fpr'], sklearn_metrics['tpr'], 
                 label=f'Sklearn (AUC = {sklearn_metrics["roc_auc"]:.3f})')
    
    # Plot autoencoder ROC curve
    if autoencoder_metrics['fpr'] is not None and autoencoder_metrics['tpr'] is not None:
        plt.plot(autoencoder_metrics['fpr'], autoencoder_metrics['tpr'], 
                 label=f'Autoencoder (AUC = {autoencoder_metrics["roc_auc"]:.3f})')
    
    # Plot ensemble ROC curve
    if ensemble_metrics['fpr'] is not None and ensemble_metrics['tpr'] is not None:
        plt.plot(ensemble_metrics['fpr'], ensemble_metrics['tpr'], 
                 label=f'Ensemble (AUC = {ensemble_metrics["roc_auc"]:.3f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs("../data/plots", exist_ok=True)
    plt.savefig("../data/plots/roc_curves_comparison.png")
    plt.close()


def plot_pr_curves(sklearn_metrics, autoencoder_metrics, ensemble_metrics):
    """
    Plot Precision-Recall curves for all models
    
    Args:
        sklearn_metrics: Metrics for sklearn model
        autoencoder_metrics: Metrics for autoencoder model
        ensemble_metrics: Metrics for ensemble model
    """
    plt.figure(figsize=(10, 8))
    
    # Plot sklearn PR curve
    if (sklearn_metrics['precision_curve'] is not None and 
        sklearn_metrics['recall_curve'] is not None):
        plt.plot(sklearn_metrics['recall_curve'], sklearn_metrics['precision_curve'], 
                 label=f'Sklearn (AUC = {sklearn_metrics["pr_auc"]:.3f})')
    
    # Plot autoencoder PR curve
    if (autoencoder_metrics['precision_curve'] is not None and 
        autoencoder_metrics['recall_curve'] is not None):
        plt.plot(autoencoder_metrics['recall_curve'], autoencoder_metrics['precision_curve'], 
                 label=f'Autoencoder (AUC = {autoencoder_metrics["pr_auc"]:.3f})')
    
    # Plot ensemble PR curve
    if (ensemble_metrics['precision_curve'] is not None and 
        ensemble_metrics['recall_curve'] is not None):
        plt.plot(ensemble_metrics['recall_curve'], ensemble_metrics['precision_curve'], 
                 label=f'Ensemble (AUC = {ensemble_metrics["pr_auc"]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig("../data/plots/pr_curves_comparison.png")
    plt.close()


def plot_confusion_matrices(sklearn_metrics, autoencoder_metrics, ensemble_metrics):
    """
    Plot confusion matrices for all models
    
    Args:
        sklearn_metrics: Metrics for sklearn model
        autoencoder_metrics: Metrics for autoencoder model
        ensemble_metrics: Metrics for ensemble model
    """
    plt.figure(figsize=(15, 5))
    
    # Plot sklearn confusion matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(sklearn_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Sklearn Confusion Matrix')
    
    # Plot autoencoder confusion matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(autoencoder_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Autoencoder Confusion Matrix')
    
    # Plot ensemble confusion matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(ensemble_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Ensemble Confusion Matrix')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig("../data/plots/confusion_matrices_comparison.png")
    plt.close()


def plot_metrics_comparison(sklearn_metrics, autoencoder_metrics, ensemble_metrics):
    """
    Plot comparison of key metrics for all models
    
    Args:
        sklearn_metrics: Metrics for sklearn model
        autoencoder_metrics: Metrics for autoencoder model
        ensemble_metrics: Metrics for ensemble model
    """
    # Get metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = ['Sklearn', 'Autoencoder', 'Ensemble']
    
    # Create data for plotting
    data = {
        'Metric': [],
        'Value': [],
        'Model': []
    }
    
    for metric in metrics:
        data['Metric'].extend([metric.capitalize()] * 3)
        data['Value'].extend([sklearn_metrics[metric], autoencoder_metrics[metric], ensemble_metrics[metric]])
        data['Model'].extend(models)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    plt.title('Performance Metrics Comparison')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Model')
    
    # Save plot
    plt.savefig("../data/plots/metrics_comparison.png")
    plt.close()


def plot_anomaly_type_performance(metrics_by_type):
    """
    Plot performance by anomaly type
    
    Args:
        metrics_by_type: Dictionary of metrics by anomaly type
    """
    # Create data for plotting
    types = list(metrics_by_type.keys())
    detection_rates = [metrics_by_type[t]['detection_rate'] for t in types]
    counts = [metrics_by_type[t]['count'] for t in types]
    
    # Sort by detection rate
    sorted_indices = np.argsort(detection_rates)
    types = [types[i] for i in sorted_indices]
    detection_rates = [detection_rates[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Bar plot for detection rates
    ax1 = plt.gca()
    bars = ax1.bar(types, detection_rates, color='skyblue')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Detection Rate')
    ax1.set_title('Detection Rate by Anomaly Type')
    
    # Add count labels on top of bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={counts[i]}', ha='center', va='bottom', fontsize=10)
    
    # Add grid lines
    ax1.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig("../data/plots/anomaly_type_performance.png")
    plt.close()


def save_evaluation_results(sklearn_metrics, autoencoder_metrics, ensemble_metrics, metrics_by_type):
    """
    Save evaluation results to JSON file
    
    Args:
        sklearn_metrics: Metrics for sklearn model
        autoencoder_metrics: Metrics for autoencoder model
        ensemble_metrics: Metrics for ensemble model
        metrics_by_type: Metrics by anomaly type
    """
    # Create results dictionary
    results = {
        'sklearn': {},
        'autoencoder': {},
        'ensemble': {},
        'by_anomaly_type': metrics_by_type
    }
    
    # Copy metrics that can be serialized to JSON
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        if metric in sklearn_metrics and sklearn_metrics[metric] is not None:
            results['sklearn'][metric] = float(sklearn_metrics[metric])
        if metric in autoencoder_metrics and autoencoder_metrics[metric] is not None:
            results['autoencoder'][metric] = float(autoencoder_metrics[metric])
        if metric in ensemble_metrics and ensemble_metrics[metric] is not None:
            results['ensemble'][metric] = float(ensemble_metrics[metric])
    
    # Add classification reports
    if 'classification_report' in sklearn_metrics and sklearn_metrics['classification_report'] is not None:
        results['sklearn']['classification_report'] = sklearn_metrics['classification_report']
    if 'classification_report' in autoencoder_metrics and autoencoder_metrics['classification_report'] is not None:
        results['autoencoder']['classification_report'] = autoencoder_metrics['classification_report']
    if 'classification_report' in ensemble_metrics and ensemble_metrics['classification_report'] is not None:
        results['ensemble']['classification_report'] = ensemble_metrics['classification_report']
    
    # Add confusion matrices
    if 'confusion_matrix' in sklearn_metrics and sklearn_metrics['confusion_matrix'] is not None:
        results['sklearn']['confusion_matrix'] = sklearn_metrics['confusion_matrix'].tolist()
    if 'confusion_matrix' in autoencoder_metrics and autoencoder_metrics['confusion_matrix'] is not None:
        results['autoencoder']['confusion_matrix'] = autoencoder_metrics['confusion_matrix'].tolist()
    if 'confusion_matrix' in ensemble_metrics and ensemble_metrics['confusion_matrix'] is not None:
        results['ensemble']['confusion_matrix'] = ensemble_metrics['confusion_matrix'].tolist()
    
    # Save to file
    os.makedirs("../data", exist_ok=True)
    with open("../data/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation results saved to ../data/evaluation_results.json")


def main():
    # Load data and models
    data = load_data_and_models()
    
    if data is None:
        return
    
    # Evaluate sklearn model
    sklearn_metrics = evaluate_sklearn_model(
        data['sklearn_model'], 
        data['features'], 
        data['labels'], 
        data['sklearn_model_type']
    )
    
    # Evaluate autoencoder model
    autoencoder_metrics = evaluate_autoencoder(
        data['autoencoder_model'], 
        data['features'], 
        data['labels'], 
        data['thresholds']['autoencoder']
    )
    
    # Evaluate ensemble model
    ensemble_metrics = evaluate_ensemble(
        data['predictor'], 
        data['features'], 
        data['labels']
    )
    
    # Evaluate by anomaly type
    metrics_by_type = evaluate_by_anomaly_type(
        data['predictor'], 
        data['features'], 
        data['labels'], 
        data['anomaly_types']
    )
    
    # Plot ROC curves
    plot_roc_curves(sklearn_metrics, autoencoder_metrics, ensemble_metrics)
    
    # Plot PR curves
    plot_pr_curves(sklearn_metrics, autoencoder_metrics, ensemble_metrics)
    
    # Plot confusion matrices
    plot_confusion_matrices(sklearn_metrics, autoencoder_metrics, ensemble_metrics)
    
    # Plot metrics comparison
    plot_metrics_comparison(sklearn_metrics, autoencoder_metrics, ensemble_metrics)
    
    # Plot anomaly type performance
    plot_anomaly_type_performance(metrics_by_type)
    
    # Save evaluation results
    save_evaluation_results(sklearn_metrics, autoencoder_metrics, ensemble_metrics, metrics_by_type)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Sklearn Model: Accuracy={sklearn_metrics['accuracy']:.4f}, F1={sklearn_metrics['f1']:.4f}")
    print(f"Autoencoder Model: Accuracy={autoencoder_metrics['accuracy']:.4f}, F1={autoencoder_metrics['f1']:.4f}")
    print(f"Ensemble Model: Accuracy={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1']:.4f}")
    
    print("\nPerformance by Anomaly Type:")
    for anomaly_type, metrics in metrics_by_type.items():
        print(f"{anomaly_type}: {metrics['count']} samples, Detection Rate={metrics['detection_rate']:.4f}")
    
    print("\nAll plots saved to ../data/plots/")


if __name__ == "__main__":
    main()