"""
Evaluation functionality for DoS detection models.
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(
    model: tf.keras.Model,
    test_data: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 64,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a model on test data.
    
    Args:
        model: TensorFlow model to evaluate
        test_data: Test data as TF Dataset or tuple of (X, y)
        batch_size: Batch size for evaluation
        threshold: Decision threshold for binary classification
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Start timing
    start_time = time.time()
    
    # Create TF dataset if numpy arrays are provided
    if isinstance(test_data, tuple):
        X_test, y_test = test_data
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        test_dataset = test_data
        # Extract X_test and y_test for later use
        X_test, y_test = [], []
        for X_batch, y_batch in test_dataset.unbatch():
            X_test.append(X_batch.numpy())
            y_test.append(y_batch.numpy())
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    
    # Get model predictions
    logger.info(f"Evaluating model: {model.name}")
    y_pred_proba = model.predict(test_dataset)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Reshape if needed
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
        y_pred_proba = y_pred_proba.flatten()
    
    if len(y_test.shape) > 1:
        y_test = y_test.flatten()
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    # Get classification report as a dictionary
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Return evaluation results
    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "classification_report": report,
        "eval_time": eval_time
    }
    
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
    logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix as a numpy array
        model_name: Name of the model
        save_path: Optional path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        # Make sure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create dir if there is a directory path
            os.makedirs(save_dir, exist_ok=True)
            
        # Add suffix for normalized version if needed
        if normalize and not save_path.endswith("_normalized.png"):
            save_path = save_path.replace(".png", "_normalized.png")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {save_path}")
    
    plt.show()

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the ROC curve
        model_name: Name of the model
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    if save_path:
        # Create full file path for the ROC curve
        if os.path.isdir(save_path):
            plot_filename = os.path.join(save_path, f"{model_name}_roc_curve.png")
        else:
            plot_filename = save_path
            
        # Make sure directory exists
        save_dir = os.path.dirname(plot_filename)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve plot to {plot_filename}")
    
    plt.show()

def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        pr_auc: Area under the precision-recall curve
        model_name: Name of the model
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    
    if save_path:
        # Create full file path for the PR curve
        if os.path.isdir(save_path):
            plot_filename = os.path.join(save_path, f"{model_name}_pr_curve.png")
        else:
            plot_filename = save_path
            
        # Make sure directory exists
        save_dir = os.path.dirname(plot_filename)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve plot to {plot_filename}")
    
    plt.show()

def compare_models(
    eval_results: Dict[str, Dict[str, Any]],
    metric_names: List[str] = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"],
    save_path: Optional[str] = None
):
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        eval_results: Dictionary of evaluation results for each model
        metric_names: List of metric names to compare
        save_path: Optional path to save the comparison plots
    """
    # Extract metrics for each model
    model_names = list(eval_results.keys())
    metrics_data = {metric: [eval_results[model][metric] for model in model_names] for metric in metric_names}
    
    # Plot comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    width = 0.15
    x = np.arange(len(model_names))
    
    for i, metric in enumerate(metric_names):
        ax.bar(x + i*width, metrics_data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on Different Metrics')
    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, metric in enumerate(metric_names):
        for j, value in enumerate(metrics_data[metric]):
            ax.text(x[j] + i*width, value + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        # Create full file path for the comparison plot
        if os.path.isdir(save_path):
            plot_filename = os.path.join(save_path, "model_comparison.png")
        else:
            # If save_path already includes the filename
            plot_filename = save_path
            
        # Make sure directory exists
        save_dir = os.path.dirname(plot_filename)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {plot_filename}")
    
    plt.show()
    
    # Create a comparison table as DataFrame
    comparison_df = pd.DataFrame({
        metric: [eval_results[model][metric] for model in model_names] 
        for metric in metric_names
    }, index=model_names)
    
    # Add evaluation time
    comparison_df['eval_time'] = [eval_results[model]['eval_time'] for model in model_names]
    
    # Print the comparison table
    print("\nModel Comparison Table:")
    print(comparison_df.round(4))
    
    # Save comparison to CSV if path is provided
    if save_path:
        # Determine the base directory for saving additional files
        if os.path.isdir(save_path):
            base_dir = save_path
        else:
            base_dir = os.path.dirname(save_path)
            
        # Create base directory if it doesn't exist
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
            
        # Save CSV
        csv_filename = os.path.join(base_dir, "model_comparison.csv")
        comparison_df.to_csv(csv_filename)
        logger.info(f"Saved model comparison table to {csv_filename}")
        
        # Also save detailed results as JSON
        json_filename = os.path.join(base_dir, "detailed_evaluation.json")
        with open(json_filename, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Saved detailed evaluation results to {json_filename}")
    
    return comparison_df

def evaluate_all_models(
    models_dict: Dict[str, tf.keras.Model],
    test_data: Dict[str, Any],
    eval_params: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple models with the same test data and parameters.
    
    Args:
        models_dict: Dictionary of models {model_name: model_instance}
        test_data: Dictionary containing test data
        eval_params: Dictionary of evaluation parameters
        
    Returns:
        Dictionary of evaluation results for each model
    """
    results = {}
    plot_save_dir = eval_params.get("plot_save_dir", "plots")
    
    # Make sure the plot directory exists
    if plot_save_dir:
        os.makedirs(plot_save_dir, exist_ok=True)
    
    for model_name, model in models_dict.items():
        logger.info(f"Evaluating model: {model_name}")
        
        # Get appropriate data for the model type
        if "lstm" in model_name.lower() or "gru" in model_name.lower() or "transformer" in model_name.lower():
            # Sequence models use the sequence datasets
            test_dataset = test_data.get("test_sequence_dataset")
        else:
            # Non-sequence models use the regular datasets
            test_dataset = test_data.get("test_dataset")
        
        # Evaluate the model
        model_results = evaluate_model(
            model=model,
            test_data=test_dataset,
            batch_size=eval_params.get("batch_size", 64),
            threshold=eval_params.get("threshold", 0.5)
        )
        
        # Generate plots
        if eval_params.get("generate_plots", True):
            # Create full file paths for plots
            cm_path = os.path.join(plot_save_dir, f"{model_name}_confusion_matrix.png")
            roc_path = os.path.join(plot_save_dir, f"{model_name}_roc_curve.png")
            pr_path = os.path.join(plot_save_dir, f"{model_name}_pr_curve.png")
            
            # Plot confusion matrix
            plot_confusion_matrix(
                cm=np.array(model_results["confusion_matrix"]),
                model_name=model_name,
                save_path=cm_path,
                normalize=False
            )
            
            # Plot normalized confusion matrix
            plot_confusion_matrix(
                cm=np.array(model_results["confusion_matrix"]),
                model_name=model_name,
                save_path=cm_path.replace(".png", "_normalized.png"),
                normalize=True
            )
            
            # Plot ROC curve
            plot_roc_curve(
                fpr=np.array(model_results["fpr"]),
                tpr=np.array(model_results["tpr"]),
                roc_auc=model_results["roc_auc"],
                model_name=model_name,
                save_path=roc_path
            )
            
            # Plot precision-recall curve
            plot_precision_recall_curve(
                precision=np.array(model_results["precision_curve"]),
                recall=np.array(model_results["recall_curve"]),
                pr_auc=model_results["pr_auc"],
                model_name=model_name,
                save_path=pr_path
            )
        
        # Store results
        results[model_name] = model_results
    
    # Compare models
    if len(results) > 1 and eval_params.get("compare_models", True):
        comparison_path = os.path.join(plot_save_dir, "model_comparison.png")
        compare_models(
            eval_results=results,
            metric_names=eval_params.get("comparison_metrics", ["accuracy", "precision", "recall", "f1_score", "roc_auc"]),
            save_path=comparison_path
        )
    
    return results