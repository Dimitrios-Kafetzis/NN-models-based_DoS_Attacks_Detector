"""
Helper utilities for the DoS detection project.
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure(base_dir: str = "."):
    """
    Create the directory structure for the project.
    
    Args:
        base_dir: Base directory for the project
    """
    dirs = [
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "saved_models"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "plots"),
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    logger.info(f"Set random seeds to {seed}")

def log_memory_usage():
    """
    Log memory usage of the process.
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    except ImportError:
        logger.warning("psutil module not available, skipping memory usage logging")

def calculate_model_size(model: tf.keras.Model) -> float:
    """
    Calculate the size of a TensorFlow model in MB.
    
    Args:
        model: TensorFlow model
        
    Returns:
        Model size in MB
    """
    # Get model size in bytes
    weights = model.get_weights()
    size_bytes = sum(np.prod(w.shape) * w.dtype.itemsize for w in weights)
    
    # Convert to MB
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb

def log_model_summary(models_dict: Dict[str, tf.keras.Model]):
    """
    Log summary information for multiple models.
    
    Args:
        models_dict: Dictionary of models {model_name: model_instance}
    """
    for model_name, model in models_dict.items():
        # Calculate model size
        model_size = calculate_model_size(model)
        
        # Calculate total parameters
        total_params = model.count_params()
        trainable_params = sum(np.prod(v.get_shape()) for v in model.trainable_weights)
        non_trainable_params = total_params - trainable_params
        
        # Log model information
        logger.info(f"Model: {model_name}")
        logger.info(f"  - Size: {model_size:.2f} MB")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Non-trainable parameters: {non_trainable_params:,}")
        
        # Log model layers
        logger.info(f"  - Layers:")
        for i, layer in enumerate(model.layers):
            logger.info(f"    - {i+1}: {layer.name} ({layer.__class__.__name__}) - "
                       f"Output shape: {layer.output_shape} - "
                       f"Params: {layer.count_params():,}")

def time_function(func):
    """
    Decorator to time the execution of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def create_inference_pipeline(
    model: tf.keras.Model,
    scaler_path: str,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Callable:
    """
    Create an inference pipeline for a trained model.
    
    Args:
        model: Trained TensorFlow model
        scaler_path: Path to the saved scaler
        feature_names: List of feature names (optional)
        threshold: Threshold for binary classification
        
    Returns:
        Inference function that takes input data and returns predictions
    """
    import pickle
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    def predict(input_data: Union[np.ndarray, pd.DataFrame, Dict]) -> Dict[str, Any]:
        """
        Make predictions using the trained model.
        
        Args:
            input_data: Input data as numpy array, DataFrame or dictionary
            
        Returns:
            Dictionary with prediction results
        """
        # Convert input data to the right format
        if isinstance(input_data, dict) and feature_names:
            # Convert dictionary to array in the right order
            input_array = np.array([input_data.get(feature, 0) for feature in feature_names]).reshape(1, -1)
        elif isinstance(input_data, pd.DataFrame):
            input_array = input_data.values
        else:
            input_array = input_data
            
        # Ensure 2D shape
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        
        # Apply the same preprocessing as during training
        scaled_input = scaler.transform(input_array)
        
        # Make prediction
        prediction_proba = model.predict(scaled_input)
        prediction_class = (prediction_proba >= threshold).astype(int)
        
        # Return results
        return {
            "prediction": prediction_class.flatten().tolist(),
            "probability": prediction_proba.flatten().tolist(),
            "threshold": threshold
        }
    
    return predict

def explain_model_predictions(
    model: tf.keras.Model,
    X_sample: np.ndarray,
    feature_names: List[str],
    n_samples: int = 5,
    save_path: Optional[str] = None
):
    """
    Explain model predictions using a simple feature importance analysis.
    
    Args:
        model: Trained TensorFlow model
        X_sample: Sample data for explanation
        feature_names: List of feature names
        n_samples: Number of samples to explain
        save_path: Optional path to save the explanation plots
    """
    # Make sure we have the right number of samples
    X_explain = X_sample[:n_samples] if X_sample.shape[0] > n_samples else X_sample
    
    # Get baseline predictions
    baseline_preds = model.predict(X_explain).flatten()
    
    # Calculate feature importance by perturbing one feature at a time
    importances = np.zeros((X_explain.shape[0], X_explain.shape[1]))
    
    for i in range(X_explain.shape[1]):
        # Create a copy with the i-th feature zeroed out
        X_perturbed = X_explain.copy()
        X_perturbed[:, i] = 0
        
        # Get predictions with the perturbed feature
        perturbed_preds = model.predict(X_perturbed).flatten()
        
        # Calculate importance as the difference in predictions
        importances[:, i] = np.abs(baseline_preds - perturbed_preds)
    
    # Plot feature importances for each sample
    for sample_idx in range(X_explain.shape[0]):
        # Sort features by importance
        sorted_indices = np.argsort(importances[sample_idx])[::-1]
        sorted_importances = importances[sample_idx][sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]
        
        # Plot the top 10 most important features
        plt.figure(figsize=(10, 6))
        plt.barh(range(min(10, len(sorted_features))), sorted_importances[:10], align='center')
        plt.yticks(range(min(10, len(sorted_features))), sorted_features[:10])
        plt.xlabel('Feature Importance (Absolute Prediction Difference)')
        plt.title(f'Feature Importance for Sample {sample_idx+1} (Prediction: {baseline_preds[sample_idx]:.4f})')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"feature_importance_sample_{sample_idx+1}.png"), dpi=300, bbox_inches='tight')
        
        plt.show()