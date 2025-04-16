"""
Training functionality for DoS detection models.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(
    model: tf.keras.Model,
    train_data: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
    val_data: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
    epochs: int = 20,
    batch_size: int = 64,
    class_weights: Optional[Dict[int, float]] = None,
    early_stopping_patience: int = 5,
    model_save_path: Optional[str] = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Train a model with early stopping and optional class weights.
    
    Args:
        model: TensorFlow model to train
        train_data: Training data as TF Dataset or tuple of (X, y)
        val_data: Validation data as TF Dataset or tuple of (X, y)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        class_weights: Optional class weights for imbalanced data
        early_stopping_patience: Number of epochs with no improvement after which to stop
        model_save_path: Path to save the best model
        verbose: Verbosity mode (0, 1, or 2)
        
    Returns:
        Dictionary containing training history and metadata
    """
    # Start timing
    start_time = time.time()
    
    # Create TF datasets if numpy arrays are provided
    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = train_data
    
    if isinstance(val_data, tuple):
        X_val, y_val = val_data
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = val_data
    
    # Callbacks
    callbacks = []
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint callback if save path is provided
    if model_save_path:
        # Create a unique filename using timestamp
        timestamp = int(time.time())
        model_dir = os.path.dirname(model_save_path)
        model_name = os.path.basename(model_save_path).split('.')[0]
        unique_save_path = os.path.join(model_dir, f"{model_name}_{timestamp}")  

        # Create the directory
        os.makedirs(unique_save_path, exist_ok=True)
        
        # Log the unique path
        logger.info(f"Using unique model save path: {unique_save_path}")
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=unique_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # TensorBoard callback for visualization
    log_dir = os.path.join("logs", model.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    # Train the model
    logger.info(f"Starting training for model: {model.name}")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=verbose
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Return training results
    return {
        "history": history.history,
        "training_time": training_time,
        "stopped_epoch": early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else epochs,
        "best_epoch": early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') else np.argmin(history.history['val_loss']),
        "best_val_loss": min(history.history['val_loss']),
        "best_val_accuracy": max(history.history['val_accuracy']),
        "model_save_path": model_save_path
    }

def plot_training_history(history: Dict[str, List[float]], model_name: str, save_path: Optional[str] = None):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model for plot titles
        save_path: Optional path to save the plots
    """
    # Create a directory for plots if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics for {model_name}', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(history['loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy
    axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision and recall
    axes[1, 0].plot(history['precision'], label='Training Precision')
    axes[1, 0].plot(history['val_precision'], label='Validation Precision')
    axes[1, 0].plot(history['recall'], label='Training Recall')
    axes[1, 0].plot(history['val_recall'], label='Validation Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision and Recall over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot AUC
    axes[1, 1].plot(history['auc'], label='Training AUC')
    axes[1, 1].plot(history['val_auc'], label='Validation AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('AUC over Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot if a save path is provided
    if save_path:
        plot_filename = os.path.join(save_path, f"{model_name}_training_history.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {plot_filename}")
    
    plt.show()
    
def train_all_models(
    models_dict: Dict[str, tf.keras.Model],
    train_data: Dict[str, Any],
    train_params: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models with the same training data and parameters.
    
    Args:
        models_dict: Dictionary of models {model_name: model_instance}
        train_data: Dictionary containing training and validation data
        train_params: Dictionary of training parameters
        
    Returns:
        Dictionary of training results for each model
    """
    results = {}
    
    for model_name, model in models_dict.items():
        logger.info(f"Training model: {model_name}")
        
        # Get appropriate data for the model type
        if "lstm" in model_name.lower() or "gru" in model_name.lower() or "transformer" in model_name.lower():
            # Sequence models use the sequence datasets
            train_dataset = train_data.get("train_sequence_dataset")
            val_dataset = train_data.get("val_sequence_dataset")
        else:
            # Non-sequence models use the regular datasets
            train_dataset = train_data.get("train_dataset")
            val_dataset = train_data.get("val_dataset")
        
        # Set model save path
        model_save_path = os.path.join(
            train_params.get("model_save_dir", "saved_models"),
            f"{model_name}.h5"
        )
        
        # Train the model
        model_results = train_model(
            model=model,
            train_data=train_dataset,
            val_data=val_dataset,
            epochs=train_params.get("epochs", 20),
            batch_size=train_params.get("batch_size", 64),
            class_weights=train_data.get("class_weights"),
            early_stopping_patience=train_params.get("early_stopping_patience", 5),
            model_save_path=model_save_path,
            verbose=train_params.get("verbose", 1)
        )
        
        # Plot training history
        plot_training_history(
            history=model_results["history"],
            model_name=model_name,
            save_path=train_params.get("plot_save_dir", "plots")
        )
        
        # Store results
        results[model_name] = model_results
        
    return results