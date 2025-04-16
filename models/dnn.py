"""
Deep Neural Network model implementation for DoS detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom F1Score metric for TensorFlow 2.10.0
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        # Avoid division by zero
        return tf.math.divide_no_nan(2 * p * r, p + r)
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def create_dnn_model(
    input_dim: int,
    hidden_units: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001,
    name: str = "dnn_model"
) -> Model:
    """
    Create a Deep Neural Network with multiple dense layers.
    
    Args:
        input_dim: Dimension of input features
        hidden_units: List of hidden units for each dense layer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for the optimizer
        name: Name of the model
        
    Returns:
        Compiled TensorFlow model
    """
    # Input layer
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    x = inputs
    
    # Hidden layers with increasing complexity
    for i, units in enumerate(hidden_units):
        # Use larger dropout for larger layers
        current_dropout = dropout_rate * (1.0 - i / len(hidden_units))
        
        x = layers.Dense(
            units=units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(rate=current_dropout, name=f"dropout_{i+1}")(x)
    
    # Output layer (binary classification)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            F1Score(name="f1_score")  # Custom F1Score metric
        ]
    )
    
    logger.info(f"Created DNN model with {len(hidden_units)} hidden layers")
    model.summary(print_fn=logger.info)
    
    return model