"""
Transformer model implementation for DoS detection.
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

class TransformerBlock(layers.Layer):
    """
    Transformer block implementation.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs, training=False):
        # Multi-head self attention
        attention_output = self.att(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
        })
        return config

def create_transformer_model(
    input_shape: tuple,
    num_layers: int = 2,
    d_model: int = 64,
    num_heads: int = 4,
    dff: int = 128,
    dense_units: List[int] = [32],
    dropout_rate: float = 0.1,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001,
    name: str = "transformer_model"
) -> Model:
    """
    Create a Transformer-based model for DoS detection.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        num_layers: Number of transformer layers
        d_model: Dimension of the transformer model
        num_heads: Number of attention heads
        dff: Dimension of the feed-forward network
        dense_units: List of units for dense layers after transformer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for the optimizer
        name: Name of the model
        
    Returns:
        Compiled TensorFlow model
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name="input")
    
    # Initial projection to d_model dimensions
    x = layers.Dense(d_model, kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    
    # Positional encoding could be added here
    # For simplicity, we'll rely on the model to learn position information
    
    # Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(
            embed_dim=d_model,
            num_heads=num_heads,
            ff_dim=dff,
            rate=dropout_rate
        )(x)
    
    # Global pooling to get a fixed-size vector
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units=units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(rate=dropout_rate, name=f"dropout_{i+1}")(x)
    
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
    
    logger.info(f"Created Transformer model with {num_layers} transformer layers")
    model.summary(print_fn=logger.info)
    
    return model