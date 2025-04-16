"""
Configuration parameters for DoS Detection project.
"""

import os
import tensorflow as tf

# Data paths
DATA_DIR = os.path.join(os.getcwd(), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.getcwd(), "data", "processed")
MODEL_SAVE_DIR = os.path.join(os.getcwd(), "saved_models")

# Ensure directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_SAVE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset parameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64
SHUFFLE_BUFFER = 10000

# Training parameters
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 0.001

# Model parameters
# Linear model
LINEAR_HIDDEN_UNITS = [64, 32]

# DNN model
DNN_HIDDEN_UNITS = [128, 64, 32]
DNN_DROPOUT_RATE = 0.3

# LSTM model
LSTM_UNITS = [64, 32]
LSTM_DROPOUT_RATE = 0.2

# GRU model
GRU_UNITS = [64, 32]
GRU_DROPOUT_RATE = 0.2

# Transformer model
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_DENSE_UNITS = 32
TRANSFORMER_DROPOUT_RATE = 0.1

# Feature engineering
WINDOW_SIZE = 10  # For sequence models (LSTM, GRU, Transformer)
STEP_SIZE = 5     # For creating sequences with overlap

# Configure TensorFlow behavior
def configure_tensorflow():
    """Configure TensorFlow for reproducibility and performance."""
    # Set random seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    
    # Optimize performance
    tf.config.optimizer.set_jit(True)  # Enable XLA optimization
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth set for GPU {gpu}")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU available, using CPU for training")