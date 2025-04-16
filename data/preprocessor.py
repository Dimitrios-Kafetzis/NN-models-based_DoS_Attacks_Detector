"""
Data preprocessing functionality for network intrusion detection datasets.
Supports Bot-IoT and NSL-KDD datasets.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from typing import Tuple, Dict, List, Any, Union, Optional
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define NSL-KDD specific constants
NSL_KDD_CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
NSL_KDD_ATTACK_CATEGORIES = {
    'normal': 'normal',
    'dos': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'mailbomb'],
    'probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'],
    'r2l': ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 
            'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named'],
    'u2r': ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
}

# --------------------------------------------------------------------------------
# Generic Preprocessing Functions
# --------------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing > 0:
        logger.info(f"Found {total_missing} missing values")
        
        # For numerical columns, fill with median
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        # For categorical columns, fill with mode
        cat_cols = df.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def identify_features_and_target(df: pd.DataFrame, target_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def handle_categorical_features(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """
    Handle categorical features through one-hot encoding.
    
    Args:
        df: Input DataFrame
        categorical_cols: Optional list of categorical columns to encode
        
    Returns:
        DataFrame with categorical features encoded
    """
    # Identify categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        logger.info(f"Encoding categorical features: {categorical_cols}")
        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def normalize_features(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    scaler_type: str = 'standard',
    save_path: str = None
) -> Tuple[np.ndarray, ...]:
    """
    Normalize features using StandardScaler or MinMaxScaler.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        scaler_type: Type of scaler ('standard' or 'minmax')
        save_path: Path to save the scaler object
        
    Returns:
        Tuple of normalized feature arrays
    """
    # Select the appropriate scaler
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}. Use 'standard' or 'minmax'")
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform validation and test sets if provided
    result = [X_train_scaled]
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        result.append(X_val_scaled)
        
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        result.append(X_test_scaled)
    
    # Save the scaler for future use
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {save_path}")
    
    return tuple(result)

def handle_class_imbalance(
    X: np.ndarray, 
    y: np.ndarray, 
    strategy: str = 'class_weight'
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Handle class imbalance in the dataset.
    
    Args:
        X: Feature array
        y: Label array
        strategy: Strategy to handle imbalance ('class_weight', 'oversample', 'undersample')
        
    Returns:
        Tuple of (X, y, class_weights)
    """
    # Count classes
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    logger.info(f"Class distribution: {class_distribution}")
    
    # Class weights for handling imbalance (for loss function)
    class_weights = None
    
    if strategy == 'class_weight':
        # Calculate class weights inversely proportional to frequencies
        total_samples = len(y)
        n_classes = len(unique)
        class_weights = {
            int(cls): total_samples / (n_classes * count) 
            for cls, count in class_distribution.items()
        }
        logger.info(f"Calculated class weights: {class_weights}")
        
    elif strategy == 'oversample':
        from sklearn.utils import resample
        
        logger.info("Using oversampling strategy")
        # Find minority and majority classes
        minority_class = min(class_distribution, key=class_distribution.get)
        majority_class = max(class_distribution, key=class_distribution.get)
        
        # Separate minority and majority classes
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        
        # Oversample minority class
        X_minority_upsampled, y_minority_upsampled = resample(
            X_minority, 
            y_minority,
            replace=True,
            n_samples=len(X_majority),
            random_state=42
        )
        
        # Combine majority class with upsampled minority class
        X = np.vstack([X_majority, X_minority_upsampled])
        y = np.hstack([y_majority, y_minority_upsampled])
        
        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        logger.info(f"After oversampling: X shape {X.shape}, y shape {y.shape}")
        
    elif strategy == 'undersample':
        from sklearn.utils import resample
        
        logger.info("Using undersampling strategy")
        # Find minority and majority classes
        minority_class = min(class_distribution, key=class_distribution.get)
        majority_class = max(class_distribution, key=class_distribution.get)
        
        # Separate minority and majority classes
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        
        # Undersample majority class
        X_majority_downsampled, y_majority_downsampled = resample(
            X_majority, 
            y_majority,
            replace=False,
            n_samples=len(X_minority),
            random_state=42
        )
        
        # Combine minority class with downsampled majority class
        X = np.vstack([X_minority, X_majority_downsampled])
        y = np.hstack([y_minority, y_majority_downsampled])
        
        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        logger.info(f"After undersampling: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, class_weights

def split_data(
    X: np.ndarray, 
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature array
        y: Label array
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Ensure ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError(f"Data split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # First split: separate training set from the rest
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state,
        stratify=y  # Ensure same class distribution across splits
    )
    
    # Second split: separate validation and test sets
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(1 - val_ratio_adjusted), 
        random_state=random_state,
        stratify=y_temp  # Ensure same class distribution
    )
    
    logger.info(f"Data split: Train {X_train.shape}, Validation {X_val.shape}, Test {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --------------------------------------------------------------------------------
# Bot-IoT Specific Preprocessing
# --------------------------------------------------------------------------------

def feature_engineering_bot_iot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the Bot-IoT dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # Bot-IoT specific feature engineering
    # Example: calculating ratios, aggregations, etc.
    
    # If we have flow-based features like bytes and packets
    if 'bytes' in df.columns and 'pkts' in df.columns:
        # Average bytes per packet
        df_engineered['avg_bytes_per_pkt'] = df['bytes'] / df['pkts'].replace(0, 1)
    
    # If we have time-based features
    if 'dur' in df.columns:
        # Calculate packets per second and bytes per second
        if 'pkts' in df.columns:
            df_engineered['pkts_per_sec'] = df['pkts'] / df['dur'].replace(0, 1)
        if 'bytes' in df.columns:
            df_engineered['bytes_per_sec'] = df['bytes'] / df['dur'].replace(0, 1)
    
    # If we have IP-based features
    if 'srcip' in df.columns and 'dstip' in df.columns:
        # Count unique destinations per source (would require groupby operations)
        # This is a simplified placeholder for actual IP-based feature engineering
        pass
    
    logger.info(f"Feature engineering: original shape {df.shape}, new shape {df_engineered.shape}")
    
    return df_engineered

def preprocess_bot_iot_dataset(
    df: pd.DataFrame,
    target_col: str = 'label',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scaler_type: str = 'standard',
    imbalance_strategy: str = 'class_weight',
    add_engineered_features: bool = True,
    save_dir: str = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Comprehensive preprocessing pipeline for the Bot-IoT dataset.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        scaler_type: Type of scaler ('standard' or 'minmax')
        imbalance_strategy: Strategy for handling class imbalance
        add_engineered_features: Whether to add engineered features
        save_dir: Directory to save preprocessing artifacts
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with processed data and metadata
    """
    logger.info("Starting preprocessing pipeline for Bot-IoT dataset")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    if add_engineered_features:
        df = feature_engineering_bot_iot(df)
    
    # Handle categorical features
    df = handle_categorical_features(df)
    
    # Separate features and target
    X, y = identify_features_and_target(df, target_col)
    
    # Convert to numpy arrays for further processing
    X = X.values
    y = y.values
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # Handle class imbalance (only on training data)
    X_train, y_train, class_weights = handle_class_imbalance(
        X_train, y_train, strategy=imbalance_strategy
    )
    
    # Save scaler path
    scaler_path = os.path.join(save_dir, "scaler.pkl") if save_dir else None
    
    # Normalize features
    X_train_scaled, X_val_scaled, X_test_scaled = normalize_features(
        X_train, X_val, X_test, scaler_type, save_path=scaler_path
    )
    
    # Get feature names for reference
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    
    # Save feature information if directory provided
    if save_dir and feature_names:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "feature_names.pkl"), 'wb') as f:
            pickle.dump(feature_names, f)
    
    logger.info("Bot-IoT preprocessing completed successfully")
    
    # Return everything needed for model training and evaluation
    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled, 
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_weights": class_weights,
        "feature_names": feature_names,
        "preprocessing_info": {
            "scaler_type": scaler_type,
            "imbalance_strategy": imbalance_strategy,
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y)),
            "class_distribution": {
                str(cls): int(count) for cls, count in 
                zip(*np.unique(y, return_counts=True))
            }
        }
    }

# --------------------------------------------------------------------------------
# NSL-KDD Specific Preprocessing
# --------------------------------------------------------------------------------

def feature_engineering_nsl_kdd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the NSL-KDD dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # Derive new features that might be useful for DoS detection
    
    # Ratio of source bytes to destination bytes
    if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
        df_engineered['src_dst_bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
    
    # Ratio of errors to packets
    if 'serror_rate' in df.columns and 'rerror_rate' in df.columns:
        df_engineered['total_error_rate'] = df['serror_rate'] + df['rerror_rate']
    
    # Create a feature for communication intensity
    if 'count' in df.columns and 'srv_count' in df.columns:
        df_engineered['communication_intensity'] = df['count'] * df['srv_count']
    
    # A feature for service usage consistency
    if 'same_srv_rate' in df.columns and 'diff_srv_rate' in df.columns:
        df_engineered['service_consistency'] = df['same_srv_rate'] - df['diff_srv_rate']
    
    logger.info(f"NSL-KDD feature engineering: original shape {df.shape}, new shape {df_engineered.shape}")
    
    return df_engineered

def encode_categorical_features_nsl_kdd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features in the NSL-KDD dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical features
    """
    # Handle categorical features specific to NSL-KDD
    df_encoded = df.copy()
    
    # Protocol type encoding
    if 'protocol_type' in df_encoded.columns:
        protocol_dummies = pd.get_dummies(df_encoded['protocol_type'], prefix='protocol', drop_first=True)
        df_encoded = pd.concat([df_encoded, protocol_dummies], axis=1)
        df_encoded.drop('protocol_type', axis=1, inplace=True)
    
    # Service encoding
    if 'service' in df_encoded.columns:
        service_dummies = pd.get_dummies(df_encoded['service'], prefix='service', drop_first=True)
        df_encoded = pd.concat([df_encoded, service_dummies], axis=1)
        df_encoded.drop('service', axis=1, inplace=True)
    
    # Flag encoding
    if 'flag' in df_encoded.columns:
        flag_dummies = pd.get_dummies(df_encoded['flag'], prefix='flag', drop_first=True)
        df_encoded = pd.concat([df_encoded, flag_dummies], axis=1)
        df_encoded.drop('flag', axis=1, inplace=True)
    
    logger.info(f"NSL-KDD categorical encoding: original shape {df.shape}, new shape {df_encoded.shape}")
    
    return df_encoded

def create_attack_type_target(df: pd.DataFrame, binary_only: bool = False) -> pd.DataFrame:
    """
    Create appropriate target variables for the NSL-KDD dataset.
    
    Args:
        df: Input DataFrame
        binary_only: Whether to create only binary classification targets
        
    Returns:
        DataFrame with added target columns
    """
    df_with_targets = df.copy()
    
    # Ensure the label column exists
    if 'label' not in df_with_targets.columns:
        logger.error("Label column not found in dataset")
        return df_with_targets
    
    # Create binary label (0 for normal, 1 for attack)
    df_with_targets['binary_label'] = df_with_targets['label'].apply(
        lambda x: 0 if str(x).lower() == 'normal' else 1
    )
    
    if not binary_only:
        # Create multi-class attack type label
        attack_type_mapping = {}
        for category, attacks in NSL_KDD_ATTACK_CATEGORIES.items():
            if isinstance(attacks, list):
                for attack in attacks:
                    attack_type_mapping[attack.lower()] = category
            else:
                attack_type_mapping[attacks.lower()] = category
        
        # Apply the mapping
        df_with_targets['attack_category'] = df_with_targets['label'].str.lower().map(
            lambda x: attack_type_mapping.get(x, 'unknown')
        )
        
        # Create specific DoS detection target
        df_with_targets['is_dos'] = df_with_targets['attack_category'].apply(
            lambda x: 1 if x == 'dos' else 0
        )
    
    return df_with_targets

def preprocess_nsl_kdd_dataset(
    df: pd.DataFrame,
    target_type: str = 'binary',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scaler_type: str = 'standard',
    imbalance_strategy: str = 'class_weight',
    add_engineered_features: bool = True,
    save_dir: str = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Comprehensive preprocessing pipeline for the NSL-KDD dataset.
    
    Args:
        df: Input DataFrame with NSL-KDD data
        target_type: Type of target to create ('binary', 'multiclass', 'dos_specific')
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        scaler_type: Type of scaler ('standard' or 'minmax')
        imbalance_strategy: Strategy for handling class imbalance
        add_engineered_features: Whether to add engineered features
        save_dir: Directory to save preprocessing artifacts
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with processed data and metadata
    """
    logger.info("Starting preprocessing pipeline for NSL-KDD dataset")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    if add_engineered_features:
        df = feature_engineering_nsl_kdd(df)
    
    # Create appropriate target variables
    df = create_attack_type_target(df, binary_only=(target_type == 'binary'))
    
    # Determine the target column based on target_type
    if target_type == 'binary':
        target_col = 'binary_label'
    elif target_type == 'multiclass':
        target_col = 'attack_category'
    elif target_type == 'dos_specific':
        target_col = 'is_dos'
    else:
        raise ValueError(f"Invalid target_type: {target_type}. Must be 'binary', 'multiclass', or 'dos_specific'")
    
    # Drop other target columns and unnecessary columns BEFORE encoding
    columns_to_drop = ['label', 'difficulty'] 
    for col in columns_to_drop:
        if col in df.columns and col != target_col:
            df.drop(col, axis=1, inplace=True)
    
    # Drop other target columns if they're not the one we want
    other_targets = ['binary_label', 'attack_category', 'is_dos', 'attack_type']
    for col in other_targets:
        if col in df.columns and col != target_col:
            df.drop(col, axis=1, inplace=True)
    
    # Handle categorical features BEFORE separating features and target
    df = encode_categorical_features_nsl_kdd(df)
    
    # Separate features and target
    X, y = identify_features_and_target(df, target_col)
    
    # For multiclass targets, encode them
    if target_type == 'multiclass':
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        logger.info(f"Encoded multiclass labels with mapping: {label_mapping}")
    
    # Convert to numpy arrays for further processing
    X = X.values
    y = y.values
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # Handle class imbalance (only on training data)
    X_train, y_train, class_weights = handle_class_imbalance(
        X_train, y_train, strategy=imbalance_strategy
    )
    
    # Save scaler path
    scaler_path = os.path.join(save_dir, "scaler.pkl") if save_dir else None
    
    # Normalize features
    X_train_scaled, X_val_scaled, X_test_scaled = normalize_features(
        X_train, X_val, X_test, scaler_type, save_path=scaler_path
    )
    
    # Get feature names for reference
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    
    # Save artifacts if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save feature names if available
        if feature_names:
            with open(os.path.join(save_dir, "feature_names.pkl"), 'wb') as f:
                pickle.dump(feature_names, f)
        
        # Save label encoder if using multiclass
        if target_type == 'multiclass' and 'le' in locals():
            with open(os.path.join(save_dir, "label_encoder.pkl"), 'wb') as f:
                pickle.dump(le, f)
    
    logger.info("NSL-KDD preprocessing completed successfully")
    
    # Return everything needed for model training and evaluation
    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled, 
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_weights": class_weights,
        "feature_names": feature_names,
        "target_type": target_type,
        "label_mapping": label_mapping if target_type == 'multiclass' else None,
        "preprocessing_info": {
            "scaler_type": scaler_type,
            "imbalance_strategy": imbalance_strategy,
            "n_features": X_train_scaled.shape[1],
            "n_classes": len(np.unique(y)),
            "class_distribution": {
                str(cls): int(count) for cls, count in 
                zip(*np.unique(y, return_counts=True))
            }
        }
    }

# --------------------------------------------------------------------------------
# hping3 PCAP to Feature Mapping Functions
# --------------------------------------------------------------------------------

def map_cicflowmeter_to_nslkdd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CICFlowMeter features to NSL-KDD features for compatibility with trained models.
    
    Args:
        df: Input DataFrame with CICFlowMeter features
        
    Returns:
        DataFrame with features mapped to NSL-KDD format
    """
    logger.info("Mapping CICFlowMeter features to NSL-KDD format")
    
    # Create a new DataFrame for the mapped features
    mapped_df = pd.DataFrame()
    
    # Basic feature mappings
    # Note: These are approximate mappings - exact equivalents may not exist
    
    # Map duration features
    if 'Flow Duration' in df.columns:
        mapped_df['duration'] = df['Flow Duration'] / 1000  # Convert to seconds
    
    # Map protocol type
    if 'Protocol' in df.columns:
        # Map numeric protocol to strings
        protocol_map = {6: 'tcp', 17: 'udp', 1: 'icmp'}
        mapped_df['protocol_type'] = df['Protocol'].map(protocol_map).fillna('other')
    
    # Map service (approximate - port-based heuristic)
    if 'Destination Port' in df.columns:
        def map_port_to_service(port):
            if port == 80 or port == 443:
                return 'http'
            elif port == 22:
                return 'ssh'
            elif port == 21:
                return 'ftp'
            elif port == 53:
                return 'domain'
            else:
                return 'other'
        
        mapped_df['service'] = df['Destination Port'].apply(map_port_to_service)
    
    # Map flag features - simplified version based on TCP flags
    if all(col in df.columns for col in ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count']):
        def determine_flag(row):
            if row['SYN Flag Count'] > 0 and row['ACK Flag Count'] > 0:
                return 'S1'
            elif row['SYN Flag Count'] > 0:
                return 'S0'
            elif row['RST Flag Count'] > 0:
                return 'REJ'
            elif row['FIN Flag Count'] > 0:
                return 'SF'
            elif row['ACK Flag Count'] > 0:
                return 'RSTO'
            else:
                return 'OTH'
        
        mapped_df['flag'] = df.apply(determine_flag, axis=1)
    else:
        mapped_df['flag'] = 'OTH'  # Default value
    
    # Map byte counts
    if 'Total Length of Fwd Packets' in df.columns:
        mapped_df['src_bytes'] = df['Total Length of Fwd Packets']
    
    if 'Total Length of Bwd Packets' in df.columns:
        mapped_df['dst_bytes'] = df['Total Length of Bwd Packets']
    
    # Map count-related features
    if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
        # Set 'count' equivalent to total connection count
        mapped_df['count'] = df['Total Fwd Packets'] + df['Total Backward Packets']
    
    # Map rate-related features
    if 'Flow Bytes/s' in df.columns:
        # Map to error rates - not exact but closest approximation
        mapped_df['serror_rate'] = df['Flow Bytes/s'].apply(lambda x: min(1.0, x / 10000))
    
    if 'Flow Packets/s' in df.columns:
        mapped_df['rerror_rate'] = df['Flow Packets/s'].apply(lambda x: min(1.0, x / 1000))
    
    # Fill in remaining features with reasonable defaults
    # These are approximations and not exact matches
    
    # Default values for land (0 = not same host/port, 1 = same host/port)
    mapped_df['land'] = 0
    
    # Default values for various counts
    zero_default_features = [
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
        'is_host_login', 'is_guest_login'
    ]
    
    for feature in zero_default_features:
        mapped_df[feature] = 0
    
    # Default values for rate-related features
    if 'same_srv_rate' not in mapped_df.columns:
        mapped_df['same_srv_rate'] = 1.0  # Assuming same service
    
    if 'diff_srv_rate' not in mapped_df.columns:
        mapped_df['diff_srv_rate'] = 0.0  # Assuming same service
    
    # Host-based features
    if 'dst_host_count' not in mapped_df.columns:
        mapped_df['dst_host_count'] = 1
    
    if 'dst_host_srv_count' not in mapped_df.columns:
        mapped_df['dst_host_srv_count'] = 1
    
    # Add label column (for consistency, will be replaced during prediction)
    mapped_df['label'] = 'normal'
    
    logger.info(f"Mapped {len(df)} CICFlowMeter records to NSL-KDD format with {len(mapped_df.columns)} features")
    
    return mapped_df

def preprocess_hping3_pcap(
    pcap_csv_path: str,
    nslkdd_model_dir: str,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess a CSV file generated from hping3 PCAP to match the NSL-KDD format
    used for training the model.
    
    Args:
        pcap_csv_path: Path to the CSV file generated from the PCAP
        nslkdd_model_dir: Directory containing the NSL-KDD model artifacts
        save_path: Optional path to save the preprocessed data
        
    Returns:
        DataFrame with preprocessed features ready for model prediction
    """
    logger.info(f"Preprocessing hping3 PCAP CSV: {pcap_csv_path}")
    
    # Load the CSV file
    try:
        df = pd.read_csv(pcap_csv_path)
        logger.info(f"Loaded CSV with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise
    
    # Map CICFlowMeter features to NSL-KDD format
    mapped_df = map_cicflowmeter_to_nslkdd(df)
    
    # Load the scaler used for the NSL-KDD model
    scaler_path = os.path.join(nslkdd_model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler not found at {scaler_path}")
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Check for feature names used during training
    feature_names_path = os.path.join(nslkdd_model_dir, "feature_names.pkl")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'rb') as f:
            training_features = pickle.load(f)
        
        logger.info(f"Using {len(training_features)} features from original training")
        
        # Ensure all training features exist in our mapped DataFrame
        for feature in training_features:
            if feature not in mapped_df.columns:
                logger.warning(f"Feature {feature} missing from mapped data, adding with zeros")
                mapped_df[feature] = 0
        
        # Keep only the features used in training
        mapped_df = mapped_df[training_features]
    
    # Handle categorical features
    mapped_df = encode_categorical_features_nsl_kdd(mapped_df)
    
    # Apply the same scaling used during training
    features_array = mapped_df.values
    scaled_features = scaler.transform(features_array)
    
    # Create a DataFrame with scaled values
    scaled_df = pd.DataFrame(scaled_features)
    
    # Save preprocessed data if requested
    if save_path:
        scaled_df.to_csv(save_path, index=False)
        logger.info(f"Saved preprocessed data to {save_path}")
    
    return scaled_df