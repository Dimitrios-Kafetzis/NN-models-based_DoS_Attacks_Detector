"""
Data loading functionality for network intrusion detection datasets.
Supports Bot-IoT and NSL-KDD datasets.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Optional, List, Union
import logging
import requests
import zipfile
import io
import gzip
import shutil
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs for NSL-KDD dataset
# Primary URLs (original source)
NSL_KDD_TRAIN_URL = "https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/NSL-KDD-Train.txt"
NSL_KDD_TEST_URL = "https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/NSL-KDD-Test.txt"

# Alternative URLs (GitHub mirrors)
NSL_KDD_TRAIN_ALT_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
NSL_KDD_TEST_ALT_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

# Additional mirrors
NSL_KDD_TRAIN_ALT2_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
NSL_KDD_TEST_ALT2_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

# NSL-KDD feature names
NSL_KDD_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Attack type mapping for NSL-KDD
NSL_KDD_ATTACK_TYPES = {
    'normal': 'normal',
    'back': 'dos',
    'land': 'dos',
    'neptune': 'dos',
    'pod': 'dos',
    'smurf': 'dos',
    'teardrop': 'dos',
    'mailbomb': 'dos',
    'apache2': 'dos',
    'processtable': 'dos',
    'udpstorm': 'dos',
    'ipsweep': 'probe',
    'nmap': 'probe',
    'portsweep': 'probe',
    'satan': 'probe',
    'mscan': 'probe',
    'saint': 'probe',
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'multihop': 'r2l',
    'phf': 'r2l',
    'spy': 'r2l',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
    'sendmail': 'r2l',
    'named': 'r2l',
    'snmpgetattack': 'r2l',
    'snmpguess': 'r2l',
    'xlock': 'r2l',
    'xsnoop': 'r2l',
    'worm': 'r2l',
    'buffer_overflow': 'u2r',
    'loadmodule': 'u2r',
    'perl': 'u2r',
    'rootkit': 'u2r',
    'httptunnel': 'u2r',
    'ps': 'u2r',
    'sqlattack': 'u2r',
    'xterm': 'u2r'
}

# --------------------------------------------------------------------------------
# Bot-IoT Dataset Loading Functions
# --------------------------------------------------------------------------------

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from a dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        # Try to detect the separator automatically
        df = pd.read_csv(file_path, sep=None, engine='python')
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def combine_multiple_csvs(directory_path: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Combine multiple CSV files into a single DataFrame.
    
    Args:
        directory_path: Directory containing CSV files
        file_pattern: Pattern to match CSV files
        
    Returns:
        Combined DataFrame
    """
    import glob
    
    logger.info(f"Searching for files matching {file_pattern} in {directory_path}")
    all_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not all_files:
        raise ValueError(f"No files found matching {file_pattern} in {directory_path}")
    
    logger.info(f"Found {len(all_files)} files. Combining...")
    
    # List to store individual dataframes
    dfs = []
    
    for filename in all_files:
        df = load_csv_data(filename)
        dfs.append(df)
        logger.info(f"Added {filename} with {df.shape[0]} rows")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    
    return combined_df

def sample_dataset(df: pd.DataFrame, attack_ratio: float = 0.5, random_state: int = 42) -> pd.DataFrame:
    """
    Create a balanced sample of the dataset for faster development or testing.
    
    Args:
        df: Input DataFrame
        attack_ratio: Desired ratio of attack samples to total samples
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    # Look for common label column names
    label_col = None
    for col in ['label', 'binary_label', 'Label', 'attack', 'Attack']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find a label column in the dataset")
    
    # Assuming label column contains 0 for normal and 1 for attack
    attack_samples = df[df[label_col] == 1]
    normal_samples = df[df[label_col] == 0]
    
    logger.info(f"Original dataset: {len(normal_samples)} normal samples, {len(attack_samples)} attack samples")
    
    # Calculate sizes for the new dataset
    total_desired_size = min(len(df), 100000)  # Limit to 100k samples for development
    attack_size = int(total_desired_size * attack_ratio)
    normal_size = total_desired_size - attack_size
    
    # Sample from each class
    if len(normal_samples) > normal_size:
        normal_samples = normal_samples.sample(normal_size, random_state=random_state)
    
    if len(attack_samples) > attack_size:
        attack_samples = attack_samples.sample(attack_size, random_state=random_state)
    
    # Combine and shuffle
    sampled_df = pd.concat([normal_samples, attack_samples]).sample(frac=1, random_state=random_state)
    logger.info(f"Sampled dataset: {len(sampled_df[sampled_df[label_col] == 0])} normal samples, "
                f"{len(sampled_df[sampled_df[label_col] == 1])} attack samples")
    
    return sampled_df

def load_bot_iot_dataset(data_dir: str, is_training: bool = True, sample: bool = False) -> pd.DataFrame:
    """
    Main function to load the Bot-IoT dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        is_training: Whether we're loading for training (will load all files) or testing
        sample: Whether to create a smaller balanced sample of the dataset
        
    Returns:
        DataFrame containing the dataset
    """
    logger.info(f"Loading Bot-IoT dataset from {data_dir}")
    
    try:
        # Try to load the combined dataset if it already exists
        combined_file = os.path.join(data_dir, "combined_dataset.csv")
        if os.path.exists(combined_file):
            logger.info(f"Loading pre-combined dataset from {combined_file}")
            df = load_csv_data(combined_file)
        else:
            # Otherwise combine all CSV files
            df = combine_multiple_csvs(data_dir)
            # Save combined dataset for future use
            logger.info(f"Saving combined dataset to {combined_file}")
            df.to_csv(combined_file, index=False)
        
        if sample:
            df = sample_dataset(df)
        
        logger.info(f"Final dataset shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error in load_bot_iot_dataset: {e}")
        raise

# --------------------------------------------------------------------------------
# NSL-KDD Dataset Loading Functions
# --------------------------------------------------------------------------------

def download_nslkdd_dataset(data_dir: str) -> Tuple[str, str]:
    """
    Download the NSL-KDD dataset if not already present.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Tuple of (train_file_path, test_file_path)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    train_file = os.path.join(data_dir, "NSL-KDD-Train.csv")
    test_file = os.path.join(data_dir, "NSL-KDD-Test.csv")
    
    # Define the header for both files
    header = "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,"
    header += "num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,"
    header += "num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,"
    header += "is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,"
    header += "same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,"
    header += "dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,"
    header += "dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,"
    header += "dst_host_rerror_rate,dst_host_srv_rerror_rate,label,difficulty"
    
    # Download and process training set if it doesn't exist
    if not os.path.exists(train_file):
        logger.info(f"Downloading NSL-KDD training set to {train_file}")
        
        # Check for local files first
        local_train_files = [
            os.path.join(data_dir, "KDDTrain+.txt"),
            os.path.join(data_dir, "KDDTrain.txt"),
            os.path.join(data_dir, "NSL-KDD-Train.txt")
        ]
        
        local_file_found = False
        for local_file in local_train_files:
            if os.path.exists(local_file):
                logger.info(f"Found local training file: {local_file}")
                with open(local_file, 'r') as src, open(train_file, 'w') as dst:
                    dst.write(header + "\n")
                    dst.write(src.read())
                local_file_found = True
                break
        
        if not local_file_found:
            # Try downloading from multiple sources
            urls_to_try = [
                NSL_KDD_TRAIN_URL,
                NSL_KDD_TRAIN_ALT_URL,
                NSL_KDD_TRAIN_ALT2_URL
            ]
            
            downloaded = False
            for url in urls_to_try:
                try:
                    logger.info(f"Trying to download from {url}")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Save with header
                    with open(train_file, 'w') as f:
                        f.write(header + "\n")
                        f.write(response.text)
                    
                    logger.info(f"Successfully downloaded NSL-KDD training set from {url}")
                    downloaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to download from {url}: {e}")
            
            if not downloaded:
                error_msg = "Could not download NSL-KDD training set from any source and no local file found"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    # Download and process test set if it doesn't exist
    if not os.path.exists(test_file):
        logger.info(f"Downloading NSL-KDD test set to {test_file}")
        
        # Check for local files first
        local_test_files = [
            os.path.join(data_dir, "KDDTest+.txt"),
            os.path.join(data_dir, "KDDTest.txt"),
            os.path.join(data_dir, "NSL-KDD-Test.txt")
        ]
        
        local_file_found = False
        for local_file in local_test_files:
            if os.path.exists(local_file):
                logger.info(f"Found local test file: {local_file}")
                with open(local_file, 'r') as src, open(test_file, 'w') as dst:
                    dst.write(header + "\n")
                    dst.write(src.read())
                local_file_found = True
                break
        
        if not local_file_found:
            # Try downloading from multiple sources
            urls_to_try = [
                NSL_KDD_TEST_URL,
                NSL_KDD_TEST_ALT_URL,
                NSL_KDD_TEST_ALT2_URL
            ]
            
            downloaded = False
            for url in urls_to_try:
                try:
                    logger.info(f"Trying to download from {url}")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Save with header
                    with open(test_file, 'w') as f:
                        f.write(header + "\n")
                        f.write(response.text)
                    
                    logger.info(f"Successfully downloaded NSL-KDD test set from {url}")
                    downloaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to download from {url}: {e}")
            
            if not downloaded:
                error_msg = "Could not download NSL-KDD test set from any source and no local file found"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    return train_file, test_file

def process_nslkdd_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the NSL-KDD dataset for machine learning.
    
    Args:
        df: Input DataFrame with NSL-KDD data
        
    Returns:
        Processed DataFrame
    """
    # Remove the difficulty column if present
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)
    
    # Process the label column - extract the attack type and create binary label
    if 'label' in df.columns:
        # Clean up the label column
        df['label'] = df['label'].str.strip().str.lower()
        
        # Create a new attack_type column
        df['attack_type'] = df['label'].apply(lambda x: NSL_KDD_ATTACK_TYPES.get(x, 'unknown'))
        
        # Create a binary label column (0 for normal, 1 for attack)
        df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Create more specific attack type labels for DoS
        df['is_dos'] = df['attack_type'].apply(lambda x: 1 if x == 'dos' else 0)
    
    return df

def load_nslkdd_dataset(data_dir: str, split: str = 'both') -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load the NSL-KDD dataset.
    
    Args:
        data_dir: Directory to store/load the dataset
        split: Which split to load ('train', 'test', or 'both')
        
    Returns:
        If split is 'both': Tuple of (train_df, test_df)
        Otherwise: DataFrame for the specified split
    """
    logger.info(f"Loading NSL-KDD dataset from {data_dir}")
    
    # Download the dataset if needed
    train_file, test_file = download_nslkdd_dataset(data_dir)
    
    if split.lower() == 'train' or split.lower() == 'both':
        logger.info(f"Loading NSL-KDD training data from {train_file}")
        train_df = pd.read_csv(train_file)
        train_df = process_nslkdd_dataset(train_df)
        logger.info(f"Loaded NSL-KDD training data with shape {train_df.shape}")
    
    if split.lower() == 'test' or split.lower() == 'both':
        logger.info(f"Loading NSL-KDD test data from {test_file}")
        test_df = pd.read_csv(test_file)
        test_df = process_nslkdd_dataset(test_df)
        logger.info(f"Loaded NSL-KDD test data with shape {test_df.shape}")
    
    if split.lower() == 'both':
        return train_df, test_df
    elif split.lower() == 'train':
        return train_df
    elif split.lower() == 'test':
        return test_df
    else:
        raise ValueError(f"Invalid split option: {split}. Must be 'train', 'test', or 'both'.")

# --------------------------------------------------------------------------------
# PCAP Processing Functions for hping3 Testing
# --------------------------------------------------------------------------------

def process_pcap_to_csv(pcap_file: str, output_file: str, use_cicflowmeter: bool = True) -> str:
    """
    Process a PCAP file to extract features compatible with the training dataset.
    
    Args:
        pcap_file: Path to the PCAP file
        output_file: Path for the output CSV file
        use_cicflowmeter: Whether to use CICFlowMeter for feature extraction
        
    Returns:
        Path to the generated CSV file
    """
    logger.info(f"Processing PCAP file: {pcap_file}")
    
    if use_cicflowmeter:
        # Check if CICFlowMeter is installed
        try:
            # This command assumes CICFlowMeter is installed and in PATH
            # You may need to adjust this based on your system setup
            cmd = ["cicflowmeter", "-f", pcap_file, "-c", output_file]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Error running CICFlowMeter: {result.stderr}")
                raise RuntimeError(f"CICFlowMeter failed: {result.stderr}")
                
            logger.info(f"Successfully processed PCAP using CICFlowMeter: {output_file}")
            return output_file
        except FileNotFoundError:
            logger.warning("CICFlowMeter not found in PATH. Falling back to manual extraction.")
    
    # Fallback: Use pyshark or scapy for manual feature extraction
    try:
        import pyshark
        
        # Basic feature extraction with pyshark
        logger.info("Using pyshark for basic feature extraction")
        
        # Open the pcap file
        cap = pyshark.FileCapture(pcap_file)
        
        # Extract basic features
        packets = []
        for packet in cap:
            try:
                if hasattr(packet, 'ip'):
                    packet_info = {
                        'timestamp': packet.sniff_time,
                        'src_ip': packet.ip.src,
                        'dst_ip': packet.ip.dst,
                        'protocol': packet.transport_layer if hasattr(packet, 'transport_layer') else '',
                        'length': int(packet.length)
                    }
                    
                    # Add port information if available
                    if hasattr(packet, 'tcp'):
                        packet_info['src_port'] = int(packet.tcp.srcport)
                        packet_info['dst_port'] = int(packet.tcp.dstport)
                    elif hasattr(packet, 'udp'):
                        packet_info['src_port'] = int(packet.udp.srcport)
                        packet_info['dst_port'] = int(packet.udp.dstport)
                    else:
                        packet_info['src_port'] = 0
                        packet_info['dst_port'] = 0
                        
                    packets.append(packet_info)
            except Exception as e:
                logger.warning(f"Error processing packet: {e}")
                continue
        
        # Convert to DataFrame
        packets_df = pd.DataFrame(packets)
        
        # Group by flow (src_ip, dst_ip, src_port, dst_port, protocol)
        flows = []
        for name, group in packets_df.groupby(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']):
            src_ip, dst_ip, src_port, dst_port, protocol = name
            
            # Compute basic flow statistics
            flow = {
                'Flow ID': f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}",
                'Source IP': src_ip,
                'Source Port': src_port,
                'Destination IP': dst_ip,
                'Destination Port': dst_port,
                'Protocol': protocol,
                'Timestamp': group['timestamp'].min(),
                'Flow Duration': (group['timestamp'].max() - group['timestamp'].min()).total_seconds() * 1000,
                'Total Packets': len(group),
                'Total Length': group['length'].sum(),
                'Avg Packet Size': group['length'].mean(),
                'Packet Rate': len(group) / ((group['timestamp'].max() - group['timestamp'].min()).total_seconds() if (group['timestamp'].max() - group['timestamp'].min()).total_seconds() > 0 else 1)
            }
            flows.append(flow)
        
        # Save as CSV
        flows_df = pd.DataFrame(flows)
        flows_df.to_csv(output_file, index=False)
        
        logger.info(f"Extracted {len(flows)} flows and saved to {output_file}")
        return output_file
    
    except ImportError:
        logger.error("pyshark not installed. Cannot process PCAP file.")
        raise

# --------------------------------------------------------------------------------
# General TensorFlow Dataset Creation Functions
# --------------------------------------------------------------------------------

def create_tf_dataset(
    features: np.ndarray, 
    labels: np.ndarray, 
    batch_size: int, 
    shuffle: bool = True,
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from numpy arrays.
    
    Args:
        features: Feature array
        labels: Label array
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_sequence_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    batch_size: int,
    shuffle: bool = True,
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    Create a sequence dataset for recurrent models.
    
    Args:
        features: Feature array
        labels: Label array
        window_size: Size of the sliding window
        step_size: Step size for the sliding window
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        
    Returns:
        TensorFlow dataset with sequences
    """
    # Convert to float32 for TensorFlow
    features = features.astype(np.float32)
    
    # Lists to hold sequence data
    X_sequences = []
    y_sequences = []
    
    # Create sequences
    for i in range(0, len(features) - window_size, step_size):
        X_sequences.append(features[i:i + window_size])
        # Use the label of the last element in the sequence
        y_sequences.append(labels[i + window_size - 1])
    
    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    logger.info(f"Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
    
    # Create and return the dataset
    return create_tf_dataset(X_sequences, y_sequences, batch_size, shuffle, buffer_size)