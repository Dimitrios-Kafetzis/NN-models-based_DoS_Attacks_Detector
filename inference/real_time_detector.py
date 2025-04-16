"""
Real-time network traffic detection for DoS attacks.
This module handles loading trained models and applying them to live or captured traffic.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import time
import datetime
import pickle
import argparse
import json
from typing import Dict, List, Any, Tuple, Optional, Union
import threading
import subprocess

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_extractor import process_pcap_file, capture_live_traffic
from data.preprocessor import preprocess_hping3_pcap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DoSDetector:
    """Class for DoS attack detection using trained models."""
    
    def __init__(
        self, 
        model_path: str,
        model_artifacts_dir: str,
        model_type: str = "dnn",
        threshold: float = 0.5,
        dataset_type: str = "nsl_kdd"
    ):
        """
        Initialize the DoS detector.
        
        Args:
            model_path: Path to the saved model file
            model_artifacts_dir: Directory containing model artifacts (scaler, etc.)
            model_type: Type of model ('linear', 'dnn', 'lstm', 'gru', 'transformer')
            threshold: Threshold for binary classification
            dataset_type: Type of dataset used for training ('bot_iot', 'nsl_kdd')
        """
        self.model_path = model_path
        self.model_artifacts_dir = model_artifacts_dir
        self.model_type = model_type
        self.threshold = threshold
        self.dataset_type = dataset_type
        
        # Load the model and artifacts
        self.load_model()
        self.load_artifacts()
        
        logger.info(f"Initialized DoS detector with {model_type} model trained on {dataset_type}")
    
    def load_model(self):
        """Load the TensorFlow model."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_artifacts(self):
        """Load model artifacts (scaler, etc.)."""
        # Load scaler
        scaler_path = os.path.join(self.model_artifacts_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")
            self.scaler = None
        
        # Load feature names if available
        feature_names_path = os.path.join(self.model_artifacts_dir, "feature_names.pkl")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            logger.info(f"Loaded feature names from {feature_names_path}")
        else:
            logger.warning(f"Feature names not found at {feature_names_path}")
            self.feature_names = None
    
    def preprocess_pcap(self, pcap_file: str) -> Tuple[pd.DataFrame, str]:
        """
        Preprocess a PCAP file for inference.
        
        Args:
            pcap_file: Path to the PCAP file
            
        Returns:
            Tuple of (preprocessed_data, features_csv_path)
        """
        logger.info(f"Preprocessing PCAP file: {pcap_file}")
        
        # Extract features from PCAP
        features_csv = os.path.splitext(pcap_file)[0] + "_features.csv"
        features_csv = process_pcap_file(pcap_file, features_csv)
        
        # Preprocess based on dataset type
        if self.dataset_type == "nsl_kdd":
            processed_df = preprocess_hping3_pcap(features_csv, self.model_artifacts_dir)
        else:
            # For Bot-IoT or other datasets, implement specific preprocessing
            raise NotImplementedError(f"Preprocessing for {self.dataset_type} not implemented yet")
        
        logger.info(f"Preprocessed PCAP data with shape {processed_df.shape}")
        return processed_df, features_csv
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make predictions on preprocessed data.
        
        Args:
            X: Preprocessed features
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Making predictions on data with shape {X.shape}")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Make prediction
        y_pred_proba = self.model.predict(X)
        
        # Convert to binary predictions based on threshold
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Flatten outputs if needed
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            y_pred_proba = y_pred_proba.flatten()
        
        # Calculate statistics
        attack_count = np.sum(y_pred)
        total_count = len(y_pred)
        attack_ratio = attack_count / total_count if total_count > 0 else 0
        
        results = {
            "predictions": y_pred.tolist(),
            "probabilities": y_pred_proba.tolist(),
            "attack_count": int(attack_count),
            "total_count": int(total_count),
            "attack_ratio": float(attack_ratio),
            "is_attack_detected": bool(attack_count > 0),
            "threshold": float(self.threshold)
        }
        
        logger.info(f"Prediction results: {attack_count}/{total_count} flows classified as attacks ({attack_ratio:.2%})")
        return results
    
    def analyze_pcap(self, pcap_file: str) -> Dict[str, Any]:
        """
        Analyze a PCAP file for DoS attacks.
        
        Args:
            pcap_file: Path to the PCAP file
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing PCAP file: {pcap_file}")
        
        # Preprocess the PCAP file
        X, features_csv = self.preprocess_pcap(pcap_file)
        
        # Make predictions
        results = self.predict(X)
        
        # Add metadata
        results["pcap_file"] = pcap_file
        results["features_csv"] = features_csv
        results["timestamp"] = datetime.datetime.now().isoformat()
        results["model_type"] = self.model_type
        results["dataset_type"] = self.dataset_type
        
        return results
    
    def start_real_time_detection(
        self, 
        interface: str,
        output_dir: str,
        capture_interval: int = 10,
        bpf_filter: str = None,
        callback: Optional[callable] = None
    ):
        """
        Start real-time detection on a network interface.
        
        Args:
            interface: Network interface to monitor
            output_dir: Directory to save output files
            capture_interval: Interval between captures in seconds
            bpf_filter: Berkeley Packet Filter string
            callback: Optional callback function for detection events
        """
        logger.info(f"Starting real-time detection on interface {interface}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run detection in a loop
        try:
            while True:
                # Generate timestamped filenames
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                pcap_file = os.path.join(output_dir, f"capture_{timestamp}.pcap")
                
                # Capture traffic
                logger.info(f"Capturing traffic for {capture_interval} seconds...")
                capture_live_traffic(interface, pcap_file, capture_interval, bpf_filter)
                
                # Check if the capture has traffic
                if os.path.getsize(pcap_file) > 100:  # Minimal size check
                    # Analyze the capture
                    results = self.analyze_pcap(pcap_file)
                    
                    # Save results
                    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Call callback if provided and attack detected
                    if callback and results["is_attack_detected"]:
                        callback(results)
                else:
                    logger.info(f"Captured file {pcap_file} is too small, skipping analysis")
                
        except KeyboardInterrupt:
            logger.info("Real-time detection stopped by user")
        except Exception as e:
            logger.error(f"Error in real-time detection: {e}")
            raise

def detect_dos_attacks(
    pcap_file: str,
    model_path: str,
    model_artifacts_dir: str,
    model_type: str = "dnn",
    dataset_type: str = "nsl_kdd",
    threshold: float = 0.5,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Detect DoS attacks in a PCAP file.
    
    Args:
        pcap_file: Path to the PCAP file
        model_path: Path to the saved model
        model_artifacts_dir: Directory containing model artifacts
        model_type: Type of model used
        dataset_type: Type of dataset used for training
        threshold: Detection threshold
        output_file: Optional path to save results
        
    Returns:
        Dictionary with detection results
    """
    # Initialize detector
    detector = DoSDetector(
        model_path=model_path,
        model_artifacts_dir=model_artifacts_dir,
        model_type=model_type,
        threshold=threshold,
        dataset_type=dataset_type
    )
    
    # Analyze PCAP file
    results = detector.analyze_pcap(pcap_file)
    
    # Save results if output file is provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detection results to {output_file}")
    
    return results

def real_time_detection(
    interface: str,
    model_path: str,
    model_artifacts_dir: str,
    output_dir: str,
    model_type: str = "dnn",
    dataset_type: str = "nsl_kdd",
    threshold: float = 0.5,
    capture_interval: int = 10,
    bpf_filter: str = None
):
    """
    Run real-time DoS attack detection on a network interface.
    
    Args:
        interface: Network interface to monitor
        model_path: Path to the saved model
        model_artifacts_dir: Directory containing model artifacts
        output_dir: Directory to save output files
        model_type: Type of model used
        dataset_type: Type of dataset used for training
        threshold: Detection threshold
        capture_interval: Interval between captures in seconds
        bpf_filter: Berkeley Packet Filter string
    """
    # Define callback for attack detection
    def attack_callback(results):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attack_ratio = results["attack_ratio"] * 100
        logger.warning(f"[{timestamp}] DoS ATTACK DETECTED! {results['attack_count']} of {results['total_count']} flows classified as attacks ({attack_ratio:.2f}%)")
    
    # Initialize detector
    detector = DoSDetector(
        model_path=model_path,
        model_artifacts_dir=model_artifacts_dir,
        model_type=model_type,
        threshold=threshold,
        dataset_type=dataset_type
    )
    
    # Start real-time detection
    detector.start_real_time_detection(
        interface=interface,
        output_dir=output_dir,
        capture_interval=capture_interval,
        bpf_filter=bpf_filter,
        callback=attack_callback
    )

def main():
    """Main function to run the detector from command line."""
    parser = argparse.ArgumentParser(description="DoS Attack Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for pcap analysis
    pcap_parser = subparsers.add_parser("analyze_pcap", help="Analyze a PCAP file")
    pcap_parser.add_argument("--pcap", required=True, help="Path to the PCAP file")
    pcap_parser.add_argument("--model", required=True, help="Path to the saved model")
    pcap_parser.add_argument("--artifacts", required=True, help="Path to model artifacts directory")
    pcap_parser.add_argument("--model-type", default="dnn", help="Type of model")
    pcap_parser.add_argument("--dataset-type", default="nsl_kdd", help="Type of dataset used for training")
    pcap_parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    pcap_parser.add_argument("--output", help="Path to save results JSON")
    
    # Parser for real-time detection
    rt_parser = subparsers.add_parser("real_time", help="Run real-time detection")
    rt_parser.add_argument("--interface", required=True, help="Network interface to monitor")
    rt_parser.add_argument("--model", required=True, help="Path to the saved model")
    rt_parser.add_argument("--artifacts", required=True, help="Path to model artifacts directory")
    rt_parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    rt_parser.add_argument("--model-type", default="dnn", help="Type of model")
    rt_parser.add_argument("--dataset-type", default="nsl_kdd", help="Type of dataset used for training")
    rt_parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    rt_parser.add_argument("--interval", type=int, default=10, help="Capture interval in seconds")
    rt_parser.add_argument("--filter", help="Berkeley Packet Filter string")
    
    args = parser.parse_args()
    
    if args.command == "analyze_pcap":
        results = detect_dos_attacks(
            pcap_file=args.pcap,
            model_path=args.model,
            model_artifacts_dir=args.artifacts,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
            threshold=args.threshold,
            output_file=args.output
        )
        
        # Print summary
        print("\n" + "="*80)
        print("DoS ATTACK DETECTION RESULTS".center(80))
        print("="*80)
        print(f"PCAP file: {args.pcap}")
        print(f"Total flows analyzed: {results['total_count']}")
        print(f"Flows classified as attacks: {results['attack_count']} ({results['attack_ratio']*100:.2f}%)")
        print(f"Attack detected: {'YES' if results['is_attack_detected'] else 'NO'}")
        print("="*80 + "\n")
        
    elif args.command == "real_time":
        real_time_detection(
            interface=args.interface,
            model_path=args.model,
            model_artifacts_dir=args.artifacts,
            output_dir=args.output_dir,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
            threshold=args.threshold,
            capture_interval=args.interval,
            bpf_filter=args.filter
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()