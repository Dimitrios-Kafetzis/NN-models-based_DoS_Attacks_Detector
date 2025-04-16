#!/usr/bin/env python3
"""
Continuous monitoring script for DoS attack detection demo.
"""

import os
import sys
import time
import argparse
import logging
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.real_time_detector import DoSDetector, detect_dos_attacks, real_time_detection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DoSMonitoringDashboard:
    """Real-time dashboard for DoS attack monitoring."""
    
    def __init__(
        self, 
        interface: str,
        models_dir: str,
        artifacts_dir: str,
        output_dir: str,
        capture_interval: int = 10,
        plot_update_interval: int = 1000,  # ms
        model_types: list = ['linear', 'dnn', 'lstm', 'gru', 'transformer']
    ):
        """
        Initialize the monitoring dashboard.
        
        Args:
            interface: Network interface to monitor
            models_dir: Directory containing saved models
            artifacts_dir: Directory containing model artifacts
            output_dir: Directory to save output files
            capture_interval: Interval between captures in seconds
            plot_update_interval: Interval for plot updates in milliseconds
            model_types: List of model types to use
        """
        self.interface = interface
        self.models_dir = models_dir
        self.artifacts_dir = artifacts_dir
        self.output_dir = output_dir
        self.capture_interval = capture_interval
        self.plot_update_interval = plot_update_interval
        self.model_types = model_types
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize detectors
        self.detectors = {}
        for model_type in model_types:
            model_path = os.path.join(models_dir, f"{model_type}_model.h5")
            if os.path.exists(model_path):
                logger.info(f"Loading {model_type} model from {model_path}")
                self.detectors[model_type] = DoSDetector(
                    model_path=model_path,
                    model_artifacts_dir=artifacts_dir,
                    model_type=model_type,
                    threshold=0.5,
                    dataset_type="nsl_kdd"  # Default to NSL-KDD
                )
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        # Initialize data for plotting
        self.timestamps = []
        self.attack_ratios = {model_type: [] for model_type in self.detectors.keys()}
        self.attack_detected = {model_type: [] for model_type in self.detectors.keys()}
        
        # Status flag
        self.running = False
        
        logger.info(f"Initialized monitoring dashboard with {len(self.detectors)} models")
    
    def start_monitoring(self):
        """Start the monitoring thread and dashboard."""
        if not self.detectors:
            logger.error("No valid models loaded. Cannot start monitoring.")
            return
        
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Set up the dashboard
        self._setup_dashboard()
        
        logger.info("Monitoring and dashboard started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.running:
            try:
                # Generate timestamped filenames
                timestamp = datetime.datetime.now()
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                pcap_file = os.path.join(self.output_dir, f"capture_{timestamp_str}.pcap")
                
                # Capture traffic
                logger.info(f"Capturing traffic for {self.capture_interval} seconds...")
                from data.feature_extractor import capture_live_traffic
                capture_live_traffic(self.interface, pcap_file, self.capture_interval)
                
                # Check if the capture has traffic
                if os.path.exists(pcap_file) and os.path.getsize(pcap_file) > 100:
                    # Analyze with each model
                    results = {}
                    for model_type, detector in self.detectors.items():
                        try:
                            model_results = detector.analyze_pcap(pcap_file)
                            results[model_type] = model_results
                            
                            # Update plotting data
                            with self.data_lock:
                                if len(self.timestamps) > 60:  # Keep last 60 data points
                                    self.timestamps.pop(0)
                                    self.attack_ratios[model_type].pop(0)
                                    self.attack_detected[model_type].pop(0)
                                
                                self.timestamps.append(timestamp)
                                self.attack_ratios[model_type].append(model_results["attack_ratio"])
                                self.attack_detected[model_type].append(1 if model_results["is_attack_detected"] else 0)
                            
                            # Log results
                            attack_ratio = model_results["attack_ratio"] * 100
                            attack_status = "DETECTED" if model_results["is_attack_detected"] else "not detected"
                            logger.info(f"{model_type.upper()} model: Attack {attack_status} - {model_results['attack_count']}/{model_results['total_count']} flows ({attack_ratio:.2f}%)")
                            
                        except Exception as e:
                            logger.error(f"Error analyzing with {model_type} model: {e}")
                    
                    # Save combined results
                    results_file = os.path.join(self.output_dir, f"results_{timestamp_str}.json")
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                
                else:
                    logger.info(f"Captured file {pcap_file} is too small or does not exist, skipping analysis")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
    
    def _setup_dashboard(self):
        """Set up the real-time dashboard using matplotlib."""
        # Initialize data lock for thread safety
        self.data_lock = threading.Lock()
        
        # Set up the figure with multiple subplots
        plt.ion()  # Enable interactive mode
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle('DoS Attack Detection Dashboard', fontsize=16)
        
        # First subplot: Attack ratio over time for each model
        self.axes[0].set_title('Attack Ratio Over Time')
        self.axes[0].set_xlabel('Time')
        self.axes[0].set_ylabel('Attack Ratio')
        self.axes[0].set_ylim(0, 1.0)
        self.axes[0].grid(True)
        
        # Second subplot: Attack detection status (binary)
        self.axes[1].set_title('Attack Detection Status')
        self.axes[1].set_xlabel('Time')
        self.axes[1].set_ylabel('Attack Detected')
        self.axes[1].set_ylim(-0.1, 1.1)
        self.axes[1].set_yticks([0, 1])
        self.axes[1].set_yticklabels(['No', 'Yes'])
        self.axes[1].grid(True)
        
        # Initialize lines for each model in both plots
        self.ratio_lines = {}
        self.status_lines = {}
        
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, model_type in enumerate(self.detectors.keys()):
            color = colors[i % len(colors)]
            self.ratio_lines[model_type], = self.axes[0].plot([], [], f'{color}-', label=model_type)
            self.status_lines[model_type], = self.axes[1].plot([], [], f'{color}-', label=model_type)
        
        # Add legend
        self.axes[0].legend()
        self.axes[1].legend()
        
        # Create animation
        self.ani = FuncAnimation(
            self.fig, 
            self._update_plot,
            interval=self.plot_update_interval,
            blit=False
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def _update_plot(self, frame):
        """Update the plots with new data."""
        with self.data_lock:
            if not self.timestamps:
                return
            
            # Convert timestamps to matplotlib format for plotting
            plot_times = [t.timestamp() for t in self.timestamps]
            
            # Update attack ratio plot
            for model_type, line in self.ratio_lines.items():
                if model_type in self.attack_ratios:
                    line.set_data(plot_times, self.attack_ratios[model_type])
            
            # Update attack detection status plot
            for model_type, line in self.status_lines.items():
                if model_type in self.attack_detected:
                    line.set_data(plot_times, self.attack_detected[model_type])
            
            # Adjust x-axis limits to show all data
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view(scalex=True, scaley=False)
        
        return list(self.ratio_lines.values()) + list(self.status_lines.values())

def main():
    """Main function to run the monitoring dashboard."""
    parser = argparse.ArgumentParser(description="DoS Attack Detection Dashboard")
    parser.add_argument("--interface", required=True, help="Network interface to monitor")
    parser.add_argument("--models-dir", required=True, help="Directory containing saved models")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing model artifacts")
    parser.add_argument("--output-dir", default="./monitoring_output", help="Directory to save output files")
    parser.add_argument("--interval", type=int, default=10, help="Capture interval in seconds")
    parser.add_argument("--models", nargs="+", default=['linear', 'dnn', 'lstm', 'gru', 'transformer'], 
                        help="List of models to use")
    
    args = parser.parse_args()
    
    dashboard = DoSMonitoringDashboard(
        interface=args.interface,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        capture_interval=args.interval,
        model_types=args.models
    )
    
    try:
        dashboard.start_monitoring()
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        dashboard.stop_monitoring()
    except Exception as e:
        logger.error(f"Error in dashboard: {e}")
        dashboard.stop_monitoring()

if __name__ == "__main__":
    main()