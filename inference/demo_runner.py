#!/usr/bin/env python3
"""
Demo runner for DoS detection project.
"""

import os
import argparse
import logging
import subprocess
import threading
import time
import signal
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, name):
    """Run a command in a subprocess and handle logging."""
    logger.info(f"Starting {name}: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Function to handle output from the process
    def handle_output(stream, prefix):
        for line in stream:
            print(f"{prefix} | {line.strip()}")
    
    # Create threads to handle stdout and stderr
    stdout_thread = threading.Thread(
        target=handle_output,
        args=(process.stdout, f"{name} OUT")
    )
    stderr_thread = threading.Thread(
        target=handle_output,
        args=(process.stderr, f"{name} ERR")
    )
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    return process

def run_analysis(output_dir):
    """Run analysis on the collected results."""
    logger.info("Running analysis on collected data...")
    
    analysis_cmd = [
        "python3", "-m", "inference.analyze_results",
        "--results-dir", output_dir
    ]
    
    subprocess.run(analysis_cmd, check=True)
    logger.info(f"Analysis complete. Results saved to {output_dir}")

def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="DoS Detection Demo Runner")
    parser.add_argument("--detector-interface", default="eth0", help="Network interface for the detector")
    parser.add_argument("--models-dir", default="./saved_models", help="Directory containing saved models")
    parser.add_argument("--artifacts-dir", default="./data/processed", help="Directory containing model artifacts")
    parser.add_argument("--output-dir", default="./demo_output", help="Directory to save output files")
    parser.add_argument("--target-ip", required=True, help="Target IP address for attacks")
    parser.add_argument("--attack-duration", type=int, default=20, help="Attack duration in seconds")
    parser.add_argument("--pause-duration", type=int, default=40, help="Pause duration in seconds")
    parser.add_argument("--capture-interval", type=int, default=5, help="Traffic capture interval in seconds")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Commands to run
    detector_cmd = [
        "python3", "-m", "inference.monitoring_dashboard",
        "--interface", args.detector_interface,
        "--models-dir", args.models_dir,
        "--artifacts-dir", args.artifacts_dir,
        "--output-dir", args.output_dir,
        "--interval", str(args.capture_interval)
    ]
    
    attack_cmd = [
        "python3", "-m", "inference.attack_generator",
        "--target", args.target_ip,
        "--attack-duration", str(args.attack_duration),
        "--pause-duration", str(args.pause_duration),
        "--random"
    ]
    
    # Run the detector and attack generator
    detector_process = run_command(detector_cmd, "DETECTOR")
    
    # Wait a bit to ensure detector is running
    time.sleep(5)
    
    attack_process = run_command(attack_cmd, "ATTACKER")
    
    # Set up signal handler for clean shutdown
    processes = [detector_process, attack_process]
    
    def signal_handler(sig, frame):
        logger.info("Shutting down demo...")
        for p in processes:
            try:
                if p.poll() is None:  # If process is still running
                    p.terminate()
                    p.wait(timeout=5)
            except:
                try:
                    p.kill()
                except:
                    pass
        logger.info("Demo stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for processes to complete
    try:
        detector_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)
    
    # Run analysis after demo completes
    run_analysis(args.output_dir)
    
    logger.info("Demo completed")

if __name__ == "__main__":
    main()