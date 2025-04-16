#!/bin/bash
# run_demo.sh - Simple script to run the DoS detection demo

# Define default values
INTERFACE="eth0"
TARGET_IP="192.168.1.100"  # Change to the IP of the machine running the detector
MODELS_DIR="./saved_models"
ARTIFACTS_DIR="./data/processed"
OUTPUT_DIR="./demo_output"

# Check command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --interface)
      INTERFACE="$2"
      shift 2
      ;;
    --target)
      TARGET_IP="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Make sure directories exist
mkdir -p "$OUTPUT_DIR"

# Run the demo
python3 -m inference.demo_runner \
  --detector-interface "$INTERFACE" \
  --models-dir "$MODELS_DIR" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --target-ip "$TARGET_IP" \
  --attack-duration 20 \
  --pause-duration 40 \
  --capture-interval 5

exit 0