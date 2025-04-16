# DoS Attack Detection Demo

This directory contains scripts for demonstrating the DoS attack detection capabilities of the trained models.

## Components

- `real_time_detector.py`: Core functionality for DoS attack detection.
- `monitoring_dashboard.py`: Real-time dashboard for monitoring network traffic and visualizing DoS attack detection.
- `attack_generator.py`: Tool for generating simulated DoS attacks using hping3.
- `demo_runner.py`: Script to run the complete demo with both detector and attack generator.
- `analyze_results.py`: Tool to analyze the results after a demo run.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- hping3 (for attack generation)
- tcpdump (for traffic capture)
- matplotlib, pandas, numpy (for visualization)

## Running the Demo

### Setup

1. Make sure you have trained models in the `saved_models` directory:
   - linear_model.h5
   - dnn_model.h5
   - lstm_model.h5
   - gru_model.h5
   - transformer_model.h5

2. Ensure model artifacts (feature names, scaler) are in the `data/processed` directory.

3. Install required packages:
   ```bash
   sudo apt-get install hping3 tcpdump
   pip install matplotlib pandas numpy tensorflow


## Deployment on Two Machines

For the best demonstration, deploy on two separate machines:

### Machine 1: Target/Detector
- Run the monitoring dashboard
- Receives the attacks
- Analyzes traffic for DoS detection

### Machine 2: Attacker
- Runs the attack generator
- Sends various DoS attacks to Machine 1

This setup allows for a realistic demonstration of DoS attack detection in a controlled environment.

## Final Steps

1. Copy all scripts to the appropriate locations in your project.
2. Make the scripts executable with `chmod +x`.
3. Test the scripts individually first:
   - Test the dashboard without attacks
   - Test the attack generator with minimal attacks
   - Finally, run the full demo

This complete demo will showcase how your models detect different types of DoS attacks in real-time, providing valuable visualizations and analysis of their performance.