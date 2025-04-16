# NN-based DoS Attacks Detector

A comprehensive Deep Learning-based system for detecting Denial of Service (DoS) attacks in network traffic. This project implements multiple neural network architectures to identify malicious traffic patterns, providing both real-time detection capabilities and model performance evaluation.

## Overview

This DoS attack detection system uses various deep learning models to identify DoS attacks by analyzing network traffic patterns. The system supports both the Bot-IoT and NSL-KDD datasets and includes tools for real-time monitoring, attack simulation, and performance evaluation.

Key features:
- Multiple neural network architectures (Linear, DNN, LSTM, GRU, Transformer)
- Support for Bot-IoT and NSL-KDD datasets
- Comprehensive preprocessing pipeline for network traffic features
- Real-time attack detection dashboard
- Attack simulation tools for demonstration and testing
- Detailed model evaluation and comparison

## Project Structure

```
dos_attacks_detector/
│
├── data/                   # Data storage and preprocessing
│   ├── nsl_kdd_dataset/    # NSL-KDD dataset files
│   ├── feature_extractor.py # PCAP feature extraction
│   ├── loader.py           # Data loading functions
│   ├── preprocessor.py     # Data preprocessing functions
│   └── analysis.py         # Dataset analysis functions
│
├── models/                 # Model architectures
│   ├── linear.py           # Linear model
│   ├── dnn.py              # Deep Neural Network
│   ├── lstm.py             # LSTM model
│   ├── gru.py              # GRU model
│   └── transformer.py      # Transformer model
│
├── training/               # Training functionality
│   └── trainer.py          # Model training functions
│
├── evaluation/             # Evaluation functionality
│   └── evaluator.py        # Model evaluation functions
│
├── inference/              # Inference and demo tools
│   ├── real_time_detector.py  # Real-time detection
│   ├── monitoring_dashboard.py # Dashboard for live monitoring
│   ├── attack_generator.py  # DoS attack generator tool
│   ├── demo_runner.py       # Demo orchestration script
│   └── analyze_results.py   # Result analysis tools
│
├── utils/                  # Helper functions
│   └── helpers.py          # Common utilities
│
├── saved_models/           # Directory for saved models
├── logs/                   # Log files
├── plots/                  # Evaluation plots and visualizations
│
├── config.py               # Configuration parameters
├── main.py                 # Main script to run the project
├── run_demo.sh             # Script to run the demo
└── requirements.txt        # Project dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- tcpdump
- hping3 (for attack simulation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Dimitrios-Kafetzis/NN-models-based_DoS_Attacks_Detector.git
cd NN-models-based_DoS_Attacks_Detector
```

2. Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install system dependencies (Linux):
```bash
sudo apt-get update
sudo apt-get install tcpdump hping3
```

5. Prepare dataset directories:
```bash
mkdir -p data/nsl_kdd_dataset
mkdir -p data/processed
mkdir -p saved_models
mkdir -p logs
mkdir -p plots
```

## Usage

### Training Models

Train models using the NSL-KDD dataset:

```bash
python main.py --dataset nsl_kdd --data_dir ./data/nsl_kdd_dataset
```

This will:
1. Download the NSL-KDD dataset if not present
2. Preprocess the data
3. Train all the models (Linear, DNN, LSTM, GRU, Transformer)
4. Evaluate the models and generate comparison plots

#### Training Options

- `--dataset`: Dataset to use ('nsl_kdd' or 'bot_iot')
- `--data_dir`: Directory containing the dataset
- `--models`: List of models to train (e.g., 'linear dnn lstm')
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--sample`: Use a smaller sample of the dataset for faster development

Example for training only specific models:
```bash
python main.py --dataset nsl_kdd --models linear dnn --epochs 10
```

### Evaluating Pre-trained Models

Evaluate without retraining:

```bash
python main.py --dataset nsl_kdd --data_dir ./data/nsl_kdd_dataset --skip_training --load_models
```

### Running the Demo

The demo shows real-time detection capabilities by simulating DoS attacks and monitoring traffic:

```bash
./run_demo.sh --interface eth0 --target 192.168.1.100
```

This will:
1. Start the monitoring dashboard to visualize attack detection in real-time
2. Generate simulated DoS attacks against the target IP
3. Analyze and save the results

For best results, run on two separate machines:
- Machine 1: Target & detector
- Machine 2: Attacker

#### Demo Components

You can also run individual components separately:

**Monitoring Dashboard:**
```bash
python -m inference.monitoring_dashboard \
  --interface eth0 \
  --models-dir ./saved_models \
  --artifacts-dir ./data/processed \
  --output-dir ./demo_output \
  --interval 5
```

**Attack Generator:**
```bash
python -m inference.attack_generator \
  --target 192.168.1.100 \
  --attack-duration 20 \
  --pause-duration 40 \
  --random
```

**Results Analysis:**
```bash
python -m inference.analyze_results \
  --results-dir ./demo_output
```

### Using Real-time Detection on Live Traffic

To use the system for detecting real DoS attacks on a network interface:

```bash
python -m inference.real_time_detector real_time \
  --interface eth0 \
  --model ./saved_models/dnn_model.h5 \
  --artifacts ./data/processed \
  --output-dir ./detection_results \
  --interval 10
```

### Analyzing a PCAP File

To analyze a previously captured PCAP file:

```bash
python -m inference.real_time_detector analyze_pcap \
  --pcap captured_traffic.pcap \
  --model ./saved_models/dnn_model.h5 \
  --artifacts ./data/processed \
  --output results.json
```

## Models

The project implements five different neural network architectures:

1. **Linear Model**: Basic neural network with linear layers
2. **DNN (Deep Neural Network)**: Deep feedforward neural network
3. **LSTM (Long Short-Term Memory)**: Recurrent neural network for sequence learning
4. **GRU (Gated Recurrent Unit)**: Alternative recurrent neural network
5. **Transformer**: Attention-based architecture

Each model is optimized for DoS attack detection and can be trained and evaluated independently.

## Datasets

### NSL-KDD Dataset

The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset for network intrusion detection. It contains various attack types, including DoS attacks.

The system will automatically download the NSL-KDD dataset when first running the training process.

### Bot-IoT Dataset

The Bot-IoT dataset contains realistic IoT traffic with various attack scenarios, including DoS attacks. Due to its size, it's not automatically downloaded. To use this dataset:

1. Download the Bot-IoT dataset from [here](https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE?path=%2FCSV%2FTraning%20and%20Testing%20Tets%20(5%25%20of%20the%20entier%20dataset)%2F10-best%20features%2F10-best-features_testing-set-V2.csv)
2. Place the CSV files in the `data/bot_iot_dataset` directory
3. Run training with the bot_iot dataset option

## Performance Evaluation

The system provides comprehensive evaluation metrics:

- Accuracy, precision, recall, F1 score
- ROC curves and AUC
- Precision-recall curves
- Confusion matrices
- Comparative model performance visualization

Evaluation results are saved in the `plots` directory.

## Real-time Detection

The real-time detection system:

1. Captures network traffic on the specified interface
2. Extracts relevant features from the traffic
3. Preprocesses the features to match the training data format
4. Applies the trained models to detect potential DoS attacks
5. Visualizes the detection results in real-time
6. Logs detection events and saves captured traffic for further analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The NSL-KDD dataset providers
- The Bot-IoT dataset team at UNSW Canberra
- TensorFlow team for the ML framework
- Contributors to the various Python libraries used in this project

## Contact

Dimitrios Kafetzis - kafetzis@aueb.gr

Project Link: [https://github.com/Dimitrios-Kafetzis/NN-models-based_DoS_Attacks_Detector](https://github.com/Dimitrios-Kafetzis/NN-models-based_DoS_Attacks_Detector)
