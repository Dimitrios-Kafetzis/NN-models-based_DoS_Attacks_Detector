#!/usr/bin/env python3
"""
Analyze results from DoS detection demo.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_results(results_dir):
    """Analyze all result files in the specified directory."""
    # Find all result JSON files
    result_files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.json')]
    result_files.sort()  # Sort by name (which includes timestamp)
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Extract data
    timestamps = []
    model_data = {}
    
    for filename in result_files:
        # Extract timestamp from filename
        timestamp_str = filename.replace('results_', '').replace('.json', '')
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            timestamps.append(timestamp)
            
            # Load the result file
            with open(os.path.join(results_dir, filename), 'r') as f:
                results = json.load(f)
            
            # Extract data for each model
            for model_type, model_results in results.items():
                if model_type not in model_data:
                    model_data[model_type] = {
                        'attack_ratio': [],
                        'attack_detected': [],
                        'total_flows': []
                    }
                
                model_data[model_type]['attack_ratio'].append(model_results.get('attack_ratio', 0))
                model_data[model_type]['attack_detected'].append(1 if model_results.get('is_attack_detected', False) else 0)
                model_data[model_type]['total_flows'].append(model_results.get('total_count', 0))
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create DataFrame
    df_dict = {'timestamp': timestamps}
    for model_type, data in model_data.items():
        for metric, values in data.items():
            df_dict[f"{model_type}_{metric}"] = values
    
    df = pd.DataFrame(df_dict)
    
    # Print summary
    print("\nSummary Statistics:")
    print(f"Time period: {min(timestamps)} to {max(timestamps)}")
    print(f"Total captures: {len(timestamps)}")
    
    for model_type in model_data.keys():
        attack_detected_count = sum(model_data[model_type]['attack_detected'])
        print(f"\n{model_type.upper()} Model:")
        print(f"  Attacks detected: {attack_detected_count}/{len(timestamps)} ({attack_detected_count/len(timestamps)*100:.1f}%)")
        print(f"  Average attack ratio: {np.mean(model_data[model_type]['attack_ratio'])*100:.2f}%")
        print(f"  Max attack ratio: {np.max(model_data[model_type]['attack_ratio'])*100:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot attack ratios
    plt.subplot(2, 1, 1)
    for model_type in model_data.keys():
        plt.plot(timestamps, model_data[model_type]['attack_ratio'], label=f"{model_type}")
    
    plt.title('Attack Ratio Over Time')
    plt.xlabel('Time')
    plt.ylabel('Attack Ratio')
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    
    # Plot attack detection status
    plt.subplot(2, 1, 2)
    for model_type in model_data.keys():
        plt.plot(timestamps, model_data[model_type]['attack_detected'], label=f"{model_type}")
    
    plt.title('Attack Detection Status')
    plt.xlabel('Time')
    plt.ylabel('Attack Detected')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "attack_detection_summary.png"), dpi=300)
    
    # Create comparative model performance plot
    models = list(model_data.keys())
    detection_rates = [sum(model_data[model]['attack_detected'])/len(timestamps) for model in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, detection_rates)
    plt.title('Model Detection Rate Comparison')
    plt.xlabel('Model')
    plt.ylabel('Detection Rate')
    plt.ylim(0, 1.0)
    for i, rate in enumerate(detection_rates):
        plt.text(i, rate + 0.02, f"{rate:.2f}", ha='center')
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=300)
    
    # Save analysis results to CSV
    df.to_csv(os.path.join(results_dir, "analysis_results.csv"), index=False)
    print(f"\nAnalysis complete. Results saved to {results_dir}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Analyze DoS Detection Results")
    parser.add_argument("--results-dir", required=True, help="Directory containing result files")
    
    args = parser.parse_args()
    analyze_results(args.results_dir)

if __name__ == "__main__":
    main()