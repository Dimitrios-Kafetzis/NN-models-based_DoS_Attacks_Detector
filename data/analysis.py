"""
Data analysis functionality for the Bot-IoT dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dataset_summary(df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a summary of the dataset characteristics.
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the summary
        
    Returns:
        Dictionary with dataset summary information
    """
    # General dataset information
    n_samples = len(df)
    n_features = len(df.columns)
    
    # Count classes and calculate class imbalance
    if 'label' in df.columns:
        class_counts = df['label'].value_counts().to_dict()
        class_distribution = {str(k): v / n_samples for k, v in class_counts.items()}
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    else:
        class_counts = {}
        class_distribution = {}
        imbalance_ratio = None
    
    # Check for missing values
    missing_values = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    missing_percentage = total_missing / (n_samples * n_features) * 100
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Summary statistics for numeric columns
    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
    
    # Compile summary
    summary = {
        "n_samples": n_samples,
        "n_features": n_features,
        "class_counts": class_counts,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "missing_values": missing_values,
        "missing_percentage": missing_percentage,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_stats": numeric_stats
    }
    
    # Create a text summary
    text_summary = [
        f"Dataset Summary:",
        f"Number of samples: {n_samples:,}",
        f"Number of features: {n_features}",
        f"Class distribution: {class_distribution}",
        f"Class imbalance ratio: {imbalance_ratio:.2f}" if imbalance_ratio else "",
        f"Missing values: {total_missing:,} ({missing_percentage:.2f}%)",
        f"Numeric columns: {len(numeric_cols)}",
        f"Categorical columns: {len(categorical_cols)}"
    ]
    
    logger.info("\n".join(text_summary))
    
    # Save summary if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as text file
        with open(save_path + ".txt", "w") as f:
            f.write("\n".join(text_summary))
            
            # Add detailed statistics
            f.write("\n\nNumeric column statistics:\n")
            for col, stats in numeric_stats.items():
                f.write(f"\n{col}:\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value}\n")
        
        logger.info(f"Saved dataset summary to {save_path}.txt")
    
    return summary

def plot_class_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot the class distribution in the dataset.
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the plot
    """
    if 'label' not in df.columns:
        logger.warning("No 'label' column found in DataFrame")
        return
    
    # Count classes
    class_counts = df['label'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add count and percentage annotations on bars
    total = len(df)
    for i, count in enumerate(class_counts.values):
        percentage = 100 * count / total
        ax.text(i, count + (total * 0.01), f"{count:,}", ha='center')
        ax.text(i, count - (total * 0.05), f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution plot to {save_path}")
    
    plt.show()

def plot_feature_distributions(df: pd.DataFrame, top_n: int = 10, save_dir: Optional[str] = None):
    """
    Plot distributions of the most important features.
    
    Args:
        df: Input DataFrame
        top_n: Number of top features to plot
        save_dir: Optional directory to save the plots
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Exclude the label column if present
    if 'label' in numeric_cols:
        numeric_cols = numeric_cols.drop('label')
    
    # Calculate feature importance using variance
    feature_variance = df[numeric_cols].var().sort_values(ascending=False)
    top_features = feature_variance.index[:top_n]
    
    logger.info(f"Plotting distributions for top {len(top_features)} features by variance")
    
    # Plot each feature distribution
    for feature in top_features:
        plt.figure(figsize=(12, 6))
        
        # Create two subplots
        if 'label' in df.columns:
            plt.subplot(1, 2, 1)
        
        # Overall distribution
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        
        # Distribution by class if label is available
        if 'label' in df.columns:
            plt.subplot(1, 2, 2)
            sns.boxplot(x='label', y=feature, data=df)
            plt.title(f'Distribution of {feature} by Class')
            plt.xlabel('Class')
            plt.ylabel(feature)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"feature_dist_{feature}.png"), dpi=300, bbox_inches='tight')
        
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot correlation matrix of features.
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the plot
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation matrix plot to {save_path}")
    
    plt.show()
    
    # Find highly correlated features
    threshold = 0.9
    high_correlations = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                high_correlations.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    if high_correlations:
        logger.info(f"Found {len(high_correlations)} pairs of highly correlated features (threshold={threshold}):")
        for feat1, feat2, corr in high_correlations:
            logger.info(f"  {feat1} and {feat2}: {corr:.3f}")
    else:
        logger.info(f"No high correlations found (threshold={threshold})")

def analyze_bot_iot_dataset(df: pd.DataFrame, output_dir: str = "data_analysis"):
    """
    Perform comprehensive analysis of the Bot-IoT dataset.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save analysis outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Performing comprehensive analysis of the Bot-IoT dataset")
    
    # Generate dataset summary
    summary = generate_dataset_summary(df, os.path.join(output_dir, "dataset_summary"))
    
    # Plot class distribution
    plot_class_distribution(df, os.path.join(output_dir, "class_distribution.png"))
    
    # Plot feature distributions
    plot_feature_distributions(df, top_n=10, save_dir=output_dir)
    
    # Plot correlation matrix
    plot_correlation_matrix(df, os.path.join(output_dir, "correlation_matrix.png"))
    
    # Return the summary for reference
    return summary