#!/usr/bin/env python3
"""
CIC-DDoS2019 Dataset Preparation Script

This script downloads, extracts, and prepares the CIC-DDoS2019 dataset for DoS detection models.
It handles downloading CSV files from http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/,
organizing them, and creating train/validation/test splits.
"""

import os
import sys
import time
import urllib.request
import urllib.error
import zipfile
import pandas as pd
import numpy as np
import logging
import argparse
import glob
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import shutil
import random
from urllib.parse import urljoin
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cicddos2019_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset_preparation")

# Dataset URL
DATASET_URL = "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/"

def create_directory_structure(base_dir):
    """
    Create the necessary directory structure for the project.
    
    Args:
        base_dir: Base directory path
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Create required subdirectories
    directories = [
        os.path.join(base_dir, "raw"), 
        os.path.join(base_dir, "processed"),
        os.path.join(base_dir, "train"),
        os.path.join(base_dir, "validation"),
        os.path.join(base_dir, "test"),
        os.path.join(base_dir, "temp")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return {
        "raw_dir": os.path.join(base_dir, "raw"),
        "processed_dir": os.path.join(base_dir, "processed"),
        "train_dir": os.path.join(base_dir, "train"),
        "validation_dir": os.path.join(base_dir, "validation"),
        "test_dir": os.path.join(base_dir, "test"),
        "temp_dir": os.path.join(base_dir, "temp")
    }

def download_file(url, destination, chunk_size=8192):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        chunk_size: Size of chunks to download
        
    Returns:
        Boolean indicating if download was successful
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        # Remove partially downloaded file
        if os.path.exists(destination):
            os.remove(destination)
        return False

def get_download_links():
    """
    Scrape the CIC-DDoS2019 dataset page to get download links.
    
    Returns:
        List of download links
    """
    logger.info(f"Scraping download links from {DATASET_URL}...")
    
    # Hardcoded direct download links since the website structure is challenging to scrape
    direct_links = [
        # CSV files
        "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/01-12/CSV-01-12.zip",
        "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/03-11/CSV-03-11.zip",
        
        # PCAP files - only add if you need these too
        # "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/PCAPs/01-12/PCAP-01-12-1.zip",
        # "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/PCAPs/01-12/PCAP-01-12-2.zip",
        # "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/PCAPs/01-12/PCAP-01-12-3.zip",
        # "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/PCAPs/01-12/PCAP-01-12-4.zip",
        # "http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/PCAPs/03-11/PCAP-03-11.zip"
    ]
    
    logger.info(f"Using {len(direct_links)} direct download links")
    for link in direct_links:
        logger.info(f"Download link: {link}")
    
    return direct_links

def download_dataset(download_dir):
    """
    Download the CIC-DDoS2019 dataset.
    
    Args:
        download_dir: Directory to save downloaded files
        
    Returns:
        List of paths to downloaded files
    """
    logger.info("Starting download of CIC-DDoS2019 dataset...")
    
    # Get download links
    download_links = get_download_links()
    
    if not download_links:
        logger.error("No download links available. Exiting.")
        return []
    
    logger.info(f"Found {len(download_links)} files to download")
    
    # Download each file
    downloaded_files = []
    for i, url in enumerate(download_links, 1):
        filename = os.path.basename(url)
        destination = os.path.join(download_dir, filename)
        
        logger.info(f"Downloading file {i}/{len(download_links)}: {filename}")
        
        # Skip if file already exists
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            logger.info(f"File {filename} already exists. Skipping download.")
            downloaded_files.append(destination)
            continue
        
        # Download the file
        if download_file(url, destination):
            logger.info(f"Successfully downloaded {filename}")
            downloaded_files.append(destination)
        else:
            logger.error(f"Failed to download {filename}")
    
    return downloaded_files

def extract_zipfile(zip_path, extract_dir):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to
        
    Returns:
        Boolean indicating if extraction was successful
    """
    try:
        logger.info(f"Extracting {os.path.basename(zip_path)}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files for progress bar
            total_files = len(zip_ref.infolist())
            
            # Extract with progress bar
            for i, file in enumerate(zip_ref.infolist(), 1):
                zip_ref.extract(file, extract_dir)
                if i % 10 == 0 or i == total_files:  # Update every 10 files or at the end
                    logger.info(f"Extracted {i}/{total_files} files")
        
        logger.info(f"Successfully extracted {zip_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def process_csv_file(args):
    """
    Process a single CSV file (for parallel processing).
    
    Args:
        args: Tuple containing (csv_path, output_path, file_index, total_files)
        
    Returns:
        Tuple of (filename, success, rows_processed)
    """
    csv_path, output_path, file_index, total_files = args
    filename = os.path.basename(csv_path)
    
    try:
        # Define column dtypes to speed up loading
        dtypes = {
            'Unnamed': 'object',
            'Flow ID': 'object',
            'Source IP': 'object',
            'Source Port': 'float32',
            'Destination IP': 'object',
            'Destination Port': 'float32',
            'Protocol': 'float32',
            'Timestamp': 'object',
            'Label': 'object'
        }
        
        # Read the CSV file and strip whitespace from column names
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines='skip')
        
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Check if this is a CIC-DDoS2019 dataset file
        if 'Label' not in df.columns:
            # Check for alternative label columns that might be present
            possible_label_cols = ['label', 'class', 'attack', 'Category', 'category']
            found_col = None
            
            for col in possible_label_cols:
                if col in df.columns:
                    found_col = col
                    # Rename to 'Label' for consistency
                    df.rename(columns={col: 'Label'}, inplace=True)
                    logger.info(f"Found alternative label column '{col}' in {filename}")
                    break
            
            if not found_col:
                logger.warning(f"File {filename} doesn't appear to be a CIC-DDoS2019 dataset file (no 'Label' column)")
                return filename, False, 0
        
        # Clean up data
        # 1. Remove any completely empty rows or columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # 2. Handle specific known issues with CIC-DDoS2019
        # Some files have 'Infinity' or 'NaN' strings - convert to proper np.inf and np.nan
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Fill missing values appropriately
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Replace inf with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Fill with median
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        # 4. Create a binary label (0 for benign, 1 for attack)
        df['binary_label'] = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
        
        # 5. Create an attack type column (for hierarchical classification)
        # Extract just the attack type from the label (remove flags like DDoS, etc.)
        df['attack_type'] = df['Label'].apply(lambda x: str(x).split('-')[-1] if str(x).upper() != 'BENIGN' else 'BENIGN')
        
        # Save the processed file
        df.to_csv(output_path, index=False)
        
        return filename, True, len(df)
    
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return filename, False, 0

def process_dataset(raw_dir, processed_dir, max_workers=None):
    """
    Process the extracted CSV files.
    
    Args:
        raw_dir: Directory containing raw CSV files
        processed_dir: Directory to save processed files
        max_workers: Maximum number of parallel workers (default: CPU count)
        
    Returns:
        List of paths to processed files
    """
    logger.info("Processing dataset files...")
    
    # Find all CSV files
    csv_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        logger.error(f"No CSV files found in {raw_dir}")
        return []
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Set up parallel processing
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    # Prepare arguments for parallel processing
    process_args = []
    for i, csv_path in enumerate(csv_files):
        filename = os.path.basename(csv_path)
        output_path = os.path.join(processed_dir, filename)
        process_args.append((csv_path, output_path, i+1, len(csv_files)))
    
    # Process files in parallel
    processed_files = []
    total_rows = 0
    success_count = 0
    
    logger.info(f"Processing files using {max_workers} workers...")
    
    # Use either parallel or sequential processing
    if max_workers > 1:
        with Pool(max_workers) as pool:
            results = list(tqdm(pool.imap(process_csv_file, process_args), total=len(process_args)))
    else:
        results = [process_csv_file(args) for args in tqdm(process_args)]
    
    # Process results
    for filename, success, rows in results:
        if success:
            success_count += 1
            total_rows += rows
            processed_files.append(os.path.join(processed_dir, filename))
    
    logger.info(f"Successfully processed {success_count}/{len(csv_files)} files")
    logger.info(f"Total processed rows: {total_rows:,}")
    
    return processed_files

def create_data_splits(processed_dir, output_dirs, random_seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/validation/test splits from processed files.
    
    Args:
        processed_dir: Directory containing processed CSV files
        output_dirs: Dictionary of output directories
        random_seed: Random seed for reproducibility
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Dictionary with metadata about the splits
    """
    logger.info("Creating train/validation/test splits...")
    
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        logger.error(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        return {}
    
    # Find all processed CSV files
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if not csv_files:
        logger.error(f"No processed CSV files found in {processed_dir}")
        return {}
    
    # First approach: split individual files into train/val/test
    # This is better for large files to avoid loading everything into memory
    logger.info("Creating splits by sampling from each file...")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create combined CSVs for each split
    train_output = os.path.join(output_dirs["train_dir"], "cicddos2019_train.csv")
    val_output = os.path.join(output_dirs["validation_dir"], "cicddos2019_val.csv")
    test_output = os.path.join(output_dirs["test_dir"], "cicddos2019_test.csv")
    
    # Process each file
    train_count = 0
    val_count = 0
    test_count = 0
    benign_count = 0
    attack_count = 0
    attack_types = {}
    
    # Create empty DataFrames for each split with headers only
    train_df = None
    val_df = None
    test_df = None
    
    for i, csv_path in enumerate(csv_files):
        logger.info(f"Processing file {i+1}/{len(csv_files)}: {os.path.basename(csv_path)}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Count labels in this file
            file_benign = (df['Label'] == 'BENIGN').sum()
            file_attack = len(df) - file_benign
            benign_count += file_benign
            attack_count += file_attack
            
            # Count attack types
            for attack_type, count in df['Label'].value_counts().items():
                if attack_type in attack_types:
                    attack_types[attack_type] += count
                else:
                    attack_types[attack_type] = count
            
            # Shuffle the DataFrame
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            # Split the data
            train_size = int(len(df) * train_ratio)
            val_size = int(len(df) * val_ratio)
            
            train_df_part = df[:train_size]
            val_df_part = df[train_size:train_size + val_size]
            test_df_part = df[train_size + val_size:]
            
            # Update counts
            train_count += len(train_df_part)
            val_count += len(val_df_part)
            test_count += len(test_df_part)
            
            # Append to output files
            if train_df is None:
                train_df = train_df_part
                val_df = val_df_part
                test_df = test_df_part
            else:
                train_df = pd.concat([train_df, train_df_part], ignore_index=True)
                val_df = pd.concat([val_df, val_df_part], ignore_index=True)
                test_df = pd.concat([test_df, test_df_part], ignore_index=True)
                
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(csv_path)}: {e}")
    
    # Save combined split files
    logger.info("Saving train/validation/test splits...")
    
    try:
        if train_df is not None and len(train_df) > 0:
            train_df.to_csv(train_output, index=False)
            logger.info(f"Saved training set with {len(train_df):,} samples to {train_output}")
            
        if val_df is not None and len(val_df) > 0:
            val_df.to_csv(val_output, index=False)
            logger.info(f"Saved validation set with {len(val_df):,} samples to {val_output}")
            
        if test_df is not None and len(test_df) > 0:
            test_df.to_csv(test_output, index=False)
            logger.info(f"Saved test set with {len(test_df):,} samples to {test_output}")
    
    except Exception as e:
        logger.error(f"Error saving split files: {e}")
    
    # Create a smaller balanced sample for quick testing
    try:
        logger.info("Creating a smaller balanced sample for quick testing...")
        
        # Create a balanced sample with equal benign and attack samples
        if test_df is not None and len(test_df) > 0:
            benign_samples = test_df[test_df['Label'] == 'BENIGN'].sample(min(10000, (test_df['Label'] == 'BENIGN').sum()), random_state=random_seed)
            attack_samples = test_df[test_df['Label'] != 'BENIGN'].sample(min(10000, (test_df['Label'] != 'BENIGN').sum()), random_state=random_seed)
            
            # Combine and shuffle
            sample_df = pd.concat([benign_samples, attack_samples], ignore_index=True)
            sample_df = sample_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            # Save the sample
            sample_output = os.path.join(output_dirs["test_dir"], "cicddos2019_sample.csv")
            sample_df.to_csv(sample_output, index=False)
            logger.info(f"Saved balanced sample with {len(sample_df):,} samples to {sample_output}")
    
    except Exception as e:
        logger.error(f"Error creating balanced sample: {e}")
    
    # Return metadata about the splits
    return {
        "total_samples": train_count + val_count + test_count,
        "train_samples": train_count,
        "validation_samples": val_count,
        "test_samples": test_count,
        "benign_samples": benign_count,
        "attack_samples": attack_count,
        "attack_types": attack_types,
        "train_ratio": train_ratio,
        "validation_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_file": train_output,
        "validation_file": val_output,
        "test_file": test_output
    }

def clean_up(temp_dir):
    """
    Clean up temporary files.
    
    Args:
        temp_dir: Directory containing temporary files
    """
    logger.info("Cleaning up temporary files...")
    
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")

def main():
    """Main function to prepare the CIC-DDoS2019 dataset."""
    parser = argparse.ArgumentParser(description="Prepare the CIC-DDoS2019 dataset for DoS detection")
    parser.add_argument("--dir", "-d", type=str, default=None,
                       help="Base directory for the dataset (will prompt if not provided)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help="Number of worker processes for parallel processing (default: CPU count - 1)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip downloading and use existing files")
    parser.add_argument("--skip-processing", action="store_true",
                       help="Skip processing and use existing processed files")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Ratio of data for training (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Ratio of data for validation (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Ratio of data for testing (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # If no directory is specified, prompt the user
    if args.dir is None:
        print("\nThe CIC-DDoS2019 dataset can be quite large (several GB).")
        print("Please specify where you want to store the dataset files.")
        
        while True:
            base_dir = input("\nEnter the full path for dataset storage: ").strip()
            base_dir = os.path.expanduser(base_dir)  # Expand ~ to user directory if present
            
            # Check if the directory exists or can be created
            if os.path.exists(base_dir):
                if not os.path.isdir(base_dir):
                    print(f"Error: {base_dir} exists but is not a directory. Please enter a valid directory path.")
                    continue
                
                # Check if the directory is writable
                if not os.access(base_dir, os.W_OK):
                    print(f"Error: You don't have write permission for {base_dir}. Please enter a writable directory.")
                    continue
                
                # Confirm the choice if directory exists
                confirm = input(f"\nThe directory {base_dir} already exists. Do you want to use it? (y/n): ").lower().strip()
                if confirm != 'y' and confirm != 'yes':
                    continue
                
                # Check for available space
                try:
                    free_space = shutil.disk_usage(base_dir).free
                    if free_space < 20 * 1024 * 1024 * 1024:  # 20 GB
                        print(f"Warning: Only {free_space / (1024**3):.1f} GB of free space available.")
                        confirm = input("The dataset might require at least 20 GB. Continue anyway? (y/n): ").lower().strip()
                        if confirm != 'y' and confirm != 'yes':
                            continue
                except Exception:
                    # If we can't determine disk space, just continue
                    pass
                
                break
            else:
                try:
                    # Try to create the directory
                    os.makedirs(base_dir, exist_ok=True)
                    break
                except Exception as e:
                    print(f"Error creating directory {base_dir}: {e}")
                    print("Please enter a valid directory path.")
                    continue
    else:
        base_dir = os.path.abspath(args.dir)
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {base_dir}: {e}")
            print("Please enter a valid directory path.")
            sys.exit(1)
    
    base_dir = os.path.join(base_dir, "cicddos2019")
    
    print("\n" + "="*80)
    print("CIC-DDoS2019 DATASET PREPARATION TOOL".center(80))
    print("="*80)
    print(f"\nThis tool will prepare the CIC-DDoS2019 dataset for DoS attack detection.")
    print(f"Dataset will be set up in: {base_dir}")
    
    # Create directory structure
    directories = create_directory_structure(base_dir)
    
    # Download dataset
    if not args.skip_download:
        downloaded_files = download_dataset(directories["raw_dir"])
        
        # Extract zip files
        for zip_file in downloaded_files:
            if zip_file.endswith('.zip'):
                extract_zipfile(zip_file, directories["temp_dir"])
        
        # Move all extracted CSV files to raw directory
        for root, _, files in os.walk(directories["temp_dir"]):
            for file in files:
                if file.endswith('.csv'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(directories["raw_dir"], file)
                    shutil.copy2(src_path, dst_path)
    else:
        logger.info("Skipping download as requested")
    
    # Process dataset
    if not args.skip_processing:
        processed_files = process_dataset(directories["raw_dir"], directories["processed_dir"], args.workers)
    else:
        logger.info("Skipping processing as requested")
        processed_files = glob.glob(os.path.join(directories["processed_dir"], "*.csv"))
    
    # Create data splits
    if processed_files:
        splits_metadata = create_data_splits(
            directories["processed_dir"], 
            directories,
            random_seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Print summary
        if splits_metadata:
            print("\n" + "="*80)
            print("DATASET PREPARATION SUMMARY".center(80))
            print("="*80)
            print(f"\nTotal samples: {splits_metadata.get('total_samples', 0):,}")
            print(f"Train samples: {splits_metadata.get('train_samples', 0):,} ({splits_metadata.get('train_ratio', 0)*100:.1f}%)")
            print(f"Validation samples: {splits_metadata.get('validation_samples', 0):,} ({splits_metadata.get('validation_ratio', 0)*100:.1f}%)")
            print(f"Test samples: {splits_metadata.get('test_samples', 0):,} ({splits_metadata.get('test_ratio', 0)*100:.1f}%)")
            print(f"\nBenign samples: {splits_metadata.get('benign_samples', 0):,} ({splits_metadata.get('benign_samples', 0)/splits_metadata.get('total_samples', 1)*100:.1f}%)")
            print(f"Attack samples: {splits_metadata.get('attack_samples', 0):,} ({splits_metadata.get('attack_samples', 0)/splits_metadata.get('total_samples', 1)*100:.1f}%)")
            
            print("\nAttack types:")
            attack_types = splits_metadata.get('attack_types', {})
            for attack_type, count in sorted(attack_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {attack_type}: {count:,} ({count/splits_metadata.get('total_samples', 1)*100:.2f}%)")
            
            print("\nData files:")
            print(f"  - Train: {splits_metadata.get('train_file', '')}")
            print(f"  - Validation: {splits_metadata.get('validation_file', '')}")
            print(f"  - Test: {splits_metadata.get('test_file', '')}")
            print(f"  - Sample: {os.path.join(directories['test_dir'], 'cicddos2019_sample.csv')}")
    else:
        logger.error("No processed files available for creating data splits")
    
    # Clean up
    clean_up(directories["temp_dir"])
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE".center(80))
    print("="*80)
    print("\nYou can now use the processed dataset for your DoS detection models.")
    print("\nKey files for your models:")
    print(f"  - Training data: {os.path.join(directories['train_dir'], 'cicddos2019_train.csv')}")
    print(f"  - Validation data: {os.path.join(directories['validation_dir'], 'cicddos2019_val.csv')}")
    print(f"  - Test data: {os.path.join(directories['test_dir'], 'cicddos2019_test.csv')}")
    print(f"  - Sample data (for quick tests): {os.path.join(directories['test_dir'], 'cicddos2019_sample.csv')}")
    
    print("\nTo use this dataset with your DoS detection models, update your data loading code to point to these files.")

if __name__ == "__main__":
    main()