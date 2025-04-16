import os
import pandas as pd
import glob
from collections import Counter

# Update this path to your dataset location
dataset_path = "/media/dimitris/My Book/IoT-Bot_archive"
file_pattern = "data_*.csv"

# Get all CSV files
all_files = glob.glob(os.path.join(dataset_path, file_pattern))
print(f"Found {len(all_files)} CSV files")

# Analysis variables
total_samples = 0
attack_samples = 0
dos_samples = 0
categories = Counter()
missing_values = False
format_issues = False

# Examine each file
for file_idx, file_path in enumerate(all_files, 1):
    print(f"Processing file {file_idx}/{len(all_files)}: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
        
        # Check basic stats
        file_rows = len(df)
        total_samples += file_rows
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print(f"  Warning: File contains {df.isnull().sum().sum()} missing values")
            missing_values = True
            
        # Check attack distribution
        if 'attack' in df.columns:
            file_attacks = df['attack'].sum()
            attack_samples += file_attacks
            print(f"  Rows: {file_rows}, Attack samples: {file_attacks} ({file_attacks/file_rows*100:.2f}%)")
        else:
            print(f"  Warning: No 'attack' column found in {os.path.basename(file_path)}")
            format_issues = True
            
        # Check for DoS attacks
        if 'category' in df.columns:
            # Count attack categories
            file_categories = Counter(df['category'].str.lower())
            for category, count in file_categories.items():
                categories[category] += count
                
            # Count DoS attacks specifically
            dos_count = sum(1 for cat in df['category'].str.lower() if 'dos' in str(cat))
            dos_samples += dos_count
            if dos_count > 0:
                print(f"  DoS samples: {dos_count}")
        else:
            print(f"  Warning: No 'category' column found in {os.path.basename(file_path)}")
            format_issues = True
            
    except Exception as e:
        print(f"  Error processing file {os.path.basename(file_path)}: {e}")
        format_issues = True

# Print summary
print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)
print(f"Total samples: {total_samples}")
print(f"Attack samples: {attack_samples} ({attack_samples/total_samples*100:.2f}%)")
print(f"DoS attack samples: {dos_samples} ({dos_samples/total_samples*100:.2f}%)")
print("\nAttack categories:")
for category, count in categories.most_common():
    print(f"  {category}: {count} ({count/total_samples*100:.2f}%)")

print("\nData quality issues:")
print(f"  Missing values: {'Yes' if missing_values else 'No'}")
print(f"  Format inconsistencies: {'Yes' if format_issues else 'No'}")

print("\nSUITABILITY ASSESSMENT:")
if dos_samples > 1000 and not format_issues:
    print("✅ Dataset appears SUITABLE for DoS detection modeling")
    if dos_samples / total_samples < 0.1:
        print("⚠️ Warning: Class imbalance detected - consider balancing techniques")
else:
    print("❌ Dataset may NOT BE SUITABLE for DoS detection modeling")
    if dos_samples < 1000:
        print("   - Insufficient DoS samples (need at least 1000)")
    if format_issues:
        print("   - Format inconsistencies need to be resolved")