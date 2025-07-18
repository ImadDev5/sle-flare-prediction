"""
Check what's in the fold model files
"""

import pickle
import os

fold_files = [
    "results/per_fold/TAGT_fold_0.pkl",
    "results/per_fold/TAGT_fold_1.pkl", 
    "results/per_fold/TAGT_fold_2.pkl",
    "results/per_fold/TAGT_fold_3.pkl",
    "results/per_fold/TAGT_fold_4.pkl"
]

print("=== CHECKING FOLD MODEL FILES ===")

for i, fold_file in enumerate(fold_files):
    if os.path.exists(fold_file):
        try:
            with open(fold_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\nFold {i}:")
            print(f"  Type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
                for key, value in data.items():
                    print(f"    {key}: {type(value)}")
            else:
                print(f"  Content: {data}")
                
        except Exception as e:
            print(f"  Error loading fold {i}: {e}")
    else:
        print(f"Fold {i}: File not found")

# Also check summary files
summary_files = [
    "results/per_fold/tagt_summary_results.pkl",
    "results/per_fold/summary_results.pkl"
]

print("\n=== CHECKING SUMMARY FILES ===")

for summary_file in summary_files:
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n{summary_file}:")
            print(f"  Type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, dict):
                        print(f"    {key}: {list(value.keys())}")
                    else:
                        print(f"    {key}: {type(value)} - {value}")
            else:
                print(f"  Content: {data}")
                
        except Exception as e:
            print(f"  Error loading {summary_file}: {e}")
    else:
        print(f"{summary_file}: Not found")