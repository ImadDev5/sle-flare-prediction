import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color-blind friendly palette (Cividis-inspired with additional colors)
COLORBLIND_PALETTE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange  
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]

# Set global matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class ROCCurveAnalyzer:
    
    Automatically discover results files in the project directory.
    
    Args:
        base_path: Base directory to search
        
    Returns:
        Dictionary mapping model names to results file paths
    """
    base_path = Path(base_path)
    results_files = {}
    
    # Common result file patterns
    patterns = [
        "**/results/*results*.json",
        "**/validation*/*results*.json", 
        "**/*results*.json",
        "**/metrics/*.json"
    ]
    
    # Find all potential results files
    found_files = []
    for pattern in patterns:
        found_files.extend(base_path.glob(pattern))
    
    # Filter and name the files
    for file_path in found_files:
        file_name = file_path.stem.lower()
        
        # Skip config and metadata files
        if any(skip in file_name for skip in ['config', 'metadata', 'ppi_stats']):
            continue
        
        # Determine model name from file
        if 'cross_validation' in file_name:
            model_name = 'TAGT (CV)'
        elif 'optimized' in file_name:
            model_name = 'Optimized TAGT'
        elif 'baseline' in file_name:
            model_name = 'Baselines'
        elif 'tagt' in file_name and 'baseline' not in file_name:
            model_name = 'TAGT'
        elif 'breakthrough' in file_name:
            model_name = 'Breakthrough TAGT'
        elif 'production' in file_name:
            model_name = 'Production TAGT'
        else:
            model_name = file_path.stem.replace('_results', '').replace('results', '').title()
        
        # Avoid duplicates
        if model_name not in results_files:
            results_files[model_name] = str(file_path)
    
    logger.info(f"Auto-discovered {len(results_files)} results files")
    for model_name, path in results_files.items():
        logger.info(f"  {model_name}: {path}")
    
    return results_files

def main():
    """
    Main function to generate ROC curve comparison plots.
    """
    print("=" * 80)
    print("ROC CURVE COMPARISON GENERATOR")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ROCCurveAnalyzer(
        base_path="C:\\Users\\ADMIN\\OneDrive\\Desktop\\SLE",
        output_dir="C:\\Users\\ADMIN\\OneDrive\\Desktop\\SLE\\figures"
    )
    
    # Auto-discover results files
    results_files = auto_discover_results_files(analyzer.base_path)
    
    if not results_files:
        print("No results files found. Please ensure you have model results in the expected locations.")
        return
    
    print(f"\nFound {len(results_files)} model result files:")
    for model_name, path in results_files.items():
        print(f"  • {model_name}: {Path(path).name}")
    
    # Load AUC data and simulate ROC curves
    print("\nExtracting AUC values and simulating ROC curves...")
    model_aucs = analyzer.load_results_and_extract_auc(results_files)
    
    if not model_aucs:
        print("No AUC values found in results files.")
        return
    
        for model_name, auc_values in model_aucs.items():
        print(f"Processing {model_name}: {len(auc_values)} AUC values")
        
        # Simulate ROC curves from AUC values
        roc_data = analyzer.simulate_roc_from_auc(auc_values, model_name)
        
        # Interpolate to common grid and compute statistics
        roc_data = analyzer.interpolate_roc_curves(roc_data)
        
        # Store the processed data
        analyzer.model_roc_data[model_name] = roc_data
    
        print("\nGenerating ROC curve comparison plot...")
    saved_paths = analyzer.create_roc_comparison_plot(save_formats=['pdf', 'png'])
    
        report_path = analyzer.generate_summary_report()
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Models analyzed: {len(analyzer.model_roc_data)}")
    print(f"Output directory: {analyzer.output_dir}")
    print(f"Main figure (PDF): {analyzer.output_dir}/roc_comparison.pdf")
    print(f"Main figure (PNG): {analyzer.output_dir}/roc_comparison.png")
    print(f"Summary report: {report_path}")
    
    # Display model rankings
    print("\nModel Performance Ranking (by mean AUC):")
    print("-" * 45)
    
    sorted_models = sorted(analyzer.model_roc_data.items(), 
                          key=lambda x: x[1].get('mean_auc', 0), reverse=True)
    
    for i, (model_name, roc_data) in enumerate(sorted_models, 1):
        mean_auc = roc_data.get('mean_auc', 0)
        std_auc = roc_data.get('std_auc', 0)
        n_folds = roc_data.get('n_folds', 0)
        
        print(f"{i:2d}. {model_name:<20}: {mean_auc:.4f} ± {std_auc:.4f} (n={n_folds})")
    
    print("\nFigures saved in publication-ready formats:")
    print("  • PDF (vector): Ideal for papers and high-quality prints")
    print("  • PNG (300 DPI): Suitable for presentations and web use")

if __name__ == "__main__":
    main()