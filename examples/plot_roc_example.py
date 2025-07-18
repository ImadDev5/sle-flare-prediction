"""
Example usage of the ROC curve plotting visualization.

This script demonstrates how to use the ROCCurveAnalyzer to create
publication-ready ROC curve comparison plots.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from visualization.plot_roc import ROCCurveAnalyzer, auto_discover_results_files

def main():
    """Demonstrate ROC curve plotting functionality."""
    
    print("ROC Curve Plotting Example")
    print("=" * 40)
    
    # Initialize the analyzer
    analyzer = ROCCurveAnalyzer(
        base_path=project_root,
        output_dir=project_root / "figures"
    )
    
    # Auto-discover results files
    print("1. Discovering results files...")
    results_files = auto_discover_results_files(analyzer.base_path)
    
    if not results_files:
        print("No results files found!")
        return
    
    print(f"Found {len(results_files)} results files:")
    for model_name, path in results_files.items():
        print(f"  • {model_name}")
    
    # Extract AUC values and generate ROC data
    print("\n2. Extracting AUC values and simulating ROC curves...")
    model_aucs = analyzer.load_results_and_extract_auc(results_files)
    
    if not model_aucs:
        print("No AUC values found!")
        return
    
    # Process each model
    for model_name, auc_values in model_aucs.items():
        print(f"Processing {model_name}: {len(auc_values)} AUC values")
        
        # Simulate ROC curves from AUC values
        roc_data = analyzer.simulate_roc_from_auc(auc_values, model_name)
        
        # Interpolate and compute statistics
        roc_data = analyzer.interpolate_roc_curves(roc_data)
        
        # Store the processed data
        analyzer.model_roc_data[model_name] = roc_data
    
        print("\n3. Creating ROC curve comparison plot...")
    saved_paths = analyzer.create_roc_comparison_plot(save_formats=['pdf', 'png'])
    
        print("\n4. Generating summary report...")
    report_path = analyzer.generate_summary_report()
    
    # Display results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Models analyzed: {len(analyzer.model_roc_data)}")
    print(f"Output directory: {analyzer.output_dir}")
    print(f"PDF figure: {analyzer.output_dir}/roc_comparison.pdf")
    print(f"PNG figure: {analyzer.output_dir}/roc_comparison.png")
    print(f"Summary report: {report_path}")
    
    # Show model rankings
    print("\nModel Performance Rankings:")
    print("-" * 30)
    
    sorted_models = sorted(
        analyzer.model_roc_data.items(),
        key=lambda x: x[1].get('mean_auc', 0),
        reverse=True
    )
    
    for i, (model_name, roc_data) in enumerate(sorted_models, 1):
        mean_auc = roc_data.get('mean_auc', 0)
        std_auc = roc_data.get('std_auc', 0)
        ci_lower = roc_data.get('auc_ci_lower', mean_auc)
        ci_upper = roc_data.get('auc_ci_upper', mean_auc)
        
        print(f"{i}. {model_name}")
        print(f"   AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print()

if __name__ == "__main__":
    main()