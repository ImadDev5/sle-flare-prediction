# Visualization Module

Publication-ready visualization tools for the TAGT (Temporal Attention-based Graph Transformer) research project.

## Overview

This module provides high-quality plotting utilities specifically designed for creating publication-ready figures that meet journal standards. All plots are generated with:

- Color-blind friendly palettes
- High-resolution output (300 DPI for raster, vector for PDF)
- Professional typography and formatting
- Statistical confidence intervals where applicable

## Available Tools

### ROC Curve Comparison (`plot_roc.py`)

Creates publication-ready ROC curve comparison plots with the following features:

#### Key Features
- **Mean ROC curves**: Aggregated across cross-validation folds
- **Common FPR grid**: Interpolated to 101 points for smooth curves
- **Confidence intervals**: 95% CI shown as shaded ribbons
- **Color-blind friendly**: Uses carefully selected color palette
- **AUC annotations**: Mean ± std displayed in legend
- **Multiple formats**: Exports to PDF (vector) and PNG (300 DPI)

#### Methodology
1. **Data Aggregation**: Collects AUC values from multiple result files
2. **ROC Simulation**: Generates realistic ROC curves from AUC values using beta distributions
3. **Interpolation**: Maps all curves to common FPR grid (0-1, 101 points)
4. **Statistics**: Computes mean, std, and 95% confidence intervals
5. **Visualization**: Creates publication-quality plot with confidence bands

#### Usage

```python
from src.visualization.plot_roc import ROCCurveAnalyzer

# Initialize analyzer
analyzer = ROCCurveAnalyzer(
    base_path="path/to/project",
    output_dir="figures"
)

# Auto-discover and process results
results_files = auto_discover_results_files(analyzer.base_path)
model_aucs = analyzer.load_results_and_extract_auc(results_files)

# Generate ROC data for each model
for model_name, auc_values in model_aucs.items():
    roc_data = analyzer.simulate_roc_from_auc(auc_values, model_name)
    roc_data = analyzer.interpolate_roc_curves(roc_data)
    analyzer.model_roc_data[model_name] = roc_data

# Create comparison plot
analyzer.create_roc_comparison_plot(save_formats=['pdf', 'png'])
analyzer.generate_summary_report()
```

#### Command Line Usage

```bash
python src/visualization/plot_roc.py
```

This will auto-discover all results files and generate the comparison plot.

## Input Data Requirements

### Supported File Formats
- **JSON files**: Cross-validation results, baseline comparisons, model evaluations
- **Pickle files**: Complex data structures with predictions and labels

### Expected Data Structures

#### Cross-validation Results
```json
{
  "auc": {
    "mean": 0.943,
    "std": 0.018,
    "values": [0.912, 0.955, 0.958, 0.958, 0.931]
  }
}
```

#### Baseline Results
```json
{
  "random_forest": {
    "auc": {
      "scores": [0.85, 0.87, 0.83, 0.89, 0.86]
    }
  }
}
```

#### Single Model Results
```json
{
  "test_auc": 0.89,
  "best_val_auc": 0.91
}
```

## Output Files

### Generated Figures
- `roc_comparison.pdf`: Vector format (ideal for papers)
- `roc_comparison.png`: 300 DPI raster (presentations/web)

### Analysis Report
- `roc_analysis_report.txt`: Detailed summary with:
  - Model performance rankings
  - Confidence intervals
  - Analysis methodology notes
  - Data sources and quality indicators

## Dependencies

### Required Packages
```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
pathlib
```

### Optional Packages
- `pyarrow`: For parquet file support
- `pickle`: For complex data structures

## Best Practices

### For Publications
1. **Always use PDF format** for final figures in papers
2. **Include confidence intervals** to show statistical robustness  
3. **Document simulation methodology** when using simulated ROC curves
4. **Verify color accessibility** using online color-blind simulators

### For Presentations
1. **Use PNG format** (300 DPI) for slides and posters
2. **Ensure legible font sizes** at target viewing distance
3. **Include clear legends** with AUC values
4. **Test readability** on projected displays

## Technical Notes

### ROC Curve Simulation
When detailed prediction data is unavailable, the module simulates realistic ROC curves from AUC values:

- Uses **beta distributions** to create smooth, realistic curve shapes
- **Preserves AUC values** exactly through iterative adjustment
- **Maintains monotonicity** ensuring valid ROC properties
- **Applies smoothing** for professional appearance

### Statistical Methods
- **Confidence intervals**: Student's t-distribution (95% CI)
- **Cross-validation aggregation**: Mean ± standard error
- **Interpolation**: Linear interpolation to common FPR grid
- **Monotonicity enforcement**: Cumulative maximum to ensure valid ROC curves

## Troubleshooting

### Common Issues

#### "No AUC values found"
- Check that result files contain 'auc' fields
- Verify JSON structure matches expected formats
- Ensure files are in discoverable locations

#### "Failed to interpolate ROC curves"
- Verify AUC values are in valid range [0.5, 1.0]
- Check for sufficient data points per fold
- Ensure no NaN values in input data

#### "Plot formatting issues"
- Update matplotlib to latest version
- Check system font availability
- Verify sufficient disk space for high-resolution output

### Getting Help

For issues or questions:
1. Check the analysis report for data quality warnings
2. Review input file formats and structures
3. Examine log messages for specific error details
4. Consult the example script for proper usage patterns

## Version History

- **v1.0.0**: Initial release with ROC curve comparison functionality
  - Publication-ready PDF and PNG export
  - Color-blind friendly palette
  - 95% confidence intervals
  - Automatic results file discovery
  - Comprehensive analysis reporting
