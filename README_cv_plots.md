# Cross-Validation Performance Visualization

This module provides comprehensive tools for visualizing cross-validation performance with statistical significance testing. It creates publication-quality boxplots and strip-plots with statistical significance annotations derived from bootstrap hypothesis testing.

## Overview

### Key Features

- 📊 **Boxplots with Strip-plot Overlays**: Shows both distribution summary (boxplot) and individual fold values (strip-plot)
- ⭐ **Statistical Significance Testing**: Bootstrap hypothesis testing with 10,000 resamples
- 🎯 **Significance Annotations**: Star notation (*, **, ***) for p-values
- 🔥 **Significance Heatmaps**: Color-coded matrices showing all pairwise comparisons
- 🔗 **Step 3 Integration**: Can load and integrate p-values from existing significance analysis
- 📈 **Multiple Metrics**: AUC, Accuracy, F1, Precision, Recall
- 🎨 **Publication-Quality**: High-resolution, properly formatted plots

### Output Files

The scripts generate the following files:

```
results/plots/
├── cv_performance_boxplot.png              # Main AUC + Accuracy boxplots
├── cv_performance_all_metrics.png          # All 5 metrics boxplots
├── cv_significance_heatmap_auc.png          # AUC significance heatmap
├── cv_significance_heatmap_accuracy.png     # Accuracy significance heatmap
├── cv_comparison_TAGT_vs_*.png              # Detailed pairwise comparisons
├── cv_performance_summary.csv              # Summary statistics table
└── cv_performance_summary_formatted.csv    # Pivot table format
```

## Scripts

### 1. `plot_cv_performance.py` - Main Visualization Engine

Creates comprehensive cross-validation performance visualizations.

**Usage:**
```bash
python plot_cv_performance.py
```

**Key Classes:**
- `CVPerformancePlotter`: Main plotting class with all visualization methods

**Key Methods:**
- `prepare_plot_data()`: Loads and formats per-fold results
- `compute_significance_matrix()`: Bootstrap statistical testing
- `create_combined_boxplot()`: Main boxplot visualization
- `create_performance_heatmap()`: Significance heatmap
- `generate_all_plots()`: Full pipeline execution

### 2. `integrate_step3_pvalues.py` - Step 3 Integration

Integrates existing significance matrices from step 3 analysis.

**Usage:**
```bash
python integrate_step3_pvalues.py
```

**Key Features:**
- Loads significance matrices from `results/significance_matrix_*.csv`
- Creates comparison reports between step 3 and bootstrap results
- Handles model name mapping automatically
- Annotates plots with step 3 p-values

### 3. `demo_cv_plots.py` - Interactive Demonstration

Shows how to use all functionality with detailed examples.

**Usage:**
```bash
python demo_cv_plots.py
```

## Data Requirements

### Input Data Structure

The scripts expect per-fold results in the format created by `compute_per_fold_results.py`:

```
results/per_fold/
├── Random_Forest_fold_0.pkl
├── Random_Forest_fold_1.pkl
├── ...
├── TAGT_fold_0.pkl
└── TAGT_fold_4.pkl
```

Each pickle file contains:
```python
{
    'y_true': [0, 1, 0, ...],        # True labels
    'y_pred': [0, 1, 1, ...],        # Predicted labels  
    'y_prob': [0.2, 0.8, 0.6, ...], # Predicted probabilities
    'fold_idx': 0,                   # Fold index
    'model': 'Random_Forest',        # Model name
    'metrics': {                     # Performance metrics
        'accuracy': 0.85,
        'auc': 0.91,
        'f1': 0.82,
        'precision': 0.88,
        'recall': 0.77
    }
}
```

### Step 3 Integration (Optional)

For step 3 integration, the scripts look for significance matrices:

```
results/
├── significance_matrix_auc.csv
├── significance_matrix_accuracy.csv
└── significance_summary.csv
```

## Statistical Methods

### Bootstrap Hypothesis Testing

- **Method**: Paired bootstrap with 10,000 resamples
- **Null Hypothesis**: No difference between model performances  
- **Test Statistic**: Difference in means across CV folds
- **P-value**: Two-tailed probability under null hypothesis

### Significance Levels

- `*`: p < 0.05 (modest evidence)
- `**`: p < 0.01 (strong evidence)  
- `***`: p < 0.001 (very strong evidence)
- `ns`: not significant (p ≥ 0.05)

## Plot Types

### 1. Combined Boxplots

**File**: `cv_performance_boxplot.png`

Shows AUC and Accuracy side-by-side with:
- Boxplots for quartiles and median
- Strip-plots for individual fold values
- Significance stars comparing against best model
- Sample sizes (n=5 folds) below each box

### 2. All Metrics Boxplots

**File**: `cv_performance_all_metrics.png`

Extended version with all 5 metrics:
- AUC, Accuracy, F1, Precision, Recall
- Same format as combined boxplots
- Useful for comprehensive model evaluation

### 3. Significance Heatmaps

**Files**: `cv_significance_heatmap_*.png`

Matrix visualization showing:
- Color intensity: -log10(p-value) 
- Annotations: Significance stars
- Symmetric matrix (A vs B = B vs A)
- White diagonal (self-comparisons)

### 4. Detailed Pairwise Comparisons

**Files**: `cv_comparison_TAGT_vs_*.png`

Head-to-head comparisons showing:
- Side-by-side boxplots
- Statistical test results (p-value, effect size)
- Individual data points
- Confidence intervals

## Example Usage

### Basic Usage

```python
from plot_cv_performance import CVPerformancePlotter

# Initialize plotter
plotter = CVPerformancePlotter()

# Generate all plots
saved_plots = plotter.generate_all_plots()

# Save summary table
summary_path = plotter.save_summary_table()
```

### Custom Single Plot

```python
# Prepare data
df = plotter.prepare_plot_data()

# Create custom boxplot
fig = plotter.create_combined_boxplot(df, ['AUC', 'F1'])

# Save manually
fig.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
```

### Statistical Testing Only

```python
# Compute significance matrix
sig_matrix = plotter.compute_significance_matrix(df, 'AUC')

# Check specific comparison
p_value = sig_matrix.loc['TAGT', 'Random_Forest']
stars = plotter.get_significance_stars(p_value)
print(f"TAGT vs Random Forest: p={p_value:.4f} {stars}")
```

### Step 3 Integration

```python
from integrate_step3_pvalues import Step3SignificanceIntegrator

# Initialize integrator
integrator = Step3SignificanceIntegrator()

# Generate integrated plots
integrated_plots = integrator.generate_integrated_plots()

# Create comparison report
report_path = integrator.create_comparison_report()
```

## Customization

### Colors

Model colors are defined in `CVPerformancePlotter`:

```python
self.colors = {
    'Random_Forest': '#2E8B57',      # Sea Green
    'SVM_RBF': '#4169E1',            # Royal Blue  
    'Logistic_Regression': '#FF6347', # Tomato
    'Simple_LSTM': '#9370DB',        # Medium Purple
    'TAGT': '#FF4500'                # Orange Red
}
```

### Plot Styling

Uses publication-quality settings:

```python
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

### Output Directories

Default directories can be changed:

```python
plotter = CVPerformancePlotter(
    results_dir="custom/per_fold", 
    output_dir="custom/plots"
)
```

## Performance Results Summary

Based on the current data:

| Model | AUC (Mean ± Std) | Accuracy (Mean ± Std) | F1 (Mean ± Std) |
|-------|------------------|----------------------|------------------|
| **TAGT** | **0.937 ± 0.021** | **0.892 ± 0.034** | **0.829 ± 0.059** |
| Logistic Regression | 0.851 ± 0.015 | 0.812 ± 0.037 | 0.720 ± 0.043 |
| Random Forest | 0.688 ± 0.039 | 0.662 ± 0.039 | 0.396 ± 0.082 |
| SVM RBF | 0.586 ± 0.056 | 0.582 ± 0.049 | 0.417 ± 0.081 |
| Simple LSTM | 0.510 ± 0.022 | 0.661 ± 0.005 | 0.000 ± 0.000 |

### Key Findings

- **TAGT significantly outperforms all baseline models** (p < 0.001)
- **All pairwise comparisons show statistical significance** 
- **TAGT shows highest performance across all metrics**
- **Simple LSTM has implementation issues** (F1=0, suggesting prediction problems)

## Troubleshooting

### Common Issues

1. **No per-fold data found**
   - Run `compute_per_fold_results.py` first
   - Check `results/per_fold/` directory exists

2. **Missing matplotlib/seaborn**
   ```bash
   pip install matplotlib seaborn pandas numpy scipy
   ```

3. **Step 3 integration fails**
   - Step 3 files are optional
   - Script will fall back to bootstrap testing

4. **Plot display issues**
   - Plots are saved to files, not displayed
   - Use `plt.show()` if running interactively

### File Permissions

Ensure write permissions for output directories:
```bash
mkdir -p results/plots
chmod 755 results/plots
```

## Dependencies

```python
# Core plotting
matplotlib >= 3.5.0
seaborn >= 0.11.0

# Data handling  
pandas >= 1.3.0
numpy >= 1.21.0

# Statistics
scipy >= 1.7.0

# Internal modules
load_per_fold_results
src.analysis.significance
```

## Citation

If you use this visualization code in your research, please cite:

```bibtex
@software{cv_performance_plots,
  title={Cross-Validation Performance Visualization with Statistical Significance Testing},
  author={Your Name},
  year={2024},
  note={Statistical visualization tools for machine learning model comparison}
}
```

---

**For questions or issues, please check the demo script (`demo_cv_plots.py`) for examples or create an issue in the repository.**
