# Metric Bar Charts with Bootstrap Confidence Intervals

## Overview

The `plot_metrics_bars.py` script generates professional-quality horizontal bar charts for machine learning model performance comparison. It creates visualizations for five key metrics (AUC, Accuracy, Precision, Recall, F1) with bootstrap 95% confidence interval error bars.

## Features

### Visual Design
- **Horizontal bar charts** for easy model name reading
- **Bootstrap 95% confidence intervals** displayed as error bars
- **Color-coded model types**: 
  - Red bars for TAGT models
  - Teal bars for baseline models
- **Bold black outlines** for TAGT models for emphasis
- **Value labels** on each bar showing exact metric scores
- **Professional styling** suitable for publications

### Output Files

The script generates the following files in the `figures/` directory:

#### Individual Metric Charts
- `auc_bars.pdf/png` - AUC comparison chart
- `acc_bars.pdf/png` - Accuracy comparison chart  
- `prec_bars.pdf/png` - Precision comparison chart
- `recall_bars.pdf/png` - Recall comparison chart
- `f1_bars.pdf/png` - F1 score comparison chart

#### Grouped Metrics Chart
- `metric_bars.pdf/png` - All metrics in one comprehensive chart

## Data Requirements

The script expects a CSV file at `results/results_summary_with_ci.csv` with the following structure:

```csv
model,split_type,auc,auc_lower,auc_upper,acc,acc_lower,acc_upper,f1,f1_lower,f1_upper,prec,prec_lower,prec_upper,recall,recall_lower,recall_upper
```

- **Base metrics**: `auc`, `acc`, `prec`, `recall`, `f1`
- **Confidence intervals**: `{metric}_lower`, `{metric}_upper`
- **Model identification**: Model names used to identify TAGT variants

## TAGT Model Detection

The script automatically identifies TAGT model variants based on keywords in the model name:
- `tagt`
- `breakthrough` 
- `ultimate`
- `production_tagt`

These models receive special highlighting with red coloring and bold black outlines.

## Usage

```bash
python plot_metrics_bars.py
```

### Requirements
- pandas
- numpy
- matplotlib
- seaborn (optional, fallback styling available)
- pathlib

### Output
The script provides detailed logging and generates a summary report showing:
- Number of models processed
- Available metrics
- Count of TAGT vs baseline models
- Files created

## Chart Specifications

### Individual Charts (12" x 8")
- Horizontal bars sorted by metric value (ascending)
- Error bars showing 95% bootstrap confidence intervals
- Model names formatted with proper capitalization
- Legend distinguishing TAGT vs baseline models
- Grid lines for easy value reading
- Clean, publication-ready styling

### Grouped Chart (20" x 12") 
- Side-by-side subplots for all metrics
- Shared legend and consistent styling
- Model names shown only on leftmost chart
- Main title spanning all subplots
- Optimized layout for comprehensive comparison

## Technical Notes

- **Error handling**: Gracefully handles missing data and confidence intervals
- **Styling fallback**: Works with or without seaborn installed
- **High resolution**: 300 DPI output for both PDF and PNG formats
- **Memory efficient**: Closes figures after saving to prevent memory issues
- **Robust data processing**: Handles NaN values and missing confidence intervals

## Example Output

The script successfully processes 9 models with 5 metrics, identifying 5 TAGT variants and 4 baseline models. All charts include bootstrap confidence intervals calculated from cross-validation results, providing statistically meaningful comparisons between models.
