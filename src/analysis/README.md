# Model Results Analysis Module

This module provides tools for consolidating and analyzing model results from various sources across the SLE prediction project.

## Overview

The analysis module contains two main components:

1. **`collect_results.py`** - Consolidates all model result files into a unified format
2. **`utils.py`** - Provides utility functions for analyzing the consolidated results

## Features

### Results Collection (`collect_results.py`)

- **Automatic Discovery**: Finds all relevant JSON/PKL result files in the project
- **Format Detection**: Automatically detects and handles different result file formats:
  - Cross-validation results (with mean/std/values)
  - Baseline model results (multiple models in one file)
  - TAGT results (with overall metrics)
  - Optimized results (with final metrics)
  - Simple results (breakthrough, production formats)
- **Metric Harmonization**: Standardizes metric names across different sources:
  - `AUC-ROC`, `auc_roc`, `test_auc` → `auc`
  - `accuracy`, `test_accuracy` → `acc`
  - `precision`, `test_precision` → `prec`
  - `sensitivity` → `recall`
  - `specificity` → `spec`
- **MultiIndex DataFrame**: Creates a structured DataFrame with (model, split_type) MultiIndex
- **Multiple Output Formats**: Saves results as CSV (default) or Parquet (if pyarrow available)

### Analysis Utilities (`utils.py`)

- **Data Loading**: Smart loading of consolidated results (CSV/Parquet)
- **Model Comparison**: Compare models across different metrics
- **Cross-validation Analysis**: Extract and analyze CV fold results
- **Performance Ranking**: Find best performing models by metric
- **Summary Statistics**: Generate comprehensive model summaries

## Quick Start

### 1. Consolidate All Results

```bash
cd /path/to/SLE/project
python src/analysis/collect_results.py
```

This will:
- Find all result files in the project
- Extract and harmonize metrics
- Create `results/all_models_results.csv`
- Generate metadata and summary files

### 2. Analyze Results

```python
from src.analysis import load_consolidated_results, compare_models, print_results_overview

# Load consolidated results
df = load_consolidated_results()

# Print overview
print_results_overview(df)

# Compare TAGT models on AUC
tagt_models = ['tagt_cv', 'tagt', 'ultimatetagt']
comparison = compare_models(df, tagt_models, 'auc')
print(comparison)
```

## File Structure

```
src/analysis/
├── __init__.py           # Module exports
├── collect_results.py    # Main consolidation script
├── utils.py             # Analysis utilities
└── README.md           # This documentation

results/
├── all_models_results.csv          # Main consolidated results
├── all_models_results.metadata.csv # File source metadata
└── all_models_results.summary.csv  # Summary statistics
```

## Detailed Usage

### Results Collection

The `ResultsCollector` class handles the entire consolidation process:

```python
from src.analysis import ResultsCollector

# Initialize collector
collector = ResultsCollector(base_path="/path/to/project")

# Collect all results
results_df = collector.collect_all_results()

# Save results
output_path = collector.save_results(results_df)
```

### Analysis Functions

#### Load Results
```python
# Load from default location
df = load_consolidated_results()

# Load from specific path
df = load_consolidated_results("path/to/results.csv")
```

#### Model Comparison
```python
# Compare models on AUC using CV mean
comparison = compare_models(df, ['model1', 'model2'], 'auc', 'cv_mean')

# Compare on F1 score
comparison = compare_models(df, ['model1', 'model2'], 'f1')
```

#### Get Best Models
```python
# Top 5 models by AUC
best_auc = get_best_performing_models(df, metric='auc', top_n=5)

# Best models by accuracy
best_acc = get_best_performing_models(df, metric='acc')
```

#### Cross-Validation Analysis
```python
# Get CV results for a specific model
cv_results = get_cross_validation_results(df, 'tagt_cv')
print(cv_results)
```

#### Model Summary
```python
# Get detailed summary for a model
summary = get_model_summary(df, 'tagt_cv')
print(summary)
```

## DataFrame Structure

The consolidated results DataFrame has a MultiIndex structure:

```
                                     acc       auc        f1      prec    recall    spec
model                    split_type                                                     
baseline_random_forest   cv_fold_1   0.775  0.575269  0.000000  0.000000  0.000000     NaN
                         cv_fold_2   0.775  0.485663  0.000000  0.000000  0.000000     NaN
                         cv_mean     0.770  0.494581  0.000000  0.000000  0.000000     NaN
tagt_cv                  cv_fold_1   0.882  0.912308  0.800000  0.947368  0.692308     NaN
                         cv_mean     0.892  0.942997  0.822848  0.914722  0.759077     NaN
```

### Index Levels:
- **model**: Model name (e.g., 'tagt_cv', 'baseline_random_forest')
- **split_type**: Data split type (e.g., 'cv_mean', 'cv_fold_1', 'test', 'final')

### Columns (Metrics):
- **acc**: Accuracy
- **auc**: Area Under ROC Curve
- **f1**: F1 Score
- **prec**: Precision
- **recall**: Recall/Sensitivity
- **spec**: Specificity

## Supported Result File Formats

### 1. Cross-Validation Format
```json
{
  "auc": {
    "mean": 0.943,
    "std": 0.018,
    "values": [0.912, 0.955, 0.958, 0.958, 0.931]
  },
  "accuracy": {
    "mean": 0.892,
    "std": 0.027,
    "values": [0.882, 0.855, 0.934, 0.907, 0.880]
  }
}
```

### 2. Baseline Format
```json
{
  "random_forest": {
    "auc_roc": {
      "mean": 0.495,
      "std": 0.053,
      "scores": [0.575, 0.486, 0.464, 0.421, 0.527]
    }
  },
  "svm": { ... }
}
```

### 3. TAGT Format
```json
{
  "auc_roc": {
    "mean": 0.873,
    "std": 0.254,
    "scores": [0.366, 1.0, 1.0, 1.0, 1.0]
  },
  "overall": {
    "auc_roc": 0.974,
    "accuracy": 0.94,
    "precision": 0.905
  }
}
```

### 4. Optimized Format
```json
{
  "final_metrics": {
    "auc": 0.972,
    "accuracy": 0.882,
    "precision": 0.905
  },
  "best_auc": 0.972
}
```

## Statistical Analysis Ready

The consolidated DataFrame is optimized for statistical analysis:

### Paired Sample Analysis
```python
# Get CV folds for paired t-tests
model1_folds = df.loc[('model1', slice(None)), 'auc'].filter(regex='cv_fold_')
model2_folds = df.loc[('model2', slice(None)), 'auc'].filter(regex='cv_fold_')

# Perform statistical tests
from scipy.stats import ttest_rel
statistic, p_value = ttest_rel(model1_folds, model2_folds)
```

### Effect Size Calculation
```python
import numpy as np

def cohens_d(x, y):
    pooled_std = np.sqrt((x.var() + y.var()) / 2)
    return (x.mean() - y.mean()) / pooled_std

effect_size = cohens_d(model1_folds, model2_folds)
```

## Installation Requirements

### Basic Usage (CSV format)
```bash
pip install pandas numpy
```

### Parquet Support (recommended for large datasets)
```bash
pip install pandas numpy pyarrow
```

## Converting CSV to Parquet

If you want to use the more efficient Parquet format:

```python
from src.analysis import create_parquet_from_csv

# Convert existing CSV to Parquet
parquet_path = create_parquet_from_csv("results/all_models_results.csv")
```

## Troubleshooting

### Common Issues

1. **No results found**: Ensure you run `collect_results.py` first
2. **Parquet errors**: Install pyarrow or use CSV format
3. **Missing models**: Check that result files are in expected locations
4. **Metric mismatches**: Verify metric names in source files

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.analysis import ResultsCollector
collector = ResultsCollector()
results = collector.collect_all_results()
```

## Contributing

When adding new result file formats:

1. Add detection logic in `determine_extraction_method()`
2. Create extraction function following the pattern `extract_metrics_from_**()`
3. Update metric mapping in `__init__()` if needed
4. Add tests for the new format

## Output Files

After running the consolidation, you'll have:

- **`all_models_results.csv`**: Main results DataFrame
- **`all_models_results.metadata.csv`**: Source file information
- **`all_models_results.summary.csv`**: Per-model summary statistics

These files enable reproducible analysis and can be shared for collaboration.
