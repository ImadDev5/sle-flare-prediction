# Statistical Significance Analysis Report

This report summarizes the statistical significance testing results for model comparisons.

## Model Performance Summary

### Mean ± Std (CV Folds)

| Model | AUC | Accuracy | F1 | N Folds |
|-------|-----|----------|----|---------|
| baseline_logistic_regression | 0.451 ± 0.076 | 0.755 ± 0.011 | 0.124 ± 0.123 | 5 |
| baseline_lstm | 0.495 ± 0.149 | 0.680 ± 0.062 | 0.269 ± 0.164 | 5 |
| baseline_random_forest | 0.495 ± 0.059 | 0.770 ± 0.011 | 0.000 ± 0.000 | 5 |
| baseline_svm | 0.474 ± 0.088 | 0.770 ± 0.011 | 0.000 ± 0.000 | 5 |
| tagt | 0.873 ± 0.284 | 0.940 ± 0.134 | 0.829 ± 0.383 | 5 |
| tagt_cv | 0.943 ± 0.021 | 0.891 ± 0.030 | 0.823 ± 0.059 | 5 |

## Significance Testing Results

Statistical significance tests using paired bootstrap (10,000 resamples).
P-values < 0.05 indicate statistically significant differences.

### AUC_BOOTSTRAP

- Minimum p-value: 0.000000
- Maximum p-value: 0.895200
- Significant pairs (p < 0.05): 18/30
- Highly significant pairs (p < 0.01): 14/30

**Best performing model (AUC):** tagt_cv (0.9430)

### ACC_BOOTSTRAP

- Minimum p-value: 0.000000
- Maximum p-value: 2.000000
- Significant pairs (p < 0.05): 26/30
- Highly significant pairs (p < 0.01): 18/30

### F1_BOOTSTRAP

- Minimum p-value: 0.000000
- Maximum p-value: 2.000000
- Significant pairs (p < 0.05): 24/30
- Highly significant pairs (p < 0.01): 20/30

### PREC_BOOTSTRAP

- Minimum p-value: 0.000000
- Maximum p-value: 2.000000
- Significant pairs (p < 0.05): 24/30
- Highly significant pairs (p < 0.01): 20/30

### RECALL_BOOTSTRAP

- Minimum p-value: 0.000000
- Maximum p-value: 2.000000
- Significant pairs (p < 0.05): 26/30
- Highly significant pairs (p < 0.01): 20/30

## Recommendations

Based on the statistical significance analysis:

1. **Highest AUC:** tagt_cv (AUC = 0.9430)
2. **Highest Accuracy:** tagt (Acc = 0.9400)

Different models excel in different metrics. Consider the primary objective when selecting a model.

---
*Report generated automatically by significance analysis engine.*
