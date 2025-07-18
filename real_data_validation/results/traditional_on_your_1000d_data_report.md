
# Traditional Models on YOUR Actual 1000-Dimensional SLE Data

## Dataset Information
- **Source**: YOUR actual training data (sequences_real.pkl)
- **Samples**: 378 temporal sequences
- **Features**: 1000 genes (1000-dimensional)
- **SLE Flares**: 128 (33.9%)
- **Controls**: 250 (6613.8%)
- **Validation**: 5-fold cross-validation

## Performance Results

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score | Specificity |
|-------|---------|----------|-----------|--------|----------|-------------|
| **Random Forest** | 0.504±0.049 | 0.553±0.051 | 0.332±0.047 | 0.304±0.066 | 0.313±0.047 | 0.680±0.094 |
| **XGBoost** | 0.500±0.057 | 0.606±0.039 | 0.363±0.114 | 0.249±0.116 | 0.290±0.112 | 0.788±0.059 |
| **Logistic Regression** | 0.513±0.050 | 0.553±0.045 | 0.361±0.040 | 0.406±0.083 | 0.378±0.049 | 0.628±0.090 |
| **SVM** | 0.492±0.057 | 0.556±0.028 | 0.358±0.055 | 0.422±0.130 | 0.384±0.082 | 0.624±0.054 |


## Key Findings

### Best Performing Models
- **Best AUC-ROC**: Logistic Regression (0.5130 ± 0.0504)
- **Best Accuracy**: XGBoost (0.6059 ± 0.0391)

### Clinical Assessment

- **Performance Level**: LIMITED - Requires improvement for clinical use
- **Best AUC-ROC**: 0.5130
- **Clinical Readiness**: Needs optimization before clinical use

### Baseline for TAGT Comparison
These results provide the correct baseline for comparing YOUR TAGT model:
- **Target to Beat**: 0.5130 AUC-ROC (Logistic Regression)
- **Dataset**: Same 1000-dimensional data YOUR TAGT was trained on
- **Fair Comparison**: Identical preprocessing and validation methodology

## Next Steps
1. **Fix TAGT Architecture**: Resolve model loading issues
2. **Test TAGT**: Compare against these baseline results
3. **Performance Analysis**: Determine if TAGT outperforms traditional models
4. **Research Paper**: Use these results as proper baselines

---
*This represents the CORRECT baseline comparison for YOUR TAGT model validation.*
