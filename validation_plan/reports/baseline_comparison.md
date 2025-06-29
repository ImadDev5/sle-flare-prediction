
# BASELINE MODEL COMPARISON REPORT
Generated: 2025-06-21 02:18:42.042242

## Performance Summary (Mean ± Std)

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Random Forest | 0.495±0.053 | 0.770±0.010 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 |
| Svm | 0.475±0.079 | 0.770±0.010 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 |
| Logistic Regression | 0.451±0.068 | 0.755±0.010 | 0.233±0.200 | 0.084±0.076 | 0.124±0.110 |
| Lstm | 0.495±0.134 | 0.680±0.056 | 0.281±0.154 | 0.258±0.141 | 0.269±0.147 |


## Detailed Results


### Random Forest
- **AUC-ROC**: 0.4946 ± 0.0528
- **Accuracy**: 0.7700 ± 0.0100
- **Precision**: 0.0000 ± 0.0000
- **Recall**: 0.0000 ± 0.0000
- **F1-Score**: 0.0000 ± 0.0000


### Svm
- **AUC-ROC**: 0.4745 ± 0.0790
- **Accuracy**: 0.7700 ± 0.0100
- **Precision**: 0.0000 ± 0.0000
- **Recall**: 0.0000 ± 0.0000
- **F1-Score**: 0.0000 ± 0.0000


### Logistic Regression
- **AUC-ROC**: 0.4513 ± 0.0676
- **Accuracy**: 0.7550 ± 0.0100
- **Precision**: 0.2333 ± 0.2000
- **Recall**: 0.0844 ± 0.0762
- **F1-Score**: 0.1238 ± 0.1100


### Lstm
- **AUC-ROC**: 0.4948 ± 0.1336
- **Accuracy**: 0.6800 ± 0.0557
- **Precision**: 0.2806 ± 0.1538
- **Recall**: 0.2578 ± 0.1410
- **F1-Score**: 0.2685 ± 0.1468


## Comparison with Documented Claims

**Documented Claims:**
- Random Forest: 64.8% AUC-ROC
- SVM: 51.9% AUC-ROC  
- LSTM: 40.7% AUC-ROC
- TAGT: 96.3% AUC-ROC

**Actual Results:**
- Random Forest: 0.495 (Claimed: 0.648, Difference: -0.153)
- Svm: 0.475 (Claimed: 0.519, Difference: -0.044)
- Lstm: 0.495 (Claimed: 0.407, Difference: +0.088)
