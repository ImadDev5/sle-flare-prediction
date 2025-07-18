
# 🚀 OPTIMIZED TAGT MODEL VALIDATION REPORT
Generated: 2025-07-18 20:51:06.593001

## Model Architecture
- **Model Type**: Optimized Temporal Attention Graph Transformer (TAGT)
- **Optimization**: RTX 3050 Memory Efficient
- **Device**: cpu

## Validation Results


### GSE49454 Dataset
**Description**: Primary training dataset - GSE49454
**Samples**: 378
**Features**: 177

#### Cross-Validation Results (Mean ± Std)
- **AUC-ROC**: 0.4811 ± 0.0472 (95% CI: 0.4213-0.5262)
- **Accuracy**: 0.4841 ± 0.0447
- **Precision**: 0.3276 ± 0.0373
- **Recall**: 0.4920 ± 0.0666
- **F1-Score**: 0.3920 ± 0.0418
- **Specificity**: 0.4800 ± 0.0704

#### Overall Performance
- **Overall AUC-ROC**: 0.4820
- **Overall Accuracy**: 0.4841
- **Overall F1-Score**: 0.3925


### GSE99967 Dataset
**Description**: External validation dataset - GSE99967 (SLE nephritis focus)
**Samples**: 59
**Features**: 24733

#### Cross-Validation Results (Mean ± Std)
- **AUC-ROC**: nan ± nan (95% CI: nan-nan)
- **Accuracy**: 0.5091 ± 0.1070
- **Precision**: 0.0000 ± 0.0000
- **Recall**: 0.0000 ± 0.0000
- **F1-Score**: 0.0000 ± 0.0000
- **Specificity**: 0.5091 ± 0.1070

#### Overall Performance
- **Overall AUC-ROC**: 0.5000
- **Overall Accuracy**: 0.5085
- **Overall F1-Score**: 0.0000


## Clinical Interpretation

### Performance Categories
- **Excellent**: AUC-ROC > 0.90 (Outstanding clinical utility)
- **Good**: AUC-ROC 0.80-0.90 (Strong clinical utility)
- **Acceptable**: AUC-ROC 0.70-0.80 (Moderate clinical utility)
- **Poor**: AUC-ROC < 0.70 (Limited clinical utility)

### Assessment
- **GSE49454**: POOR - Limited clinical utility (AUC-ROC: 0.4811)
- **GSE99967**: POOR - Limited clinical utility (AUC-ROC: nan)
