# Step 3 vs Cross-Validation Significance Comparison

This report compares statistical significance results from step 3 analysis with bootstrap significance testing from cross-validation data.

## AUC Comparison

### Step 3 Significance Matrix

|                              |   baseline_logistic_regression |   baseline_lstm |   baseline_random_forest |   baseline_svm |   tagt |   tagt_cv |
|:-----------------------------|-------------------------------:|----------------:|-------------------------:|---------------:|-------:|----------:|
| baseline_logistic_regression |                         1      |          0.3014 |                   0.1292 |         0.0336 | 0.0008 |    0      |
| baseline_lstm                |                         0.3014 |          1      |                   0.8952 |         0.609  | 0      |    0      |
| baseline_random_forest       |                         0.1292 |          0.8952 |                   1      |         0.6374 | 0.0128 |    0      |
| baseline_svm                 |                         0.0336 |          0.609  |                   0.6374 |         1      | 0.0008 |    0      |
| tagt                         |                         0.0008 |          0      |                   0.0128 |         0.0008 | 1      |    0.6544 |
| tagt_cv                      |                         0      |          0      |                   0      |         0      | 0.6544 |    1      |

### Cross-Validation Bootstrap Significance Matrix

|                     |   Random_Forest |   SVM_RBF |   Logistic_Regression |   Simple_LSTM |   TAGT |
|:--------------------|----------------:|----------:|----------------------:|--------------:|-------:|
| Random_Forest       |               1 |         0 |                     0 |             0 |      0 |
| SVM_RBF             |               0 |         1 |                     0 |             0 |      0 |
| Logistic_Regression |               0 |         0 |                     1 |             0 |      0 |
| Simple_LSTM         |               0 |         0 |                     0 |             1 |      0 |
| TAGT                |               0 |         0 |                     0 |             0 |      1 |

## Accuracy Comparison

Step 3 significance matrix not found for this metric.

### Cross-Validation Bootstrap Significance Matrix

|                     |   Random_Forest |   SVM_RBF |   Logistic_Regression |   Simple_LSTM |   TAGT |
|:--------------------|----------------:|----------:|----------------------:|--------------:|-------:|
| Random_Forest       |          1      |         0 |                     0 |        0.9716 |      0 |
| SVM_RBF             |          0      |         1 |                     0 |        0      |      0 |
| Logistic_Regression |          0      |         0 |                     1 |        0      |      0 |
| Simple_LSTM         |          0.9716 |         0 |                     0 |        1      |      0 |
| TAGT                |          0      |         0 |                     0 |        0      |      1 |

---
*Report generated automatically by integrate_step3_pvalues.py*
