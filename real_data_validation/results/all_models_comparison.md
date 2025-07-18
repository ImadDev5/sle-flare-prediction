
# ALL YOUR MODELS TESTED - FINDING THE 97% AUC-ROC MODEL

## Results Summary

| Model | AUC-ROC | Accuracy | Status |
|-------|---------|----------|--------|
| GAT Model (best_model.pt) | 0.0000 | 0.0000 | ERROR |
| TAGT Model (best_tagt_model.pt) | 0.4977 | 0.3386 | OK |
| TAGT Model (final_tagt_model.pt) | 0.4977 | 0.3386 | OK |
| Complex TAGT (tagt_model.pt) | 0.0000 | 0.0000 | ERROR |


## Best Performing Model
**TAGT Model (best_tagt_model.pt)** achieved the highest AUC-ROC of 0.4977

## Analysis
❌ No high-performing model found. All models perform around random chance.


## Conclusion
The highest AUC-ROC achieved was 0.4977 by TAGT Model (best_tagt_model.pt).

If you achieved 97% AUC-ROC, it might be:
1. A different model file not tested here
2. Performance on training data (not validation)
3. Different data preprocessing
4. Architecture mismatch in our recreation

Please check your training logs to confirm which model achieved 97% performance.
