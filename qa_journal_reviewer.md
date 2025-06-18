# Q&A for Journal Reviewer: TAGT Project

## Deep Methodological Questions

**Q1: What is the main novelty of TAGT?**
A: TAGT integrates graph neural networks, temporal attention, and multi-modal fusion for SLE flare prediction, which is a novel combination for this clinical problem.

**Q2: How was the model validated?**
A: TAGT was validated using cross-validation on real patient datasets, with performance compared to baselines like Random Forest, SVM, and LSTM.

**Q3: What are the main evaluation metrics?**
A: Accuracy, Precision, Recall, F1-Score, and AUC-ROC. AUC-ROC is the primary metric due to class imbalance.

**Q4: How is reproducibility ensured?**
A: All code, configurations, and data processing steps are documented. The codebase is open-source and dependencies are listed in requirements.txt.

**Q5: What are the limitations?**
A: Dependence on high-quality, multi-modal data, potential overfitting, and need for further validation in diverse populations.

**Q6: How is interpretability addressed?**
A: The model can highlight which features (genes, proteins, clinical scores) contributed most to each prediction.

## Broader Impact and Future Work

**Q7: What is the clinical impact?**
A: Early intervention, personalized medicine, and improved patient outcomes.

**Q8: What are the future directions?**
A: Validating on larger datasets, improving interpretability, adding more data types, and deploying as a clinical tool.

## Simple Explanations of Technical Terms

- **Graph Neural Network:** AI that learns from connections, like a map of proteins.
- **Temporal Attention:** AI focusing on important times in patient history.
- **Multi-modal Fusion:** Combining different types of data for better prediction.
- **AUC-ROC:** A score showing how well the model separates patients with and without flares.

## In Simple Indian English

TAGT is a new AI model that helps doctors know if SLE patients will get worse. It uses many types of data and new AI ideas. In future, it will be tested more and made easier for doctors to use. 