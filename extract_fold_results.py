"""
Extract actual results from fold files
"""

import pickle
import os
import numpy as np

print("=== EXTRACTING TAGT FOLD RESULTS ===")

fold_files = [
    "results/per_fold/TAGT_fold_0.pkl",
    "results/per_fold/TAGT_fold_1.pkl", 
    "results/per_fold/TAGT_fold_2.pkl",
    "results/per_fold/TAGT_fold_3.pkl",
    "results/per_fold/TAGT_fold_4.pkl"
]

tagt_results = []

for i, fold_file in enumerate(fold_files):
    if os.path.exists(fold_file):
        try:
            with open(fold_file, 'rb') as f:
                data = pickle.load(f)
            
            metrics = data['metrics']
            print(f"\nFold {i + 1} TAGT Results:")
            print(f"  AUC-ROC: {metrics['auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            
            tagt_results.append(metrics)
            
        except Exception as e:
            print(f"  Error loading fold {i}: {e}")

if tagt_results:
    # Calculate summary statistics
    auc_values = [r['auc'] for r in tagt_results]
    acc_values = [r['accuracy'] for r in tagt_results]
    prec_values = [r['precision'] for r in tagt_results]
    recall_values = [r['recall'] for r in tagt_results]
    f1_values = [r['f1'] for r in tagt_results]
    
    print("\n" + "="*50)
    print("TAGT CROSS-VALIDATION SUMMARY:")
    print("="*50)
    print(f"AUC-ROC:   {np.mean(auc_values):.4f} ± {np.std(auc_values):.4f}")
    print(f"Accuracy:  {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}")
    print(f"Precision: {np.mean(prec_values):.4f} ± {np.std(prec_values):.4f}")
    print(f"Recall:    {np.mean(recall_values):.4f} ± {np.std(recall_values):.4f}")
    print(f"F1-Score:  {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")
    
    print(f"\nIndividual fold AUC-ROC values: {auc_values}")

# Also check traditional models
print("\n" + "="*50)
print("TRADITIONAL MODELS RESULTS:")
print("="*50)

traditional_files = [
    ("Random Forest", "results/per_fold/Random_Forest_fold_{}.pkl"),
    ("SVM RBF", "results/per_fold/SVM_RBF_fold_{}.pkl"),
    ("Logistic Regression", "results/per_fold/Logistic_Regression_fold_{}.pkl"),
    ("Simple LSTM", "results/per_fold/Simple_LSTM_fold_{}.pkl")
]

for model_name, file_pattern in traditional_files:
    model_results = []
    
    for i in range(5):
        fold_file = file_pattern.format(i)
        if os.path.exists(fold_file):
            try:
                with open(fold_file, 'rb') as f:
                    data = pickle.load(f)
                model_results.append(data['metrics'])
            except Exception as e:
                print(f"Error loading {fold_file}: {e}")
    
    if model_results:
        auc_values = [r['auc'] for r in model_results]
        acc_values = [r['accuracy'] for r in model_results]
        
        print(f"\n{model_name}:")
        print(f"  AUC-ROC: {np.mean(auc_values):.4f} ± {np.std(auc_values):.4f}")
        print(f"  Accuracy: {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}")
    else:
        print(f"\n{model_name}: No results found")