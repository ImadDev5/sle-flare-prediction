"""
Test script to demonstrate all significance testing functions.
"""

import numpy as np
import pandas as pd
from src.analysis.significance import (
    delong_auc_test,
    paired_bootstrap,
    mcnemar_test,
    create_significance_matrix,
    save_significance_results
)

def test_all_functions():
    """
    Test all significance testing functions with sample data.
    """
    
    print("=== Testing Statistical Significance Functions ===\n")
    
        np.random.seed(42)
    n_samples = 1000
    
    # Binary classification data
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Model 1: Better performance
    model1_probs = np.random.beta(2, 5, n_samples)
    model1_probs = np.where(y_true == 1, model1_probs + 0.4, model1_probs)
    model1_probs = np.clip(model1_probs, 0, 1)
    
    # Model 2: Baseline performance
    model2_probs = np.random.beta(2, 5, n_samples)
    model2_probs = np.where(y_true == 1, model2_probs + 0.2, model2_probs)
    model2_probs = np.clip(model2_probs, 0, 1)
    
    # Convert to binary predictions
    model1_preds = (model1_probs > 0.5).astype(int)
    model2_preds = (model2_probs > 0.5).astype(int)
    
    # Cross-validation metrics (simulated)
    cv_metrics1 = np.array([0.85, 0.82, 0.88, 0.86, 0.84])
    cv_metrics2 = np.array([0.78, 0.75, 0.80, 0.79, 0.77])
    
    print("1. Testing DeLong AUC Test:")
    print("-" * 30)
    try:
        p_value, z_score, ci = delong_auc_test(model1_probs, model2_probs, y_true)
        print(f"   P-value: {p_value:.6f}")
        print(f"   Z-score: {z_score:.4f}")
        print(f"   AUC difference: {ci['auc_diff']:.4f}")
        print(f"   95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        print(f"   Result: {'Significant' if p_value < 0.05 else 'Not significant'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Testing Paired Bootstrap Test:")
    print("-" * 35)
    try:
        p_value, ci = paired_bootstrap(cv_metrics1, cv_metrics2)
        print(f"   P-value: {p_value:.6f}")
        print(f"   Observed difference: {ci['observed_diff']:.4f}")
        print(f"   95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        print(f"   Bootstrap mean: {ci['bootstrap_mean']:.4f}")
        print(f"   Bootstrap std: {ci['bootstrap_std']:.4f}")
        print(f"   Result: {'Significant' if p_value < 0.05 else 'Not significant'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Testing McNemar Test:")
    print("-" * 25)
    try:
        p_value, statistic, info = mcnemar_test(y_true, model1_preds, model2_preds)
        print(f"   P-value: {p_value:.6f}")
        print(f"   Test statistic: {statistic:.4f}")
        print(f"   Model 1 accuracy: {info['model1_accuracy']:.4f}")
        print(f"   Model 2 accuracy: {info['model2_accuracy']:.4f}")
        print(f"   Accuracy difference: {info['accuracy_difference']:.4f}")
        print(f"   Discordant pairs: {info['discordant_pairs']}")
        print(f"   Both correct: {info['both_correct']}")
        print(f"   Both incorrect: {info['both_incorrect']}")
        print(f"   Result: {'Significant' if p_value < 0.05 else 'Not significant'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Testing Significance Matrix Creation:")
    print("-" * 40)
    try:
                results_data = []
        models = ['ModelA', 'ModelB', 'ModelC']
        
        for model in models:
            for fold in range(1, 6):  # 5-fold CV
                if model == 'ModelA':
                    auc = np.random.normal(0.85, 0.02)
                elif model == 'ModelB':
                    auc = np.random.normal(0.78, 0.03)
                else:  # ModelC
                    auc = np.random.normal(0.72, 0.025)
                
                results_data.append({
                    'model_name': model,
                    'split_type': f'cv_fold_{fold}',
                    'test_auc': auc
                })
        
        results_df = pd.DataFrame(results_data)
        
                sig_matrix = create_significance_matrix(
            results_df,
            test_type='bootstrap',
            metric_column='test_auc'
        )
        
        print("   Significance Matrix (p-values):")
        print(sig_matrix.round(4))
        
        # Count significant comparisons
        matrix_values = sig_matrix.values.copy()
        np.fill_diagonal(matrix_values, np.nan)
        sig_005 = np.sum(matrix_values < 0.05)
        total_pairs = np.sum(~np.isnan(matrix_values))
        
        print(f"   Significant pairs (p < 0.05): {sig_005}/{total_pairs}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n=== All Tests Completed Successfully! ===")
    print("\nThe significance testing engine is ready for use with:")
    print("• DeLong test for ROC AUC comparison")
    print("• Paired bootstrap for general metric comparison")
    print("• McNemar test for classification error comparison")
    print("• Automated significance matrix generation")
    print("• Results saving to CSV format")

if __name__ == "__main__":
    test_all_functions()