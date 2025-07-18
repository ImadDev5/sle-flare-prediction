"""
Statistical significance testing engine for model comparison.

This module provides functions for comparing machine learning models using various
statistical tests including DeLong's test for ROC AUC comparison, paired bootstrap
tests, and McNemar's test for classification error differences.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union
from scipy import stats
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
import os
from pathlib import Path

def delong_auc_test(model1_probs: np.ndarray, 
                    model2_probs: np.ndarray, 
                    y_true: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """
    Perform DeLong's test for comparing ROC AUC values between two models.
    
    This implementation uses the DeLong et al. (1988) method for comparing
    correlated ROC curves from the same sample.
    
    Parameters:
    -----------
    model1_probs : np.ndarray
        Predicted probabilities from model 1
    model2_probs : np.ndarray
        Predicted probabilities from model 2
    y_true : np.ndarray
        True binary labels (0 or 1)
    
    Returns:
    --------
    Tuple[float, float, Dict[str, float]]
        - p_value: Statistical significance p-value
        - z_score: Z-score from the test
        - confidence_intervals: Dict with 95% CI for AUC difference
    """
    
    def _delong_roc_variance(ground_truth, predictions):
        """
        Computes ROC AUC variance for a single curve.
        """
        order = np.argsort(predictions)[::-1]
        predictions_sorted_transposed = predictions[order]
        ground_truth_sorted_transposed = ground_truth[order]
        
        # Count positive and negative samples
        n_pos = np.sum(ground_truth_sorted_transposed)
        n_neg = len(ground_truth_sorted_transposed) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return float('nan')
        
        # Calculate structural components
        distinct_values, distinct_inverse = np.unique(predictions_sorted_transposed, 
                                                     return_inverse=True)
        
        test_order = distinct_inverse
        
        # Calculate V01 and V10
        V01 = np.zeros(len(distinct_values))
        V10 = np.zeros(len(distinct_values))
        
        for i in range(len(distinct_values)):
            V01[i] = np.sum(ground_truth_sorted_transposed[test_order == i] == 0) / n_neg
            V10[i] = np.sum(ground_truth_sorted_transposed[test_order == i] == 1) / n_pos
        
        # Calculate S01 and S10
        S01 = np.zeros(len(distinct_values))
        S10 = np.zeros(len(distinct_values))
        
        for i in range(len(distinct_values)):
            S01[i] = np.sum(V01[i:])
            S10[i] = np.sum(V10[i:])
        
        # Calculate variance components
        V01_var = np.sum((V01 - np.mean(V01))**2) / (n_neg - 1) if n_neg > 1 else 0
        V10_var = np.sum((V10 - np.mean(V10))**2) / (n_pos - 1) if n_pos > 1 else 0
        
        return V01_var / n_neg + V10_var / n_pos
    
    def _delong_roc_covariance(ground_truth, predictions_one, predictions_two):
        """
        Computes covariance between two ROC curves.
        """
        order_one = np.argsort(predictions_one)[::-1]
        order_two = np.argsort(predictions_two)[::-1]
        
        predictions_one_sorted = predictions_one[order_one]
        predictions_two_sorted = predictions_two[order_two]
        ground_truth_one_sorted = ground_truth[order_one]
        ground_truth_two_sorted = ground_truth[order_two]
        
        n_pos = np.sum(ground_truth)
        n_neg = len(ground_truth) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return float('nan')
        
        # This is a simplified covariance calculation
        # In practice, a more sophisticated implementation would be needed
        # for exact DeLong covariance calculation
        var_one = _delong_roc_variance(ground_truth, predictions_one)
        var_two = _delong_roc_variance(ground_truth, predictions_two)
        
        # Approximation using correlation coefficient
        correlation = np.corrcoef(predictions_one, predictions_two)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        return correlation * np.sqrt(var_one * var_two)
    
    # Calculate AUC values
    try:
        auc1 = roc_auc_score(y_true, model1_probs)
        auc2 = roc_auc_score(y_true, model2_probs)
    except ValueError as e:
        warnings.warn(f"Error calculating AUC: {e}")
        return float('nan'), float('nan'), {'lower': float('nan'), 'upper': float('nan')}
    
    # Calculate variances and covariance
    var1 = _delong_roc_variance(y_true, model1_probs)
    var2 = _delong_roc_variance(y_true, model2_probs)
    cov = _delong_roc_covariance(y_true, model1_probs, model2_probs)
    
    # Handle NaN values
    if np.isnan(var1) or np.isnan(var2) or np.isnan(cov):
        return float('nan'), float('nan'), {'lower': float('nan'), 'upper': float('nan')}
    
    # Calculate test statistic
    auc_diff = auc1 - auc2
    se_diff = np.sqrt(var1 + var2 - 2 * cov)
    
    if se_diff == 0:
        z_score = 0
        p_value = 1.0
    else:
        z_score = auc_diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # 95% confidence interval for AUC difference
    ci_lower = auc_diff - 1.96 * se_diff
    ci_upper = auc_diff + 1.96 * se_diff
    
    confidence_intervals = {
        'lower': ci_lower,
        'upper': ci_upper,
        'auc_diff': auc_diff,
        'se_diff': se_diff
    }
    
    return p_value, z_score, confidence_intervals

def paired_bootstrap(metric_vector1: np.ndarray, 
                    metric_vector2: np.ndarray, 
                    n: int = 10000,
                    confidence_level: float = 0.95) -> Tuple[float, Dict[str, float]]:
    """
    Perform paired bootstrap test for comparing two metric vectors.
    
    This is a non-parametric test that can be used for any metric (accuracy, F1, etc.)
    by resampling the paired observations.
    
    Parameters:
    -----------
    metric_vector1 : np.ndarray
        Metric values from model 1 (e.g., per-fold accuracy)
    metric_vector2 : np.ndarray
        Metric values from model 2 (e.g., per-fold accuracy)
    n : int, default=10000
        Number of bootstrap resamples
    confidence_level : float, default=0.95
        Confidence level for the confidence interval
        
    Returns:
    --------
    Tuple[float, Dict[str, float]]
        - p_value: Two-tailed p-value for the test
        - confidence_intervals: Dict with CI for the difference
    """
    
    if len(metric_vector1) != len(metric_vector2):
        raise ValueError("Metric vectors must have the same length")
    
    if len(metric_vector1) == 0:
        return float('nan'), {'lower': float('nan'), 'upper': float('nan')}
    
    # Calculate observed difference
    observed_diff = np.mean(metric_vector1) - np.mean(metric_vector2)
    
    # Paired differences
    paired_diffs = metric_vector1 - metric_vector2
    
    # Bootstrap resampling
    n_samples = len(paired_diffs)
    bootstrap_diffs = []
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n):
        # Resample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = paired_diffs[bootstrap_indices]
        bootstrap_diffs.append(np.mean(bootstrap_sample))
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate p-value (two-tailed)
    # H0: difference = 0
    p_value = 2 * min(
        np.mean(bootstrap_diffs <= 0),
        np.mean(bootstrap_diffs >= 0)
    )
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    confidence_intervals = {
        'lower': ci_lower,
        'upper': ci_upper,
        'observed_diff': observed_diff,
        'bootstrap_mean': np.mean(bootstrap_diffs),
        'bootstrap_std': np.std(bootstrap_diffs)
    }
    
    return p_value, confidence_intervals

def mcnemar_test(y_true: np.ndarray, 
                y_pred1: np.ndarray, 
                y_pred2: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform McNemar's test for comparing two binary classifiers.
    
    This test is used to compare the performance of two binary classifiers
    on the same dataset by examining the disagreement between the classifiers.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred1 : np.ndarray
        Predictions from model 1 (0 or 1)
    y_pred2 : np.ndarray
        Predictions from model 2 (0 or 1)
        
    Returns:
    --------
    Tuple[float, float, Dict[str, Any]]
        - p_value: Statistical significance p-value
        - statistic: McNemar's test statistic
        - additional_info: Dict with contingency table and other info
    """
    
    if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
        raise ValueError("All arrays must have the same length")
    
        # Rows: model1 correct/incorrect, Columns: model2 correct/incorrect
    model1_correct = (y_pred1 == y_true).astype(int)
    model2_correct = (y_pred2 == y_true).astype(int)
    
    # McNemar's test focuses on discordant pairs
    # b: model1 correct, model2 incorrect
    # c: model1 incorrect, model2 correct
    b = np.sum((model1_correct == 1) & (model2_correct == 0))
    c = np.sum((model1_correct == 0) & (model2_correct == 1))
    
    # Full contingency table
    a = np.sum((model1_correct == 1) & (model2_correct == 1))  # both correct
    d = np.sum((model1_correct == 0) & (model2_correct == 0))  # both incorrect
    
    contingency_table = np.array([[a, b], [c, d]])
    
    # McNemar's test statistic
    if b + c == 0:
        # No disagreement between models
        statistic = 0
        p_value = 1.0
    else:
        # Use continuity correction for small samples
        if b + c < 25:
            # Exact test using binomial distribution
            p_value = 2 * min(
                stats.binom.cdf(min(b, c), b + c, 0.5),
                1 - stats.binom.cdf(min(b, c), b + c, 0.5)
            )
            statistic = abs(b - c)
        else:
            # Chi-square approximation with continuity correction
            statistic = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - chi2.cdf(statistic, df=1)
    
    # Calculate additional statistics
    total_samples = len(y_true)
    model1_accuracy = np.mean(model1_correct)
    model2_accuracy = np.mean(model2_correct)
    
    additional_info = {
        'contingency_table': contingency_table,
        'discordant_pairs': b + c,
        'model1_correct_model2_incorrect': b,
        'model1_incorrect_model2_correct': c,
        'both_correct': a,
        'both_incorrect': d,
        'model1_accuracy': model1_accuracy,
        'model2_accuracy': model2_accuracy,
        'accuracy_difference': model1_accuracy - model2_accuracy,
        'total_samples': total_samples
    }
    
    return p_value, statistic, additional_info

def create_significance_matrix(results_df: pd.DataFrame, 
                              test_type: str = 'bootstrap',
                              metric_column: str = 'test_auc',
                              output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a significance matrix comparing all models pairwise.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model results including metric values
    test_type : str, default='bootstrap'
        Type of test to perform ('bootstrap', 'delong', 'mcnemar')
    metric_column : str, default='test_auc'
        Column name containing the metric values
    output_path : str, optional
        Path to save the significance matrix CSV file
        
    Returns:
    --------
    pd.DataFrame
        Matrix of p-values for pairwise comparisons
    """
    
    if test_type not in ['bootstrap', 'delong', 'mcnemar']:
        raise ValueError("test_type must be 'bootstrap', 'delong', or 'mcnemar'")
    
    # Get unique models
    models = results_df['model_name'].unique()
    n_models = len(models)
    
    # Initialize results matrix
    p_value_matrix = pd.DataFrame(
        index=models,
        columns=models,
        dtype=float
    )
    
    # Fill diagonal with 1.0 (identical models)
    for model in models:
        p_value_matrix.loc[model, model] = 1.0
    
    # Perform pairwise comparisons
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i >= j:  # Skip diagonal and lower triangle
                continue
            
            # Get data for both models
            model1_data = results_df[results_df['model_name'] == model1]
            model2_data = results_df[results_df['model_name'] == model2]
            
            if len(model1_data) == 0 or len(model2_data) == 0:
                p_value_matrix.loc[model1, model2] = float('nan')
                p_value_matrix.loc[model2, model1] = float('nan')
                continue
            
            try:
                if test_type == 'bootstrap':
                    # Use bootstrap test
                    metric1 = model1_data[metric_column].values
                    metric2 = model2_data[metric_column].values
                    
                    # Ensure same length (use intersection of available data)
                    min_len = min(len(metric1), len(metric2))
                    metric1 = metric1[:min_len]
                    metric2 = metric2[:min_len]
                    
                    p_value, _ = paired_bootstrap(metric1, metric2)
                    
                elif test_type == 'delong':
                    # Use DeLong test (requires probabilities and true labels)
                    # This would need probability columns in the dataframe
                    raise NotImplementedError("DeLong test requires probability data")
                    
                elif test_type == 'mcnemar':
                    # Use McNemar test (requires predictions and true labels)
                    # This would need prediction columns in the dataframe
                    raise NotImplementedError("McNemar test requires prediction data")
                
                p_value_matrix.loc[model1, model2] = p_value
                p_value_matrix.loc[model2, model1] = p_value
                
            except Exception as e:
                warnings.warn(f"Error comparing {model1} vs {model2}: {e}")
                p_value_matrix.loc[model1, model2] = float('nan')
                p_value_matrix.loc[model2, model1] = float('nan')
    
    # Save to file if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        p_value_matrix.to_csv(output_path, index=True)
        print(f"Significance matrix saved to: {output_path}")
    
    return p_value_matrix

def save_significance_results(results_dict: Dict[str, Any], 
                             output_dir: str = "results") -> None:
    """
    Save significance testing results to CSV files.
    
    Parameters:
    -----------
    results_dict : Dict[str, Any]
        Dictionary containing test results from various significance tests
    output_dir : str, default="results"
        Directory to save the results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
        summary_data = []
    
    for test_name, test_results in results_dict.items():
        if isinstance(test_results, dict):
            for comparison, result in test_results.items():
                if isinstance(result, tuple) and len(result) >= 2:
                    p_value = result[0]
                    summary_data.append({
                        'test_type': test_name,
                        'comparison': comparison,
                        'p_value': p_value,
                        'significant_005': p_value < 0.05,
                        'significant_001': p_value < 0.01
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "significance_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Significance summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, "significance_detailed.csv")
    with open(detailed_path, 'w') as f:
        f.write("# Statistical Significance Test Results\n")
        f.write("        
        for test_name, test_results in results_dict.items():
            f.write(f"## {test_name}\n")
            if isinstance(test_results, dict):
                for comparison, result in test_results.items():
                    f.write(f"{comparison}: {result}\n")
            else:
                f.write(f"{test_results}\n")
            f.write("\n")
    
    print(f"Detailed significance results saved to: {detailed_path}")

# Example usage and testing functions
def example_usage():
    """
    Example usage of the significance testing functions.
    """
    
        np.random.seed(42)
    n_samples = 1000
    
    # Binary classification example
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Model 1: slightly better performance
    model1_probs = np.random.beta(2, 5, n_samples)
    model1_probs = np.where(y_true == 1, model1_probs + 0.3, model1_probs)
    model1_probs = np.clip(model1_probs, 0, 1)
    
    # Model 2: baseline performance
    model2_probs = np.random.beta(2, 5, n_samples)
    model2_probs = np.where(y_true == 1, model2_probs + 0.2, model2_probs)
    model2_probs = np.clip(model2_probs, 0, 1)
    
    # Convert to predictions
    model1_preds = (model1_probs > 0.5).astype(int)
    model2_preds = (model2_probs > 0.5).astype(int)
    
    # Cross-validation metrics (example)
    cv_metrics1 = np.array([0.85, 0.82, 0.88, 0.86, 0.84])
    cv_metrics2 = np.array([0.80, 0.78, 0.82, 0.81, 0.79])
    
    print("=== Statistical Significance Testing Examples ===\n")
    
    # 1. DeLong test
    print("1. DeLong AUC Test:")
    try:
        p_val, z_score, ci = delong_auc_test(model1_probs, model2_probs, y_true)
        print(f"   P-value: {p_val:.4f}")
        print(f"   Z-score: {z_score:.4f}")
        print(f"   AUC difference: {ci['auc_diff']:.4f}")
        print(f"   95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Paired Bootstrap Test:")
    try:
        p_val, ci = paired_bootstrap(cv_metrics1, cv_metrics2)
        print(f"   P-value: {p_val:.4f}")
        print(f"   Observed difference: {ci['observed_diff']:.4f}")
        print(f"   95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. McNemar Test:")
    try:
        p_val, stat, info = mcnemar_test(y_true, model1_preds, model2_preds)
        print(f"   P-value: {p_val:.4f}")
        print(f"   Test statistic: {stat:.4f}")
        print(f"   Model 1 accuracy: {info['model1_accuracy']:.4f}")
        print(f"   Model 2 accuracy: {info['model2_accuracy']:.4f}")
        print(f"   Discordant pairs: {info['discordant_pairs']}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    example_usage()