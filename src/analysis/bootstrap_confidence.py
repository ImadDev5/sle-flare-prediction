"""
Bootstrap confidence interval calculation module for model performance metrics.

This module provides functions to calculate 95% confidence intervals using bootstrap
resampling of test-set predictions. For each metric per model, it draws 1000 bootstrap 
resamples and computes mean & confidence intervals that can be used as error bars 
in bar plots.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from pathlib import Path
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BootstrapConfidenceCalculator:
    """
    Calculator for bootstrap confidence intervals on model performance metrics.
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95, 
                 random_state: int = 42):
        """
        Initialize the bootstrap confidence calculator.
        
        Args:
            n_bootstrap: Number of bootstrap resamples (default: 1000)
            confidence_level: Confidence level for intervals (default: 0.95)
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_resample(self, data: np.ndarray) -> np.ndarray:
        """
        Perform bootstrap resampling on a 1D array.
        
        Args:
            data: Input data array
            
        Returns:
            Bootstrap resampled array of same length
        """
        n_samples = len(data)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return data[indices]
    
    def calculate_bootstrap_ci(self, data: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence intervals for a metric.
        
        Args:
            data: Array of metric values (e.g., per-fold AUC scores)
            
        Returns:
            Tuple of (mean, lower_ci, upper_ci)
        """
        if len(data) == 0:
            return np.nan, np.nan, np.nan
        
        # Remove NaN values
        data_clean = data[~np.isnan(data)]
        if len(data_clean) == 0:
            return np.nan, np.nan, np.nan
        
        # Calculate original mean
        original_mean = np.mean(data_clean)
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = self.bootstrap_resample(data_clean)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower_ci = np.percentile(bootstrap_means, lower_percentile)
        upper_ci = np.percentile(bootstrap_means, upper_percentile)
        
        return original_mean, lower_ci, upper_ci
    
    def process_cross_validation_folds(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process cross-validation fold results to calculate bootstrap CIs.
        
        Args:
            results_df: DataFrame with MultiIndex (model, split_type) and metric columns
            
        Returns:
            DataFrame with added confidence interval columns
        """
        logger.info("Processing cross-validation folds for bootstrap confidence intervals...")
        
                df_with_ci = results_df.copy()
        
        # Get all metrics
        metrics = [col for col in results_df.columns if col in ['acc', 'auc', 'f1', 'prec', 'recall', 'spec']]
        
        # For each model, collect fold data and calculate CIs
        models = results_df.index.get_level_values('model').unique()
        
        for model in models:
            logger.info(f"Processing model: {model}")
            
            try:
                model_data = results_df.loc[model]
                
                # Collect fold data for each metric
                for metric in metrics:
                    # Get per-fold values
                    fold_pattern = 'cv_fold_'
                    fold_indices = [idx for idx in model_data.index if fold_pattern in str(idx)]
                    
                    if not fold_indices:
                        # Look for other patterns indicating individual results
                        fold_indices = [idx for idx in model_data.index 
                                      if any(pattern in str(idx).lower() 
                                           for pattern in ['fold_', 'test', 'validation'])]
                    
                    if fold_indices:
                        # Extract values for this metric
                        fold_values = []
                        for fold_idx in fold_indices:
                            value = model_data.loc[fold_idx, metric]
                            if not pd.isna(value):
                                fold_values.append(value)
                        
                        if fold_values:
                            fold_values = np.array(fold_values)
                            mean_val, lower_ci, upper_ci = self.calculate_bootstrap_ci(fold_values)
                            
                            # Add CI columns to all rows for this model
                            model_indices = df_with_ci.index.get_level_values('model') == model
                            
                                                        lower_col = f"{metric}_lower"
                            upper_col = f"{metric}_upper"
                            
                            # Add columns if they don't exist
                            if lower_col not in df_with_ci.columns:
                                df_with_ci[lower_col] = np.nan
                            if upper_col not in df_with_ci.columns:
                                df_with_ci[upper_col] = np.nan
                            
                            # Set CI values for all splits of this model
                            df_with_ci.loc[model_indices, lower_col] = lower_ci
                            df_with_ci.loc[model_indices, upper_col] = upper_ci
                            
                            logger.info(f"  {metric}: mean={mean_val:.4f}, "
                                      f"CI=[{lower_ci:.4f}, {upper_ci:.4f}]")
                        else:
                            logger.warning(f"  No valid fold values found for {metric}")
                    else:
                        logger.warning(f"  No fold data found for model {model}")
                        
            except Exception as e:
                logger.error(f"Error processing model {model}: {e}")
                continue
        
        return df_with_ci
    
    def process_simple_metrics(self, results_df: pd.DataFrame, 
                              per_fold_dir: str = None) -> pd.DataFrame:
        """
        Process metrics from per-fold result files for models without aggregated CV data.
        
        Args:
            results_df: Main results DataFrame
            per_fold_dir: Directory containing per-fold result files
            
        Returns:
            DataFrame with bootstrap confidence intervals added
        """
        logger.info("Processing per-fold files for bootstrap confidence intervals...")
        
        df_with_ci = results_df.copy()
        
        if per_fold_dir is None:
            # Try to find per-fold directory
            base_path = Path("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SLE")
            per_fold_dir = base_path / "results" / "per_fold"
        else:
            per_fold_dir = Path(per_fold_dir)
        
        if not per_fold_dir.exists():
            logger.warning(f"Per-fold directory not found: {per_fold_dir}")
            return df_with_ci
        
        # Look for per-fold files
        fold_files = list(per_fold_dir.glob("*_fold_*.pkl"))
        
        # Group files by model
        model_fold_data = {}
        for file_path in fold_files:
            # Extract model name and fold number
            file_name = file_path.stem
            parts = file_name.split('_fold_')
            if len(parts) == 2:
                model_name = parts[0].lower().replace('_', ' ')
                fold_num = parts[1]
                
                if model_name not in model_fold_data:
                    model_fold_data[model_name] = []
                
                try:
                    # Load fold data
                    import pickle
                    with open(file_path, 'rb') as f:
                        fold_data = pickle.load(f)
                    
                    model_fold_data[model_name].append({
                        'fold': fold_num,
                        'data': fold_data,
                        'file': file_path
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        # Process each model's fold data
        for model_name, fold_list in model_fold_data.items():
            logger.info(f"Processing per-fold data for: {model_name}")
            
            # Extract metrics from fold files
            metrics_data = {}
            
            for fold_info in fold_list:
                fold_data = fold_info['data']
                
                # Extract metrics from fold data
                if isinstance(fold_data, dict):
                    for metric_name, value in fold_data.items():
                        if metric_name in ['acc', 'accuracy', 'auc', 'f1', 'precision', 'prec', 
                                         'recall', 'sensitivity', 'specificity', 'spec']:
                            # Normalize metric name
                            norm_name = self._normalize_metric_name(metric_name)
                            
                            if norm_name not in metrics_data:
                                metrics_data[norm_name] = []
                            
                            if not pd.isna(value):
                                metrics_data[norm_name].append(float(value))
            
            # Calculate bootstrap CIs for each metric
            for metric, values in metrics_data.items():
                if len(values) >= 2:  # Need at least 2 values for CI
                    values_array = np.array(values)
                    mean_val, lower_ci, upper_ci = self.calculate_bootstrap_ci(values_array)
                    
                    # Find corresponding rows in DataFrame
                    model_matches = []
                    for idx in df_with_ci.index.get_level_values('model').unique():
                        if self._model_name_match(model_name, idx):
                            model_matches.append(idx)
                    
                    # Update DataFrame
                    for matched_model in model_matches:
                        model_indices = df_with_ci.index.get_level_values('model') == matched_model
                        
                        lower_col = f"{metric}_lower"
                        upper_col = f"{metric}_upper"
                        
                        if lower_col not in df_with_ci.columns:
                            df_with_ci[lower_col] = np.nan
                        if upper_col not in df_with_ci.columns:
                            df_with_ci[upper_col] = np.nan
                        
                        df_with_ci.loc[model_indices, lower_col] = lower_ci
                        df_with_ci.loc[model_indices, upper_col] = upper_ci
                        
                        logger.info(f"  {matched_model} - {metric}: "
                                  f"mean={mean_val:.4f}, CI=[{lower_ci:.4f}, {upper_ci:.4f}]")
        
        return df_with_ci
    
    def _normalize_metric_name(self, metric_name: str) -> str:
        """Normalize metric names to standard format."""
        name_map = {
            'accuracy': 'acc',
            'precision': 'prec',
            'sensitivity': 'recall',
            'specificity': 'spec'
        }
        return name_map.get(metric_name.lower(), metric_name.lower())
    
    def _model_name_match(self, file_model: str, df_model: str) -> bool:
        """Check if model names match allowing for variations."""
        file_model = file_model.lower().replace('_', ' ').replace('-', ' ')
        df_model = df_model.lower().replace('_', ' ').replace('-', ' ')
        
        # Direct match
        if file_model == df_model:
            return True
        
        # Partial match
        if file_model in df_model or df_model in file_model:
            return True
        
        # Special cases
        if 'lstm' in file_model and 'lstm' in df_model:
            return True
        if 'logistic' in file_model and 'logistic' in df_model:
            return True
        if 'forest' in file_model and 'forest' in df_model:
            return True
        if 'svm' in file_model and 'svm' in df_model:
            return True
        if 'tagt' in file_model and 'tagt' in df_model:
            return True
        
        return False
    
    def add_bootstrap_confidence_intervals(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to add bootstrap confidence intervals to the results DataFrame.
        
        Args:
            results_df: Results DataFrame with model performance metrics
            
        Returns:
            Enhanced DataFrame with confidence interval columns added
        """
        logger.info("Adding bootstrap confidence intervals to results DataFrame...")
        
        # First, process CV fold data that's already in the DataFrame
        df_with_ci = self.process_cross_validation_folds(results_df)
        
        # Then, process additional per-fold data from files
        df_with_ci = self.process_simple_metrics(df_with_ci)
        
                ci_columns = [col for col in df_with_ci.columns if col.endswith('_lower') or col.endswith('_upper')]
        logger.info(f"Added {len(ci_columns)} confidence interval columns: {ci_columns}")
        
        return df_with_ci
    
    def create_summary_with_cis(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary DataFrame with means and confidence intervals for plotting.
        
        Args:
            results_df: DataFrame with confidence intervals
            
        Returns:
            Summary DataFrame suitable for plotting with error bars
        """
        logger.info("Creating summary DataFrame with confidence intervals...")
        
        # Get metrics and models
        base_metrics = ['acc', 'auc', 'f1', 'prec', 'recall', 'spec']
        models = results_df.index.get_level_values('model').unique()
        
        summary_data = []
        
        for model in models:
            try:
                model_data = results_df.loc[model]
                
                # Get the mean/best performance for this model
                # Prefer cv_mean, then final, then best, then overall
                preferred_splits = ['cv_mean', 'final', 'best', 'overall', 'test']
                selected_split = None
                
                for split in preferred_splits:
                    if split in model_data.index:
                        selected_split = split
                        break
                
                if selected_split is None:
                    # Use the first available split
                    selected_split = model_data.index[0]
                
                row_data = {'model': model, 'split_type': selected_split}
                
                # Add metrics and their CIs
                for metric in base_metrics:
                    if metric in model_data.columns:
                        # Get the metric value
                        metric_value = model_data.loc[selected_split, metric]
                        row_data[metric] = metric_value
                        
                        # Get CI values (these should be the same across all splits for a model)
                        lower_col = f"{metric}_lower"
                        upper_col = f"{metric}_upper"
                        
                        if lower_col in model_data.columns:
                            row_data[lower_col] = model_data.loc[selected_split, lower_col]
                        if upper_col in model_data.columns:
                            row_data[upper_col] = model_data.loc[selected_split, upper_col]
                
                summary_data.append(row_data)
                
            except Exception as e:
                logger.warning(f"Error processing model {model} for summary: {e}")
                continue
        
        summary_df = pd.DataFrame(summary_data)
        
        # Set model as index for easier plotting
        if not summary_df.empty:
            summary_df.set_index('model', inplace=True)
        
        logger.info(f"Created summary DataFrame with shape {summary_df.shape}")
        return summary_df

def main():
    """
    Main function to demonstrate bootstrap confidence interval calculation.
    """
    from src.analysis.collect_results import ResultsCollector
    
    # Initialize components
    logger.info("Initializing bootstrap confidence interval calculation...")
    
    collector = ResultsCollector(base_path="C:\\Users\\ADMIN\\OneDrive\\Desktop\\SLE")
    bootstrap_calc = BootstrapConfidenceCalculator(n_bootstrap=1000, confidence_level=0.95)
    
    # Collect results
    logger.info("Collecting all model results...")
    results_df = collector.collect_all_results()
    
    if results_df.empty:
        logger.error("No results found. Cannot calculate confidence intervals.")
        return
    
    logger.info(f"Original results shape: {results_df.shape}")
    logger.info(f"Available metrics: {list(results_df.columns)}")
    
    # Add bootstrap confidence intervals
    results_with_ci = bootstrap_calc.add_bootstrap_confidence_intervals(results_df)
    
    logger.info(f"Results with CI shape: {results_with_ci.shape}")
    logger.info(f"Available columns: {list(results_with_ci.columns)}")
    
        summary_df = bootstrap_calc.create_summary_with_cis(results_with_ci)
    
    # Save results
    base_path = Path("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SLE")
    results_dir = base_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save full results with CIs
    full_results_path = results_dir / "all_results_with_bootstrap_ci.csv"
    results_with_ci.to_csv(full_results_path)
    logger.info(f"Full results with bootstrap CIs saved to: {full_results_path}")
    
    # Save summary for plotting
    summary_path = results_dir / "results_summary_with_ci.csv"
    summary_df.to_csv(summary_path)
    logger.info(f"Summary results for plotting saved to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVAL SUMMARY")
    print("="*80)
    print(f"Number of bootstrap resamples: {bootstrap_calc.n_bootstrap}")
    print(f"Confidence level: {bootstrap_calc.confidence_level*100}%")
    print(f"Total models processed: {len(results_with_ci.index.get_level_values('model').unique())}")
    
    # Show available CI columns
    ci_columns = [col for col in results_with_ci.columns if '_lower' in col or '_upper' in col]
    print(f"Confidence interval columns added: {len(ci_columns)}")
    for col in sorted(ci_columns):
        print(f"  • {col}")
    
    print("\nSample of results with confidence intervals:")
    print(summary_df.head())
    
    return results_with_ci, summary_df

if __name__ == "__main__":
    main()