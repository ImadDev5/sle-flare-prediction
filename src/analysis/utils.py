"""
Utility functions for working with consolidated model results.

This module provides convenient functions for loading and analyzing the consolidated results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

def load_consolidated_results(results_path: str = None) -> pd.DataFrame:
    """
    Load the consolidated results DataFrame.
    
    Args:
        results_path: Path to the results file. If None, uses default location.
        
    Returns:
        Loaded DataFrame with MultiIndex (model, split_type)
    """
    if results_path is None:
        # Try parquet first, then CSV
        base_path = Path("results")
        parquet_path = base_path / "all_models_results.parquet"
        csv_path = base_path / "all_models_results.csv"
        
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path, index=[0, 1])
            except ImportError:
                warnings.warn("Parquet support not available, falling back to CSV")
                
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=[0, 1])
            return df
        else:
            raise FileNotFoundError("No consolidated results file found. Run collect_results.py first.")
    else:
        path = Path(results_path)
        if path.suffix.lower() == '.parquet':
            return pd.read_parquet(path, index=[0, 1])
        else:
            return pd.read_csv(path, index_col=[0, 1])

def get_model_summary(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Get a summary of results for a specific model.
    
    Args:
        df: Consolidated results DataFrame
        model_name: Name of the model to summarize
        
    Returns:
        Summary DataFrame for the specified model
    """
    if model_name not in df.index.get_level_values('model'):
        available_models = list(df.index.get_level_values('model').unique())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    model_data = df.loc[model_name]
    
    # Calculate summary statistics
    summary = model_data.describe()
    
    # Add additional info
    summary.loc['non_null_count'] = model_data.count()
    summary.loc['null_count'] = model_data.isnull().sum()
    
    return summary

def compare_models(df: pd.DataFrame, models: List[str], metric: str, split_type: str = 'cv_mean') -> pd.DataFrame:
    """
    Compare specific models on a given metric.
    
    Args:
        df: Consolidated results DataFrame
        models: List of model names to compare
        metric: Metric to compare (e.g., 'auc', 'acc', 'f1')
        split_type: Split type to use for comparison (default: 'cv_mean')
        
    Returns:
        Comparison DataFrame
    """
    if metric not in df.columns:
        available_metrics = list(df.columns)
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")
    
    comparison_data = []
    
    for model in models:
        if model in df.index.get_level_values('model'):
            try:
                value = df.loc[(model, split_type), metric]
                comparison_data.append({
                    'model': model,
                    'metric': metric,
                    'split_type': split_type,
                    'value': value
                })
            except KeyError:
                # Try alternative split types if the requested one doesn't exist
                available_splits = df.loc[model].index.tolist()
                if available_splits:
                    alt_split = available_splits[0]  # Use first available
                    value = df.loc[(model, alt_split), metric]
                    comparison_data.append({
                        'model': model,
                        'metric': metric,
                        'split_type': alt_split,
                        'value': value
                    })
                    warnings.warn(f"Split type '{split_type}' not available for {model}, using '{alt_split}'")
                else:
                    warnings.warn(f"No data available for model '{model}'")
        else:
            warnings.warn(f"Model '{model}' not found in data")
    
    comparison_df = pd.DataFrame(comparison_data)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('value', ascending=False)
    
    return comparison_df

def get_best_performing_models(df: pd.DataFrame, metric: str = 'auc', split_type: str = 'cv_mean', top_n: int = 5) -> pd.DataFrame:
    """
    Get the best performing models for a specific metric.
    
    Args:
        df: Consolidated results DataFrame
        metric: Metric to rank by (default: 'auc')
        split_type: Split type to use (default: 'cv_mean')
        top_n: Number of top models to return
        
    Returns:
        DataFrame with top performing models
    """
    if metric not in df.columns:
        available_metrics = list(df.columns)
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")
    
    # Get all models and their performance
    all_models = df.index.get_level_values('model').unique()
    model_performances = []
    
    for model in all_models:
        try:
            if (model, split_type) in df.index:
                value = df.loc[(model, split_type), metric]
                if pd.notna(value):
                    model_performances.append({
                        'model': model,
                        'metric': metric,
                        'split_type': split_type,
                        'value': value
                    })
        except KeyError:
            continue
    
    # If no results for the requested split type, try other split types
    if not model_performances:
        for model in all_models:
            model_data = df.loc[model]
            for available_split in model_data.index:
                value = model_data.loc[available_split, metric]
                if pd.notna(value):
                    model_performances.append({
                        'model': model,
                        'metric': metric,
                        'split_type': available_split,
                        'value': value
                    })
                    break  # Use first available split for each model
    
    if not model_performances:
        return pd.DataFrame()
    
    performance_df = pd.DataFrame(model_performances)
    performance_df = performance_df.sort_values('value', ascending=False)
    
    return performance_df.head(top_n)

def get_cross_validation_results(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Get cross-validation results for a specific model.
    
    Args:
        df: Consolidated results DataFrame
        model_name: Name of the model
        
    Returns:
        DataFrame with CV fold results
    """
    if model_name not in df.index.get_level_values('model'):
        available_models = list(df.index.get_level_values('model').unique())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    model_data = df.loc[model_name]
    
    # Filter for CV folds
    cv_splits = [split for split in model_data.index if split.startswith('cv_fold_')]
    
    if not cv_splits:
        return pd.DataFrame()
    
    cv_data = model_data.loc[cv_splits]
    
    # Add summary statistics
    if len(cv_splits) > 1:
        # Add mean and std
        cv_data.loc['cv_mean'] = cv_data.mean()
        cv_data.loc['cv_std'] = cv_data.std()
    
    return cv_data

def create_parquet_from_csv(csv_path: str = None, output_path: str = None) -> str:
    """
    Convert CSV results to Parquet format (requires pyarrow installation).
    
    Args:
        csv_path: Path to CSV file. If None, uses default location.
        output_path: Output parquet path. If None, creates alongside CSV.
        
    Returns:
        Path to created parquet file
    """
    try:
        import pyarrow
    except ImportError:
        raise ImportError("pyarrow is required for parquet support. Install with: pip install pyarrow")
    
    if csv_path is None:
        csv_path = "results/all_models_results.csv"
    
    if output_path is None:
        output_path = Path(csv_path).with_suffix('.parquet')
    
    # Load CSV and save as parquet
    df = pd.read_csv(csv_path, index_col=[0, 1])
    df.to_parquet(output_path)
    
    print(f"Converted {csv_path} to {output_path}")
    return str(output_path)

def print_results_overview(df: pd.DataFrame):
    """
    Print a comprehensive overview of the results.
    
    Args:
        df: Consolidated results DataFrame
    """
    print("=" * 80)
    print("CONSOLIDATED RESULTS OVERVIEW")
    print("=" * 80)
    
    print(f"Total models: {len(df.index.get_level_values('model').unique())}")
    print(f"Total split types: {len(df.index.get_level_values('split_type').unique())}")
    print(f"Total metrics: {len(df.columns)}")
    print(f"Total data points: {df.count().sum()}")
    print(f"DataFrame shape: {df.shape}")
    
    print("\n" + "-" * 40)
    print("MODELS:")
    print("-" * 40)
    for model in sorted(df.index.get_level_values('model').unique()):
        # Count available data points for this model
        model_data = df.loc[model]
        data_points = model_data.count().sum()
        print(f"  • {model} ({data_points} data points)")
    
    print("\n" + "-" * 40)
    print("METRICS:")
    print("-" * 40)
    for metric in df.columns:
        non_null_count = df[metric].count()
        print(f"  • {metric} ({non_null_count} values)")
    
    print("\n" + "-" * 40)
    print("SPLIT TYPES:")
    print("-" * 40)
    for split_type in sorted(df.index.get_level_values('split_type').unique()):
        count = len(df.xs(split_type, level='split_type'))
        print(f"  • {split_type} ({count} models)")
    
    print("\n" + "-" * 40)
    print("TOP PERFORMERS BY METRIC:")
    print("-" * 40)
    
    for metric in df.columns:
        try:
            best_models = get_best_performing_models(df, metric=metric, top_n=3)
            if not best_models.empty:
                print(f"\n{metric.upper()}:")
                for _, row in best_models.iterrows():
                    print(f"  1. {row['model']}: {row['value']:.4f} ({row['split_type']})")
        except:
            continue

if __name__ == "__main__":
    # Demonstration of utility functions
    try:
        df = load_consolidated_results()
        print_results_overview(df)
        
        # Example: Compare TAGT models on AUC
        print("\n" + "=" * 80)
        print("TAGT MODEL COMPARISON (AUC)")
        print("=" * 80)
        
        tagt_models = [model for model in df.index.get_level_values('model').unique() if 'tagt' in model.lower()]
        if tagt_models:
            comparison = compare_models(df, tagt_models, 'auc')
            print(comparison)
        
        # Example: Cross-validation results for a specific model
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION RESULTS EXAMPLE")
        print("=" * 80)
        
        if 'tagt_cv' in df.index.get_level_values('model'):
            cv_results = get_cross_validation_results(df, 'tagt_cv')
            print("TAGT CV Results:")
            print(cv_results)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run collect_results.py first to generate the consolidated results.")