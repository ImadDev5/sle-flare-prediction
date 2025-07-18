"""
Module for collecting and consolidating all model result files into a unified results repository.

This module loads every relevant JSON/PKL file (optimized_results.json, cross_validation_results.json, 
baseline_results.json, tagt_results.json, etc.), harmonizes metric names, and builds a single 
Pandas DataFrame with a MultiIndex for statistical analysis.
"""

import json
import pickle
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsCollector:
    """
    Collector class for consolidating model results from various sources.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the results collector.
        
        Args:
            base_path: Base path to search for result files. If None, uses current directory.
        """
        self.base_path = Path(base_path) if base_path else Path(".")
        self.metric_mapping = {
            # Harmonize metric names
            'auc_roc': 'auc',
            'auc-roc': 'auc',
            'AUC-ROC': 'auc',
            'test_auc': 'auc',
            'best_val_auc': 'auc',
            'accuracy': 'acc',
            'test_accuracy': 'acc',
            'precision': 'prec',
            'test_precision': 'prec',
            'recall': 'recall',
            'test_recall': 'recall',
            'sensitivity': 'recall',  # sensitivity = recall
            'f1': 'f1',
            'test_f1': 'f1',
            'f1_score': 'f1',
            'specificity': 'spec'
        }
        self.results_data = []
        
    def find_result_files(self) -> Dict[str, List[Path]]:
        """
        Find all relevant result files in the directory structure.
        
        Returns:
            Dictionary mapping file types to lists of file paths
        """
        file_patterns = {
            'json': ['*.json'],
            'pkl': ['*.pkl', '*.pickle']
        }
        
        found_files = {'json': [], 'pkl': []}
        
        # Search for JSON files
        for pattern in file_patterns['json']:
            found_files['json'].extend(self.base_path.rglob(pattern))
            
        # Search for PKL files  
        for pattern in file_patterns['pkl']:
            found_files['pkl'].extend(self.base_path.rglob(pattern))
            
        # Filter for result files (exclude config files and data files)
        result_keywords = ['result', 'metric', 'evaluation', 'performance', 'validation', 'baseline', 'tagt', 'optimized', 'breakthrough', 'production']
        exclude_keywords = ['config', 'metadata', 'ppi_stats', 'model_metadata']
        
        filtered_files = {'json': [], 'pkl': []}
        
        for file_type in found_files:
            for file_path in found_files[file_type]:
                file_name_lower = file_path.name.lower()
                
                # Skip if contains exclude keywords
                if any(keyword in file_name_lower for keyword in exclude_keywords):
                    continue
                    
                # Include if contains result keywords or is in results/validation directories
                if (any(keyword in file_name_lower for keyword in result_keywords) or
                    'results' in str(file_path).lower() or 
                    'validation' in str(file_path).lower()):
                    filtered_files[file_type].append(file_path)
                    
        return filtered_files
    
    def load_json_file(self, file_path: Path) -> Optional[Dict]:
        """
        Load and parse a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON file: {file_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load JSON file {file_path}: {e}")
            return None
    
    def load_pkl_file(self, file_path: Path) -> Optional[Any]:
        """
        Load and parse a pickle file.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            Unpickled data or None if loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Successfully loaded PKL file: {file_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load PKL file {file_path}: {e}")
            return None
    
    def harmonize_metric_name(self, metric_name: str) -> str:
        """
        Harmonize metric names to standard format.
        
        Args:
            metric_name: Original metric name
            
        Returns:
            Standardized metric name
        """
        return self.metric_mapping.get(metric_name.lower(), metric_name.lower())
    
    def extract_metrics_from_cross_validation(self, data: Dict, source_info: Dict) -> List[Dict]:
        """
        Extract metrics from cross-validation results format.
        
        Args:
            data: Cross-validation results data
            source_info: Information about the source file
            
        Returns:
            List of metric records
        """
        records = []
        
        for metric_name, metric_data in data.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                harmonized_name = self.harmonize_metric_name(metric_name)
                
                # Add mean record
                records.append({
                    'model': source_info['model'],
                    'split_type': 'cv_mean',
                    'metric': harmonized_name,
                    'value': metric_data['mean'],
                    'std': metric_data.get('std', np.nan),
                    'source_file': str(source_info['file_path']),
                    'data_type': 'aggregated'
                })
                
                # Add individual fold records if available
                if 'values' in metric_data and isinstance(metric_data['values'], list):
                    for i, value in enumerate(metric_data['values']):
                        records.append({
                            'model': source_info['model'],
                            'split_type': f'cv_fold_{i+1}',
                            'metric': harmonized_name,
                            'value': value,
                            'std': np.nan,
                            'source_file': str(source_info['file_path']),
                            'data_type': 'individual'
                        })
        
        return records
    
    def extract_metrics_from_baseline(self, data: Dict, source_info: Dict) -> List[Dict]:
        """
        Extract metrics from baseline results format.
        
        Args:
            data: Baseline results data
            source_info: Information about the source file
            
        Returns:
            List of metric records
        """
        records = []
        
        for model_name, model_data in data.items():
            if isinstance(model_data, dict):
                for metric_name, metric_data in model_data.items():
                    if isinstance(metric_data, dict) and 'mean' in metric_data:
                        harmonized_name = self.harmonize_metric_name(metric_name)
                        
                        # Add mean record
                        records.append({
                            'model': f"baseline_{model_name}",
                            'split_type': 'cv_mean',
                            'metric': harmonized_name,
                            'value': metric_data['mean'],
                            'std': metric_data.get('std', np.nan),
                            'source_file': str(source_info['file_path']),
                            'data_type': 'aggregated'
                        })
                        
                        # Add individual fold records if available
                        if 'scores' in metric_data and isinstance(metric_data['scores'], list):
                            for i, value in enumerate(metric_data['scores']):
                                records.append({
                                    'model': f"baseline_{model_name}",
                                    'split_type': f'cv_fold_{i+1}',
                                    'metric': harmonized_name,
                                    'value': value,
                                    'std': np.nan,
                                    'source_file': str(source_info['file_path']),
                                    'data_type': 'individual'
                                })
        
        return records
    
    def extract_metrics_from_tagt(self, data: Dict, source_info: Dict) -> List[Dict]:
        """
        Extract metrics from TAGT results format.
        
        Args:
            data: TAGT results data
            source_info: Information about the source file
            
        Returns:
            List of metric records
        """
        records = []
        
        # Handle individual metrics
        for metric_name, metric_data in data.items():
            if metric_name == 'overall':
                # Handle overall metrics
                for overall_metric, value in metric_data.items():
                    harmonized_name = self.harmonize_metric_name(overall_metric)
                    records.append({
                        'model': source_info['model'],
                        'split_type': 'overall',
                        'metric': harmonized_name,
                        'value': value,
                        'std': np.nan,
                        'source_file': str(source_info['file_path']),
                        'data_type': 'aggregated'
                    })
            elif metric_name == 'predictions':
                # Skip predictions data
                continue
            elif isinstance(metric_data, dict) and 'mean' in metric_data:
                harmonized_name = self.harmonize_metric_name(metric_name)
                
                # Add mean record
                records.append({
                    'model': source_info['model'],
                    'split_type': 'cv_mean',
                    'metric': harmonized_name,
                    'value': metric_data['mean'],
                    'std': metric_data.get('std', np.nan),
                    'source_file': str(source_info['file_path']),
                    'data_type': 'aggregated'
                })
                
                # Add individual fold records if available
                if 'scores' in metric_data and isinstance(metric_data['scores'], list):
                    for i, value in enumerate(metric_data['scores']):
                        records.append({
                            'model': source_info['model'],
                            'split_type': f'cv_fold_{i+1}',
                            'metric': harmonized_name,
                            'value': value,
                            'std': np.nan,
                            'source_file': str(source_info['file_path']),
                            'data_type': 'individual'
                        })
        
        return records
    
    def extract_metrics_from_optimized(self, data: Dict, source_info: Dict) -> List[Dict]:
        """
        Extract metrics from optimized results format.
        
        Args:
            data: Optimized results data
            source_info: Information about the source file
            
        Returns:
            List of metric records
        """
        records = []
        
        # Handle final metrics
        if 'final_metrics' in data:
            for metric_name, value in data['final_metrics'].items():
                harmonized_name = self.harmonize_metric_name(metric_name)
                records.append({
                    'model': source_info['model'],
                    'split_type': 'final',
                    'metric': harmonized_name,
                    'value': value,
                    'std': np.nan,
                    'source_file': str(source_info['file_path']),
                    'data_type': 'final'
                })
        
        # Handle best_auc separately if present
        if 'best_auc' in data:
            records.append({
                'model': source_info['model'],
                'split_type': 'best',
                'metric': 'auc',
                'value': data['best_auc'],
                'std': np.nan,
                'source_file': str(source_info['file_path']),
                'data_type': 'best'
            })
        
        return records
    
    def extract_metrics_from_simple(self, data: Dict, source_info: Dict) -> List[Dict]:
        """
        Extract metrics from simple results format (breakthrough, production).
        
        Args:
            data: Simple results data
            source_info: Information about the source file
            
        Returns:
            List of metric records
        """
        records = []
        
        # Handle final_metrics if present
        if 'final_metrics' in data:
            for metric_name, value in data['final_metrics'].items():
                # Skip NaN values
                if pd.isna(value):
                    continue
                harmonized_name = self.harmonize_metric_name(metric_name)
                records.append({
                    'model': source_info['model'],
                    'split_type': 'final',
                    'metric': harmonized_name,
                    'value': value,
                    'std': np.nan,
                    'source_file': str(source_info['file_path']),
                    'data_type': 'final'
                })
        
        # Handle direct metrics (breakthrough format)
        metric_fields = ['test_auc', 'test_f1', 'test_accuracy', 'best_val_auc']
        for field in metric_fields:
            if field in data:
                harmonized_name = self.harmonize_metric_name(field)
                records.append({
                    'model': source_info['model'],
                    'split_type': 'test' if field.startswith('test_') else 'validation',
                    'metric': harmonized_name,
                    'value': data[field],
                    'std': np.nan,
                    'source_file': str(source_info['file_path']),
                    'data_type': 'final'
                })
        
        return records
    
    def determine_model_name(self, file_path: Path, data: Dict) -> str:
        """
        Determine the model name based on file path and content.
        
        Args:
            file_path: Path to the result file
            data: Loaded data from the file
            
        Returns:
            Model name string
        """
        file_name = file_path.stem.lower()
        
        # Check for explicit model type in data
        if isinstance(data, dict):
            if 'model_type' in data:
                return data['model_type'].replace(' ', '_').lower()
            if 'config' in data and isinstance(data['config'], dict):
                if 'model_name' in data['config']:
                    return data['config']['model_name'].lower()
        
        # Determine from filename
        if 'cross_validation' in file_name:
            return 'tagt_cv'
        elif 'optimized' in file_name:
            return 'optimized_tagt'
        elif 'baseline' in file_name:
            return 'baselines'
        elif 'tagt' in file_name and 'baseline' not in file_name:
            return 'tagt'
        elif 'breakthrough' in file_name:
            return 'breakthrough_tagt'
        elif 'production' in file_name:
            return 'production_tagt'
        elif 'validation' in file_name:
            return 'validation'
        else:
            return file_path.stem.lower().replace('_results', '').replace('results', '')
    
    def determine_extraction_method(self, data: Dict, file_path: Path) -> str:
        """
        Determine which extraction method to use based on data structure.
        
        Args:
            data: Loaded data
            file_path: Path to the file
            
        Returns:
            Method name to use for extraction
        """
        if not isinstance(data, dict):
            return 'simple'
        
        # Check for cross-validation format (metrics with mean/std/values)
        has_cv_format = any(
            isinstance(v, dict) and 'mean' in v and 'values' in v 
            for v in data.values() if isinstance(v, dict)
        )
        
        # Check for baseline format (nested model structure)
        has_baseline_format = any(
            k in ['random_forest', 'svm', 'logistic_regression', 'lstm'] 
            for k in data.keys()
        )
        
        # Check for TAGT format (has overall section or claimed metrics)
        has_tagt_format = 'overall' in data or any(
            isinstance(v, dict) and 'claimed' in v 
            for v in data.values() if isinstance(v, dict)
        )
        
        # Check for optimized format (has final_metrics)
        has_optimized_format = 'final_metrics' in data or 'best_auc' in data
        
        if has_baseline_format:
            return 'baseline'
        elif has_tagt_format:
            return 'tagt'
        elif has_optimized_format:
            return 'optimized'
        elif has_cv_format:
            return 'cross_validation'
        else:
            return 'simple'
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """
        Process a single result file and extract metrics.
        
        Args:
            file_path: Path to the result file
            
        Returns:
            List of metric records
        """
        logger.info(f"Processing file: {file_path}")
        
        # Load file based on extension
        if file_path.suffix.lower() == '.json':
            data = self.load_json_file(file_path)
        elif file_path.suffix.lower() in ['.pkl', '.pickle']:
            data = self.load_pkl_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
        
        if data is None:
            return []
        
        # Determine model name and extraction method
        model_name = self.determine_model_name(file_path, data)
        extraction_method = self.determine_extraction_method(data, file_path)
        
        source_info = {
            'model': model_name,
            'file_path': file_path,
            'extraction_method': extraction_method
        }
        
        logger.info(f"Using extraction method '{extraction_method}' for model '{model_name}'")
        
        # Extract metrics based on format
        if extraction_method == 'cross_validation':
            return self.extract_metrics_from_cross_validation(data, source_info)
        elif extraction_method == 'baseline':
            return self.extract_metrics_from_baseline(data, source_info)
        elif extraction_method == 'tagt':
            return self.extract_metrics_from_tagt(data, source_info)
        elif extraction_method == 'optimized':
            return self.extract_metrics_from_optimized(data, source_info)
        elif extraction_method == 'simple':
            return self.extract_metrics_from_simple(data, source_info)
        else:
            logger.warning(f"Unknown extraction method: {extraction_method}")
            return []
    
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect all results from found files and create a unified DataFrame.
        
        Returns:
            Pandas DataFrame with MultiIndex (model, split_type) and metrics as columns
        """
        logger.info("Starting results collection...")
        
        # Find all result files
        found_files = self.find_result_files()
        total_files = len(found_files['json']) + len(found_files['pkl'])
        logger.info(f"Found {total_files} result files to process")
        
        # Process all files
        all_records = []
        for file_type in found_files:
            for file_path in found_files[file_type]:
                records = self.process_file(file_path)
                all_records.extend(records)
        
        if not all_records:
            logger.warning("No records extracted from any files")
            return pd.DataFrame()
        
        logger.info(f"Extracted {len(all_records)} metric records")
        
                df = pd.DataFrame(all_records)
        
                logger.info("Creating pivot table with MultiIndex...")
        
        # Pivot to get metrics as columns
        pivot_df = df.pivot_table(
            index=['model', 'split_type'],
            columns='metric',
            values='value',
            aggfunc='first'  # Take first value if duplicates exist
        )
        
                metadata_df = df.groupby(['model', 'split_type']).agg({
            'std': 'first',
            'source_file': 'first',
            'data_type': 'first'
        }).add_suffix('_meta')
        
        # Store metadata separately (avoid pandas attribute warning)
        setattr(pivot_df, '_metadata', metadata_df)
        
        logger.info(f"Created DataFrame with shape {pivot_df.shape}")
        logger.info(f"Models: {list(pivot_df.index.get_level_values('model').unique())}")
        logger.info(f"Split types: {list(pivot_df.index.get_level_values('split_type').unique())}")
        logger.info(f"Metrics: {list(pivot_df.columns)}")
        
        return pivot_df
    
    def save_results(self, df: pd.DataFrame, output_path: str = None) -> str:
        """
        Save the consolidated results to a parquet file (or CSV if parquet not available).
        
        Args:
            df: DataFrame to save
            output_path: Output file path. If None, uses default location.
            
        Returns:
            Path where the file was saved
        """
        if output_path is None:
            results_dir = self.base_path / 'results'
            results_dir.mkdir(exist_ok=True)
            # Try parquet first, fallback to CSV
            try:
                # Test if parquet is available
                import pyarrow
                output_path = results_dir / 'all_models_results.parquet'
                use_parquet = True
            except ImportError:
                logger.warning("Parquet support not available. Falling back to CSV format.")
                output_path = results_dir / 'all_models_results.csv'
                use_parquet = False
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            use_parquet = output_path.suffix.lower() == '.parquet'
        
        # Save main DataFrame
        try:
            if use_parquet:
                df.to_parquet(output_path)
                logger.info(f"Results saved to parquet: {output_path}")
            else:
                df.to_csv(output_path)
                logger.info(f"Results saved to CSV: {output_path}")
        except ImportError as e:
            # Fallback to CSV if parquet fails
            if use_parquet:
                logger.warning(f"Parquet save failed: {e}. Falling back to CSV.")
                csv_path = output_path.with_suffix('.csv')
                df.to_csv(csv_path)
                output_path = csv_path
                logger.info(f"Results saved to CSV: {output_path}")
            else:
                raise
        
        # Save metadata if available
        if hasattr(df, '_metadata'):
            if use_parquet and output_path.suffix.lower() == '.parquet':
                try:
                    metadata_path = output_path.with_suffix('.metadata.parquet')
                    df._metadata.to_parquet(metadata_path)
                    logger.info(f"Metadata saved to parquet: {metadata_path}")
                except ImportError:
                    metadata_path = output_path.with_suffix('.metadata.csv')
                    df._metadata.to_csv(metadata_path)
                    logger.info(f"Metadata saved to CSV: {metadata_path}")
            else:
                metadata_path = output_path.with_suffix('.metadata.csv')
                df._metadata.to_csv(metadata_path)
                logger.info(f"Metadata saved to CSV: {metadata_path}")
        
        # Save summary statistics
        summary_path = output_path.with_suffix('.summary.csv')
        summary_stats = self.generate_summary_statistics(df)
        summary_stats.to_csv(summary_path)
        logger.info(f"Summary statistics saved to: {summary_path}")
        
        return str(output_path)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the collected results.
        
        Args:
            df: Consolidated results DataFrame
            
        Returns:
            Summary statistics DataFrame
        """
        summary_data = []
        
        for model in df.index.get_level_values('model').unique():
            model_data = df.loc[model]
            
            for metric in df.columns:
                metric_values = model_data[metric].dropna()
                if len(metric_values) > 0:
                    summary_data.append({
                        'model': model,
                        'metric': metric,
                        'count': len(metric_values),
                        'mean': metric_values.mean(),
                        'std': metric_values.std(),
                        'min': metric_values.min(),
                        'max': metric_values.max(),
                        'median': metric_values.median()
                    })
        
        return pd.DataFrame(summary_data)

def main():
    """
    Main function to run the results collection process.
    """
    # Initialize collector
    collector = ResultsCollector(base_path="C:\\Users\\ADMIN\\OneDrive\\Desktop\\SLE")
    
    # Collect all results
    results_df = collector.collect_all_results()
    
    if results_df.empty:
        logger.error("No results were collected. Please check your data files.")
        return
    
    # Try to save as parquet first (as per task requirements), fallback to CSV
    try:
        parquet_path = collector.base_path / 'results' / 'all_models_results.parquet'
        output_path = collector.save_results(results_df, str(parquet_path))
    except ImportError:
        logger.info("Parquet not available, using CSV format instead.")
        output_path = collector.save_results(results_df)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS COLLECTION SUMMARY")
    print("="*80)
    print(f"Total models processed: {len(results_df.index.get_level_values('model').unique())}")
    print(f"Total split types: {len(results_df.index.get_level_values('split_type').unique())}")
    print(f"Total metrics: {len(results_df.columns)}")
    print(f"Results saved to: {output_path}")
    print("\nDataFrame shape:", results_df.shape)
    print("\nDataFrame head:")
    print(results_df.head(10))
    
    # Show available models and metrics
    print("\n" + "-"*40)
    print("AVAILABLE MODELS:")
    print("-"*40)
    for model in sorted(results_df.index.get_level_values('model').unique()):
        print(f"  • {model}")
    
    print("\n" + "-"*40)
    print("AVAILABLE METRICS:")
    print("-"*40)
    for metric in sorted(results_df.columns):
        print(f"  • {metric}")
    
    print("\n" + "-"*40)
    print("SPLIT TYPES:")
    print("-"*40)
    for split_type in sorted(results_df.index.get_level_values('split_type').unique()):
        print(f"  • {split_type}")

if __name__ == "__main__":
    main()