"""
Step 4: Confidence-interval calculation & bootstrapping integration script

For each metric per model, this script draws 1000 bootstrap resamples of the test-set 
predictions, computes mean & 95% CI, and appends to the main DataFrame (columns 
`metric_lower`, `metric_upper`). These will be used as error bars in bar plots.

This script integrates the bootstrap confidence interval calculation into the 
existing analysis pipeline and updates the main results DataFrame.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add project root to path
sys.path.append('.')

from src.analysis.collect_results import ResultsCollector
from src.analysis.bootstrap_confidence import BootstrapConfidenceCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_bootstrap_confidence_intervals():
    """
    Main function to integrate bootstrap confidence intervals into the analysis pipeline.
    
    This function:
    1. Loads the existing results DataFrame
    2. Calculates bootstrap confidence intervals for each metric per model
    3. Appends CI columns to the main DataFrame
    4. Saves the enhanced DataFrame for use in downstream analysis
    """
    
    logger.info("="*80)
    logger.info("Step 4: Confidence-interval calculation & bootstrapping")
    logger.info("="*80)
    
    # Initialize components
    base_path = Path("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SLE")
    collector = ResultsCollector(base_path=str(base_path))
    
    # Use 1000 bootstrap resamples and 95% confidence level as specified
    bootstrap_calc = BootstrapConfidenceCalculator(
        n_bootstrap=1000, 
        confidence_level=0.95, 
        random_state=42
    )
    
    logger.info("Collecting all model results...")
    
    # Step 1: Collect all existing results
    main_dataframe = collector.collect_all_results()
    
    if main_dataframe.empty:
        logger.error("No results found. Cannot proceed with bootstrap CI calculation.")
        return None
    
    logger.info(f"Main DataFrame loaded with shape: {main_dataframe.shape}")
    logger.info(f"Available metrics: {list(main_dataframe.columns)}")
    logger.info(f"Models: {list(main_dataframe.index.get_level_values('model').unique())}")
    
    # Step 2: Calculate bootstrap confidence intervals
    logger.info("Drawing 1000 bootstrap resamples and computing 95% confidence intervals...")
    
    enhanced_dataframe = bootstrap_calc.add_bootstrap_confidence_intervals(main_dataframe)
    
    # Step 3: Verify the enhancement
    original_columns = set(main_dataframe.columns)
    new_columns = set(enhanced_dataframe.columns) - original_columns
    
    logger.info(f"Enhanced DataFrame shape: {enhanced_dataframe.shape}")
    logger.info(f"Original columns: {len(original_columns)}")
    logger.info(f"New CI columns added: {len(new_columns)}")
    
    # List the new CI columns
    ci_columns = [col for col in new_columns if col.endswith(('_lower', '_upper'))]
    logger.info("Bootstrap confidence interval columns added:")
    for col in sorted(ci_columns):
        logger.info(f"  • {col}")
    
    # Step 4: Save enhanced DataFrame
    results_dir = base_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save the main enhanced DataFrame
    main_output_path = results_dir / "main_results_with_bootstrap_ci.csv"
    enhanced_dataframe.to_csv(main_output_path)
    logger.info(f"Enhanced main DataFrame saved to: {main_output_path}")
    
    # Also save as parquet if possible for better performance
    try:
        parquet_path = results_dir / "main_results_with_bootstrap_ci.parquet"
        enhanced_dataframe.to_parquet(parquet_path)
        logger.info(f"Enhanced DataFrame also saved as parquet: {parquet_path}")
    except ImportError:
        logger.warning("Parquet support not available. Saved as CSV only.")
    
    # Step 5: Create summary for analysis
    summary_df = bootstrap_calc.create_summary_with_cis(enhanced_dataframe)
    summary_path = results_dir / "model_summary_with_bootstrap_ci.csv"
    summary_df.to_csv(summary_path)
    logger.info(f"Model summary with CI saved to: {summary_path}")
    
    # Step 6: Generate validation report
    generate_validation_report(enhanced_dataframe, results_dir)
    
    logger.info("="*80)
    logger.info("STEP 4 COMPLETION SUMMARY")
    logger.info("="*80)
    logger.info(f"✓ Bootstrap resamples: 1000")
    logger.info(f"✓ Confidence level: 95%")
    logger.info(f"✓ Models processed: {len(enhanced_dataframe.index.get_level_values('model').unique())}")
    logger.info(f"✓ Metrics enhanced: {len([col for col in enhanced_dataframe.columns if not col.endswith(('_lower', '_upper'))])}")
    logger.info(f"✓ CI columns added: {len(ci_columns)}")
    logger.info(f"✓ Main DataFrame enhanced: {main_output_path}")
    logger.info(f"✓ Ready for error bar plotting")
    
    return enhanced_dataframe

def generate_validation_report(enhanced_df, output_dir):
    """
    Generate a validation report showing the bootstrap confidence intervals.
    
    Args:
        enhanced_df: DataFrame with bootstrap confidence intervals
        output_dir: Directory to save the report
    """
    
    logger.info("Generating bootstrap confidence interval validation report...")
    
    report_path = output_dir / "bootstrap_ci_validation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("BOOTSTRAP CONFIDENCE INTERVAL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Bootstrap resamples: 1000\n")
        f.write(f"Confidence level: 95%\n\n")
        
        # Get base metrics
        base_metrics = ['acc', 'auc', 'f1', 'prec', 'recall', 'spec']
        available_metrics = [m for m in base_metrics if m in enhanced_df.columns]
        
        models = enhanced_df.index.get_level_values('model').unique()
        
        for metric in available_metrics:
            f.write(f"\n{metric.upper()} BOOTSTRAP CONFIDENCE INTERVALS\n")
            f.write("-" * 50 + "\n")
            
            lower_col = f"{metric}_lower"
            upper_col = f"{metric}_upper"
            
            for model in models:
                try:
                    model_data = enhanced_df.loc[model]
                    
                    # Get the best available split for this model
                    preferred_splits = ['cv_mean', 'final', 'best', 'overall', 'test']
                    selected_split = None
                    
                    for split in preferred_splits:
                        if split in model_data.index:
                            selected_split = split
                            break
                    
                    if selected_split is None:
                        selected_split = model_data.index[0]
                    
                    # Extract values
                    value = model_data.loc[selected_split, metric]
                    
                    if pd.isna(value):
                        f.write(f"{model:30s}: No data\n")
                        continue
                    
                    if lower_col in model_data.columns and upper_col in model_data.columns:
                        lower = model_data.loc[selected_split, lower_col]
                        upper = model_data.loc[selected_split, upper_col]
                        
                        if not pd.isna(lower) and not pd.isna(upper):
                            ci_width = upper - lower
                            f.write(f"{model:30s}: {value:.4f} ± {ci_width/2:.4f} [{lower:.4f}, {upper:.4f}]\n")
                        else:
                            f.write(f"{model:30s}: {value:.4f} [No CI computed]\n")
                    else:
                        f.write(f"{model:30s}: {value:.4f} [No CI available]\n")
                        
                except Exception as e:
                    f.write(f"{model:30s}: Error - {e}\n")
        
        # Add interpretation guide
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n")
        f.write("• Bootstrap confidence intervals provide uncertainty estimates for model performance\n")
        f.write("• 95% CI means we are 95% confident the true performance lies within the interval\n")
        f.write("• Wider intervals indicate higher uncertainty in the performance estimate\n")
        f.write("• Overlapping intervals suggest no significant performance difference\n")
        f.write("• These intervals can be used as error bars in bar plots for visualization\n")
        f.write("\nCOLUMNS ADDED TO MAIN DATAFRAME:\n")
        
        ci_columns = [col for col in enhanced_df.columns if col.endswith(('_lower', '_upper'))]
        for col in sorted(ci_columns):
            f.write(f"• {col}\n")
    
    logger.info(f"Validation report saved to: {report_path}")

def main():
    """Main execution function."""
    try:
        enhanced_dataframe = integrate_bootstrap_confidence_intervals()
        
        if enhanced_dataframe is not None:
            logger.info("\n" + "="*80)
            logger.info("STEP 4 COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info("The main DataFrame has been enhanced with bootstrap confidence intervals.")
            logger.info("Each metric now has corresponding '_lower' and '_upper' columns.")
            logger.info("These can be used as error bars in bar plots for model comparison.")
            logger.info("Files saved in the 'results' directory.")
            return enhanced_dataframe
        else:
            logger.error("Step 4 failed to complete.")
            return None
            
    except Exception as e:
        logger.error(f"Error in Step 4 execution: {e}")
        raise

if __name__ == "__main__":
    main()