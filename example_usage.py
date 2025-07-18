"""
Example script demonstrating how to use the results collection and analysis module.

This script shows various ways to analyze the consolidated model results.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from src.analysis import (
    load_consolidated_results,
    compare_models,
    get_best_performing_models,
    get_cross_validation_results,
    print_results_overview
)

def main():
    """Demonstrate usage of the results analysis module."""
    
    print("="*80)
    print("MODEL RESULTS ANALYSIS EXAMPLE")
    print("="*80)
    
    try:
        # Load consolidated results
        print("Loading consolidated results...")
        df = load_consolidated_results()
        
        # Print comprehensive overview
        print_results_overview(df)
        
        # Example 1: Compare all TAGT variants on AUC
        print("\n" + "="*80)
        print("EXAMPLE 1: TAGT MODEL COMPARISON ON AUC")
        print("="*80)
        
        tagt_models = [model for model in df.index.get_level_values('model').unique() 
                      if 'tagt' in model.lower()]
        
        if tagt_models:
            comparison = compare_models(df, tagt_models, 'auc')
            print("\nTAGT Models ranked by AUC:")
            for i, (_, row) in enumerate(comparison.iterrows(), 1):
                print(f"{i}. {row['model']}: {row['value']:.4f} ({row['split_type']})")
        
        # Example 2: Find best models across all metrics
        print("\n" + "="*80)
        print("EXAMPLE 2: TOP 3 MODELS FOR EACH METRIC")
        print("="*80)
        
        metrics = ['auc', 'acc', 'f1', 'prec', 'recall']
        for metric in metrics:
            if metric in df.columns:
                print(f"\nTop 3 models by {metric.upper()}:")
                best_models = get_best_performing_models(df, metric=metric, top_n=3)
                for i, (_, row) in enumerate(best_models.iterrows(), 1):
                    print(f"  {i}. {row['model']}: {row['value']:.4f}")
        
        # Example 3: Cross-validation analysis
        print("\n" + "="*80)
        print("EXAMPLE 3: CROSS-VALIDATION ANALYSIS")
        print("="*80)
        
        cv_models = [model for model in df.index.get_level_values('model').unique() 
                    if any(split.startswith('cv_fold_') for split in df.loc[model].index)]
        
        print("Models with cross-validation data:")
        for model in cv_models:
            cv_results = get_cross_validation_results(df, model)
            if not cv_results.empty and 'auc' in cv_results.columns:
                auc_scores = cv_results['auc'].dropna()
                cv_folds = [col for col in auc_scores.index if col.startswith('cv_fold_')]
                if cv_folds:
                    fold_scores = auc_scores[cv_folds]
                    mean_auc = fold_scores.mean()
                    std_auc = fold_scores.std()
                    print(f"\n{model}:")
                    print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}")
                    print(f"  Fold scores: {fold_scores.values}")
        
        # Example 4: Statistical comparison (if scipy available)
        print("\n" + "="*80)
        print("EXAMPLE 4: STATISTICAL COMPARISON")
        print("="*80)
        
        try:
            from scipy.stats import ttest_rel
            import numpy as np
            
            # Compare two models with CV data
            if len(cv_models) >= 2:
                model1, model2 = cv_models[0], cv_models[1]
                
                # Get CV fold results for both models
                cv1 = get_cross_validation_results(df, model1)
                cv2 = get_cross_validation_results(df, model2)
                
                if not cv1.empty and not cv2.empty and 'auc' in cv1.columns and 'auc' in cv2.columns:
                    # Get fold scores
                    folds1 = [score for idx, score in cv1['auc'].items() if idx.startswith('cv_fold_')]
                    folds2 = [score for idx, score in cv2['auc'].items() if idx.startswith('cv_fold_')]
                    
                    if len(folds1) == len(folds2) and len(folds1) > 1:
                        folds1, folds2 = np.array(folds1), np.array(folds2)
                        
                        # Paired t-test
                        t_stat, p_value = ttest_rel(folds1, folds2)
                        
                        # Effect size (Cohen's d)
                        diff = folds1 - folds2
                        cohens_d = diff.mean() / diff.std()
                        
                        print(f"Comparing {model1} vs {model2} (AUC):")
                        print(f"  {model1}: {folds1.mean():.4f} ± {folds1.std():.4f}")
                        print(f"  {model2}: {folds2.mean():.4f} ± {folds2.std():.4f}")
                        print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.3f}")
                        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
                        
                        if p_value < 0.05:
                            winner = model1 if folds1.mean() > folds2.mean() else model2
                            print(f"  Result: {winner} significantly better (p < 0.05)")
                        else:
                            print("  Result: No significant difference (p >= 0.05)")
                        
        except ImportError:
            print("scipy not available for statistical tests")
        except Exception as e:
            print(f"Statistical comparison failed: {e}")
        
        # Example 5: Data export for further analysis
        print("\n" + "="*80)
        print("EXAMPLE 5: DATA EXPORT")
        print("="*80)
        
        # Export specific data subsets
        exports_dir = Path("results/exports")
        exports_dir.mkdir(exist_ok=True)
        
        # Export CV means only
        cv_means = df.xs('cv_mean', level='split_type', drop_level=False)
        cv_means.to_csv(exports_dir / "cv_means.csv")
        print(f"CV means exported to: {exports_dir / 'cv_means.csv'}")
        
        # Export best TAGT model results
        if 'tagt_cv' in df.index.get_level_values('model'):
            tagt_results = df.loc['tagt_cv']
            tagt_results.to_csv(exports_dir / "tagt_cv_results.csv")
            print(f"TAGT CV results exported to: {exports_dir / 'tagt_cv_results.csv'}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("All consolidated results are available in the DataFrame 'df'")
        print("for further analysis, visualization, or statistical testing.")
        
    except FileNotFoundError:
        print("Error: Consolidated results not found.")
        print("Please run 'python src/analysis/collect_results.py' first.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()