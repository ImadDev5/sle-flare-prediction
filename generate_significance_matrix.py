Create a comprehensive summary report of the significance analysis.
    
    Parameters:
    -----------
    cv_df : pd.DataFrame
        Cross-validation results DataFrame
    significance_results : dict
        Dictionary of significance test results
    output_dir : str
        Directory to save the report
    """
    
    report_path = os.path.join(output_dir, "significance_analysis_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Statistical Significance Analysis Report\n\n")
        f.write("This report summarizes the statistical significance testing results for model comparisons.\n\n")
        
        # Model summary
        f.write("## Model Performance Summary\n\n")
        
        model_summary = cv_df.groupby('model_name').agg({
            'auc': ['mean', 'std', 'count'],
            'acc': ['mean', 'std'],
            'f1': ['mean', 'std'],
        }).round(4)
        
        f.write("### Mean ± Std (CV Folds)\n\n")
        f.write("| Model | AUC | Accuracy | F1 | N Folds |\n")
        f.write("|-------|-----|----------|----|---------|\n")
        
        for model in model_summary.index:
            auc_mean = model_summary.loc[model, ('auc', 'mean')]
            auc_std = model_summary.loc[model, ('auc', 'std')]
            acc_mean = model_summary.loc[model, ('acc', 'mean')]
            acc_std = model_summary.loc[model, ('acc', 'std')]
            f1_mean = model_summary.loc[model, ('f1', 'mean')]
            f1_std = model_summary.loc[model, ('f1', 'std')]
            n_folds = int(model_summary.loc[model, ('auc', 'count')])
            
            f.write(f"| {model} | {auc_mean:.3f} ± {auc_std:.3f} | "
                   f"{acc_mean:.3f} ± {acc_std:.3f} | "
                   f"{f1_mean:.3f} ± {f1_std:.3f} | {n_folds} |\n")
        
        f.write("\n")
        
        # Significance testing results
        f.write("## Significance Testing Results\n\n")
        f.write("Statistical significance tests using paired bootstrap (10,000 resamples).\n")
        f.write("P-values < 0.05 indicate statistically significant differences.\n\n")
        
        for metric_test, matrix in significance_results.items():
            if isinstance(matrix, pd.DataFrame):
                f.write(f"### {metric_test.upper()}\n\n")
                
                # Find most significant comparisons
                matrix_values = matrix.values.copy()
                np.fill_diagonal(matrix_values, np.nan)  # Ignore diagonal
                min_p_value = np.nanmin(matrix_values)
                max_p_value = np.nanmax(matrix_values)
                
                f.write(f"- Minimum p-value: {min_p_value:.6f}\n")
                f.write(f"- Maximum p-value: {max_p_value:.6f}\n")
                
                # Count significant pairs
                sig_005 = np.sum(matrix_values < 0.05)
                sig_001 = np.sum(matrix_values < 0.01)
                total_pairs = np.sum(~np.isnan(matrix_values))
                
                f.write(f"- Significant pairs (p < 0.05): {sig_005}/{total_pairs}\n")
                f.write(f"- Highly significant pairs (p < 0.01): {sig_001}/{total_pairs}\n\n")
                
                # Best performing model
                if 'auc' in metric_test:
                    model_means = cv_df.groupby('model_name')['auc'].mean()
                    best_model = model_means.idxmax()
                    f.write(f"**Best performing model (AUC):** {best_model} "
                           f"({model_means[best_model]:.4f})\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the statistical significance analysis:\n\n")
        
        # Find the best model across metrics
        try:
            auc_means = cv_df.groupby('model_name')['auc'].mean()
            acc_means = cv_df.groupby('model_name')['acc'].mean()
            
            best_auc_model = auc_means.idxmax()
            best_acc_model = acc_means.idxmax()
            
            f.write(f"1. **Highest AUC:** {best_auc_model} (AUC = {auc_means[best_auc_model]:.4f})\n")
            f.write(f"2. **Highest Accuracy:** {best_acc_model} (Acc = {acc_means[best_acc_model]:.4f})\n\n")
            
            if best_auc_model == best_acc_model:
                f.write(f"The **{best_auc_model}** model shows the best performance across multiple metrics.\n\n")
            else:
                f.write("Different models excel in different metrics. Consider the primary objective when selecting a model.\n\n")
                
        except Exception as e:
            f.write(f"Error generating recommendations: {e}\n\n")
        
        f.write("---\n")
        f.write("*Report generated automatically by significance analysis engine.*\n")
    
    print(f"\nComprehensive report saved to: {report_path}")

def main():
    """
    Main function to run the significance analysis.
    """
    
    print("=== Model Significance Testing Analysis ===\n")
    
    # Paths
    results_path = "results/all_models_results.csv"
    output_dir = "results"
    
    # Check if results file exists
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return
    
    try:
        # Load and prepare data
        print("Loading results data...")
        cv_df = load_and_prepare_results(results_path)
        
        if len(cv_df) == 0:
            print("No cross-validation data found for analysis.")
            return
        
        # Perform comprehensive significance analysis
        print("\nPerforming significance testing...")
        create_comprehensive_significance_analysis(cv_df, output_dir)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {output_dir}/")
        print("Files generated:")
        print("- significance_matrix_*.csv (for each metric)")
        print("- significance_summary.csv")
        print("- significance_detailed.csv")
        print("- significance_analysis_report.md")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()