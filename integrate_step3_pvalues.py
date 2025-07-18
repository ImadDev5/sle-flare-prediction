"""
Utility to integrate p-values from step 3 significance analysis into CV performance plots.

This script demonstrates how to load existing significance matrices and integrate them
with the cross-validation performance visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

from plot_cv_performance import CVPerformancePlotter

class Step3SignificanceIntegrator:
    """
    Integrate significance testing results from step 3 with CV performance plots.
    """
    
    def __init__(self, step3_results_dir="results", cv_results_dir="results/per_fold"):
        self.step3_results_dir = step3_results_dir
        self.cv_results_dir = cv_results_dir
        self.plotter = CVPerformancePlotter(cv_results_dir)
        
    def load_step3_significance_matrix(self, metric: str = 'auc') -> Optional[pd.DataFrame]:
        """
        Load significance matrix from step 3 analysis.
        
        Parameters:
        -----------
        metric : str
            Metric name ('auc', 'accuracy', etc.)
            
        Returns:
        --------
        Optional[pd.DataFrame]
            Significance matrix if found, None otherwise
        """
        # Common file patterns from step 3
        possible_files = [
            f"significance_matrix_{metric}.csv",
            f"significance_{metric}.csv",
            f"{metric}_significance_matrix.csv",
            "significance_summary.csv"
        ]
        
        for filename in possible_files:
            filepath = os.path.join(self.step3_results_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0)
                    print(f"Found step 3 significance matrix: {filepath}")
                    return df
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
        
        print(f"No step 3 significance matrix found for {metric}")
        return None
    
    def create_integrated_boxplot_with_step3_pvalues(self, metric: str = 'AUC') -> plt.Figure:
        """
        Create boxplot with significance annotations from step 3 p-values.
        
        Parameters:
        -----------
        metric : str
            Metric to plot ('AUC', 'Accuracy')
            
        Returns:
        --------
        plt.Figure
            Boxplot with step 3 significance annotations
        """
        # Prepare CV data
        df = self.plotter.prepare_plot_data()
        
        if len(df) == 0:
            raise ValueError("No CV data available")
        
        # Load step 3 significance matrix
        step3_sig_matrix = self.load_step3_significance_matrix(metric.lower())
        
                fig, ax = plt.subplots(figsize=(12, 8))
        
        # Remove rows with NaN values for this metric
        metric_df = df.dropna(subset=[metric])
        
                box_plot = sns.boxplot(
            data=metric_df, 
            x='Model', 
            y=metric, 
            ax=ax,
            palette=[self.plotter.colors.get(model, '#808080') 
                    for model in metric_df['Model'].unique()],
            showfliers=False
        )
        
        # Overlay strip plot
        strip_plot = sns.stripplot(
            data=metric_df, 
            x='Model', 
            y=metric, 
            ax=ax,
            color='black',
            alpha=0.7,
            size=6,
            jitter=0.2
        )
        
        # Customize appearance
        ax.set_title(f'{metric} Across CV Folds\n(with Step 3 Statistical Significance)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add significance annotations from step 3
        if step3_sig_matrix is not None:
            self._add_step3_significance_annotations(ax, metric_df, step3_sig_matrix, metric)
        else:
            # Fallback to computed significance
            computed_sig_matrix = self.plotter.compute_significance_matrix(metric_df, metric)
            self.plotter._add_significance_annotations(ax, metric_df, computed_sig_matrix, metric)
        
        # Add sample size annotations
        self.plotter._add_sample_size_annotations(ax, metric_df, metric)
        
        plt.tight_layout()
        return fig
    
    def _add_step3_significance_annotations(self, ax: plt.Axes, df: pd.DataFrame, 
                                          sig_matrix: pd.DataFrame, metric: str) -> None:
        """
        Add significance stars from step 3 analysis above boxplots.
        
        Parameters:
        -----------
        ax : plt.Axes
            Plot axes
        df : pd.DataFrame
            Data for the plot
        sig_matrix : pd.DataFrame
            Significance matrix from step 3
        metric : str
            Metric being plotted
        """
        models = df['Model'].unique()
        
        # Get the maximum value for positioning annotations
        y_max = df[metric].max()
        y_range = df[metric].max() - df[metric].min()
        annotation_height = y_max + 0.02 * y_range
        
        # Find the best performing model (highest median)
        model_medians = df.groupby('Model')[metric].median().sort_values(ascending=False)
        best_model = model_medians.index[0]
        
                model_mapping = self._create_model_name_mapping(models, sig_matrix.index)
        
        # Annotate comparisons with the best model
        for i, model in enumerate(models):
            if model != best_model:
                # Map model names to step 3 matrix
                step3_model = model_mapping.get(model, model)
                step3_best = model_mapping.get(best_model, best_model)
                
                if step3_model in sig_matrix.index and step3_best in sig_matrix.columns:
                    p_value = sig_matrix.loc[step3_model, step3_best]
                    stars = self.plotter.get_significance_stars(p_value)
                    
                    if stars != 'ns':
                        # Position annotation above the model's boxplot
                        ax.text(i, annotation_height, f'{stars}', 
                               ha='center', va='bottom', 
                               fontsize=12, fontweight='bold',
                               color='red' if stars == '***' else 'orange' if stars == '**' else 'black')
                        
                        # Add p-value as smaller text
                        ax.text(i, annotation_height + 0.01 * y_range, f'p={p_value:.3f}', 
                               ha='center', va='bottom', 
                               fontsize=8, style='italic', color='gray')
        
        # Add legend for significance levels
        legend_text = ("Significance vs best model (Step 3 analysis):\n"
                      "* p<0.05, ** p<0.01, *** p<0.001")
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=8, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _create_model_name_mapping(self, cv_models, step3_models) -> Dict[str, str]:
        """
        Create mapping between CV model names and step 3 model names.
        
        Parameters:
        -----------
        cv_models : list
            Model names from CV data
        step3_models : list
            Model names from step 3 significance matrix
            
        Returns:
        --------
        Dict[str, str]
            Mapping from CV names to step 3 names
        """
        mapping = {}
        
        for cv_model in cv_models:
            # Try exact match first
            if cv_model in step3_models:
                mapping[cv_model] = cv_model
                continue
            
            # Try common name variations
            variations = [
                cv_model.lower(),
                cv_model.upper(), 
                cv_model.replace('_', ' '),
                cv_model.replace(' ', '_'),
                cv_model.replace('_', '-'),
                cv_model.replace('-', '_')
            ]
            
            for variation in variations:
                if variation in step3_models:
                    mapping[cv_model] = variation
                    break
            else:
                # Partial matching
                for step3_model in step3_models:
                    if cv_model.lower() in step3_model.lower() or step3_model.lower() in cv_model.lower():
                        mapping[cv_model] = step3_model
                        break
                else:
                    # No match found
                    mapping[cv_model] = cv_model
                    print(f"Warning: No mapping found for CV model '{cv_model}' in step 3 models")
        
        return mapping
    
    def create_comparison_report(self) -> str:
        """
        Create a comprehensive comparison report between step 3 and CV significance results.
        
        Returns:
        --------
        str
            Path to the saved comparison report
        """
        report_path = os.path.join(self.plotter.output_dir, 'step3_vs_cv_significance_comparison.md')
        
        with open(report_path, 'w') as f:
            f.write("# Step 3 vs Cross-Validation Significance Comparison\n\n")
            f.write("This report compares statistical significance results from step 3 analysis ")
            f.write("with bootstrap significance testing from cross-validation data.\n\n")
            
            # Compare for each metric
            for metric in ['AUC', 'Accuracy']:
                f.write(f"## {metric} Comparison\n\n")
                
                # Load step 3 results
                step3_matrix = self.load_step3_significance_matrix(metric.lower())
                
                # Compute CV results
                df = self.plotter.prepare_plot_data()
                if len(df) > 0:
                    cv_matrix = self.plotter.compute_significance_matrix(df, metric)
                    
                    if step3_matrix is not None:
                        f.write("### Step 3 Significance Matrix\n\n")
                        f.write(step3_matrix.round(4).to_markdown())
                        f.write("\n\n")
                        
                        f.write("### Cross-Validation Bootstrap Significance Matrix\n\n")
                        f.write(cv_matrix.round(4).to_markdown())
                        f.write("\n\n")
                        
                        # Compare overlapping models
                        common_models = set(step3_matrix.index) & set(cv_matrix.index)
                        if common_models:
                            f.write("### Comparison for Common Models\n\n")
                            f.write("| Model 1 | Model 2 | Step 3 p-value | CV p-value | Agreement |\n")
                            f.write("|---------|---------|----------------|------------|----------|\n")
                            
                            for model1 in common_models:
                                for model2 in common_models:
                                    if model1 != model2:
                                        step3_p = step3_matrix.loc[model1, model2]
                                        cv_p = cv_matrix.loc[model1, model2]
                                        
                                        # Check agreement in significance
                                        step3_sig = step3_p < 0.05 if not pd.isna(step3_p) else False
                                        cv_sig = cv_p < 0.05 if not pd.isna(cv_p) else False
                                        agreement = "✅" if step3_sig == cv_sig else "❌"
                                        
                                        f.write(f"| {model1} | {model2} | {step3_p:.4f} | {cv_p:.4f} | {agreement} |\n")
                            
                            f.write("\n")
                    else:
                        f.write("Step 3 significance matrix not found for this metric.\n\n")
                        f.write("### Cross-Validation Bootstrap Significance Matrix\n\n")
                        f.write(cv_matrix.round(4).to_markdown())
                        f.write("\n\n")
                else:
                    f.write("No cross-validation data available for this metric.\n\n")
            
            f.write("---\n")
            f.write("*Report generated automatically by integrate_step3_pvalues.py*\n")
        
        print(f"Comparison report saved: {report_path}")
        return report_path
    
    def generate_integrated_plots(self) -> Dict[str, str]:
        """
        Generate all plots with step 3 significance integration.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping plot names to file paths
        """
        saved_plots = {}
        
        print("Generating integrated plots with step 3 significance...")
        
        for metric in ['AUC', 'Accuracy']:
            try:
                fig = self.create_integrated_boxplot_with_step3_pvalues(metric)
                plot_path = os.path.join(self.plotter.output_dir, f'cv_performance_{metric.lower()}_step3_integrated.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_plots[f'{metric} with Step 3 Significance'] = plot_path
                print(f"Saved: {plot_path}")
            except Exception as e:
                print(f"Error creating {metric} integrated plot: {e}")
        
                report_path = self.create_comparison_report()
        saved_plots['Comparison Report'] = report_path
        
        return saved_plots

def main():
    """
    Main function to integrate step 3 significance results with CV performance plots.
    """
    print("=" * 60)
    print("STEP 3 SIGNIFICANCE INTEGRATION")
    print("=" * 60)
    
    # Initialize integrator
    integrator = Step3SignificanceIntegrator()
    
        saved_plots = integrator.generate_integrated_plots()
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    
    if saved_plots:
        print(f"📊 Generated {len(saved_plots)} files:")
        for plot_name, path in saved_plots.items():
            print(f"  • {plot_name}: {path}")
    
    print("\n🎯 Integration features:")
    print("  • Loads existing significance matrices from step 3")
    print("  • Annotates CV boxplots with step 3 p-values")
    print("  • Creates comparison report between methods")
    print("  • Handles model name mapping automatically")
    
    return saved_plots

if __name__ == "__main__":
    main()