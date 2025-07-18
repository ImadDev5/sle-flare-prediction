"""
Cross-validation performance visualization with statistical significance testing.

This script creates boxplots and strip-plots of per-fold AUC and Accuracy across models,
visually demonstrating variance and annotating statistical significance stars.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from load_per_fold_results import PerFoldResultsLoader, get_model_summary, list_available_models
from src.analysis.significance import create_significance_matrix, paired_bootstrap

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CVPerformancePlotter:
    """
    Create comprehensive cross-validation performance visualizations with statistical testing.
    """
    
    def __init__(self, results_dir="results/per_fold", output_dir="results/plots"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.loader = PerFoldResultsLoader(results_dir)
        
                os.makedirs(output_dir, exist_ok=True)
        
        # Color palette for consistent plotting
        self.colors = {
            'Random_Forest': '#2E8B57',      # Sea Green
            'SVM_RBF': '#4169E1',            # Royal Blue  
            'Logistic_Regression': '#FF6347', # Tomato
            'Simple_LSTM': '#9370DB',        # Medium Purple
            'TAGT': '#FF4500'                # Orange Red
        }
        
    def prepare_plot_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting by extracting per-fold metrics.
        
        Returns:
        --------
        pd.DataFrame
            Long-format DataFrame with fold-level metrics for all models
        """
        plot_data = []
        
        available_models = list_available_models(self.results_dir)
        print(f"Available models: {available_models}")
        
        for model in available_models:
            model_folds = self.loader.load_all_folds(model)
            
            for fold_idx, fold_data in enumerate(model_folds):
                if fold_data and 'metrics' in fold_data:
                    metrics = fold_data['metrics']
                    
                    plot_data.append({
                        'Model': model,
                        'Fold': fold_idx + 1,
                        'AUC': metrics.get('auc', np.nan),
                        'Accuracy': metrics.get('accuracy', np.nan),
                        'F1': metrics.get('f1', np.nan),
                        'Precision': metrics.get('precision', np.nan),
                        'Recall': metrics.get('recall', np.nan)
                    })
        
        df = pd.DataFrame(plot_data)
        print(f"Prepared data for {len(df)} fold-model combinations")
        
        return df
    
    def get_significance_stars(self, p_value: float) -> str:
        """
        Convert p-value to significance stars.
        
        Parameters:
        -----------
        p_value : float
            Statistical significance p-value
            
        Returns:
        --------
        str
            Star notation (*, **, ***, ns)
        """
        if pd.isna(p_value):
            return 'ns'
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
    
    def compute_significance_matrix(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Compute significance matrix for model comparisons using bootstrap test.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Per-fold results data
        metric : str
            Metric to compare (e.g., 'AUC', 'Accuracy')
            
        Returns:
        --------
        pd.DataFrame
            Matrix of p-values for pairwise comparisons
        """
        models = df['Model'].unique()
        n_models = len(models)
        
        # Initialize significance matrix
        sig_matrix = pd.DataFrame(
            index=models, 
            columns=models, 
            dtype=float
        )
        
        # Fill diagonal with 1.0 (same model)
        for model in models:
            sig_matrix.loc[model, model] = 1.0
        
        # Compute pairwise comparisons
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:  # Skip diagonal and lower triangle
                    continue
                
                # Get metric values for both models
                model1_values = df[df['Model'] == model1][metric].dropna().values
                model2_values = df[df['Model'] == model2][metric].dropna().values
                
                if len(model1_values) > 0 and len(model2_values) > 0:
                    try:
                        # Use paired bootstrap test
                        min_len = min(len(model1_values), len(model2_values))
                        p_value, _ = paired_bootstrap(
                            model1_values[:min_len], 
                            model2_values[:min_len],
                            n=10000
                        )
                        
                        sig_matrix.loc[model1, model2] = p_value
                        sig_matrix.loc[model2, model1] = p_value
                        
                    except Exception as e:
                        print(f"Error comparing {model1} vs {model2}: {e}")
                        sig_matrix.loc[model1, model2] = np.nan
                        sig_matrix.loc[model2, model1] = np.nan
                else:
                    sig_matrix.loc[model1, model2] = np.nan
                    sig_matrix.loc[model2, model1] = np.nan
        
        return sig_matrix
    
    def create_combined_boxplot(self, df: pd.DataFrame, metrics: List[str] = ['AUC', 'Accuracy']) -> plt.Figure:
        """
        Create combined boxplot with strip-plot overlay for multiple metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Per-fold results data
        metrics : List[str]
            List of metrics to plot
            
        Returns:
        --------
        plt.Figure
            Combined boxplot figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Remove rows with NaN values for this metric
            metric_df = df.dropna(subset=[metric])
            
            if len(metric_df) == 0:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
                        box_plot = sns.boxplot(
                data=metric_df, 
                x='Model', 
                y=metric, 
                ax=ax,
                palette=[self.colors.get(model, '#808080') for model in metric_df['Model'].unique()],
                showfliers=False  # We'll show individual points with stripplot
            )
            
            # Overlay strip plot for individual fold values
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
            ax.set_title(f'{metric} Across CV Folds', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            
            # Compute and add significance annotations
            sig_matrix = self.compute_significance_matrix(metric_df, metric)
            self._add_significance_annotations(ax, metric_df, sig_matrix, metric)
            
            # Add sample size annotations
            self._add_sample_size_annotations(ax, metric_df, metric)
        
        plt.tight_layout()
        return fig
    
    def _add_significance_annotations(self, ax: plt.Axes, df: pd.DataFrame, 
                                    sig_matrix: pd.DataFrame, metric: str) -> None:
        """
        Add significance stars above boxplots for pairwise comparisons.
        
        Parameters:
        -----------
        ax : plt.Axes
            Plot axes
        df : pd.DataFrame
            Data for the plot
        sig_matrix : pd.DataFrame
            Matrix of p-values
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
        
        # Annotate comparisons with the best model
        for i, model in enumerate(models):
            if model != best_model and model in sig_matrix.index and best_model in sig_matrix.columns:
                p_value = sig_matrix.loc[model, best_model]
                stars = self.get_significance_stars(p_value)
                
                if stars != 'ns':
                    # Position annotation above the model's boxplot
                    ax.text(i, annotation_height, stars, 
                           ha='center', va='bottom', 
                           fontsize=12, fontweight='bold',
                           color='red' if stars == '***' else 'orange' if stars == '**' else 'black')
        
        # Add legend for significance levels
        legend_text = "Significance vs best model:\n* p<0.05, ** p<0.01, *** p<0.001"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=8, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _add_sample_size_annotations(self, ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
        """
        Add sample size (number of folds) annotations below each boxplot.
        
        Parameters:
        -----------
        ax : plt.Axes
            Plot axes
        df : pd.DataFrame
            Data for the plot
        metric : str
            Metric being plotted
        """
        models = df['Model'].unique()
        
        # Get the minimum value for positioning annotations
        y_min = df[metric].min()
        y_range = df[metric].max() - df[metric].min()
        annotation_height = y_min - 0.05 * y_range
        
        for i, model in enumerate(models):
            n_folds = len(df[df['Model'] == model][metric].dropna())
            ax.text(i, annotation_height, f'n={n_folds}', 
                   ha='center', va='top', 
                   fontsize=9, style='italic', color='gray')
    
    def create_detailed_comparison_plot(self, df: pd.DataFrame, 
                                      model1: str, model2: str, 
                                      metrics: List[str] = ['AUC', 'Accuracy']) -> plt.Figure:
        """
        Create detailed comparison plot between two specific models.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Per-fold results data
        model1, model2 : str
            Names of models to compare
        metrics : List[str]
            Metrics to compare
            
        Returns:
        --------
        plt.Figure
            Detailed comparison figure
        """
        # Filter data for the two models
        comparison_df = df[df['Model'].isin([model1, model2])].copy()
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Box plot with individual points
            sns.boxplot(data=comparison_df, x='Model', y=metric, ax=ax,
                       palette=[self.colors.get(model1, '#808080'), 
                               self.colors.get(model2, '#808080')])
            
            sns.stripplot(data=comparison_df, x='Model', y=metric, ax=ax,
                         color='black', alpha=0.7, size=8)
            
            # Statistical test
            model1_values = comparison_df[comparison_df['Model'] == model1][metric].dropna().values
            model2_values = comparison_df[comparison_df['Model'] == model2][metric].dropna().values
            
            if len(model1_values) > 0 and len(model2_values) > 0:
                try:
                    min_len = min(len(model1_values), len(model2_values))
                    p_value, ci = paired_bootstrap(
                        model1_values[:min_len], 
                        model2_values[:min_len]
                    )
                    
                    stars = self.get_significance_stars(p_value)
                    
                    # Add statistical annotation
                    ax.text(0.5, 0.95, f'p = {p_value:.4f} {stars}', 
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                    
                    # Add effect size (difference in means)
                    diff = np.mean(model1_values) - np.mean(model2_values)
                    ax.text(0.5, 0.85, f'Difference: {diff:.4f}', 
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=10)
                    
                except Exception as e:
                    ax.text(0.5, 0.95, f'Test failed: {str(e)[:30]}...', 
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=10, color='red')
            
            ax.set_title(f'{metric}: {model1} vs {model2}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_performance_heatmap(self, df: pd.DataFrame, metric: str = 'AUC') -> plt.Figure:
        """
        Create a heatmap showing statistical significance between all model pairs.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Per-fold results data
        metric : str
            Metric to analyze
            
        Returns:
        --------
        plt.Figure
            Significance heatmap figure
        """
        # Compute significance matrix
        sig_matrix = self.compute_significance_matrix(df, metric)
        
        # Convert p-values to stars for annotation
        star_matrix = sig_matrix.applymap(self.get_significance_stars)
        
                fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use -log10(p-value) for better visualization
        log_sig_matrix = -np.log10(sig_matrix.replace(0, 1e-10))
        
        sns.heatmap(log_sig_matrix, 
                   annot=star_matrix, 
                   fmt='', 
                   cmap='RdYlBu_r',
                   center=1.3,  # -log10(0.05) = 1.3
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': '-log10(p-value)'},
                   ax=ax)
        
        ax.set_title(f'Statistical Significance Matrix: {metric}\n'
                    f'(Bootstrap test, 10,000 resamples)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        # Add significance threshold lines
        ax.axhline(y=0, color='white', linewidth=2)
        ax.axvline(x=0, color='white', linewidth=2)
        
        # Add legend
        legend_text = ("Significance levels:\n"
                      "* p<0.05 (modest evidence)\n"
                      "** p<0.01 (strong evidence)\n"
                      "*** p<0.001 (very strong evidence)\n"
                      "ns = not significant")
        
        ax.text(1.15, 0.5, legend_text, transform=ax.transAxes, 
               fontsize=10, va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        return fig
    
    def generate_all_plots(self) -> Dict[str, str]:
        """
        Generate all cross-validation performance plots.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping plot names to saved file paths
        """
        print("Generating cross-validation performance plots...")
        
        # Prepare data
        df = self.prepare_plot_data()
        
        if len(df) == 0:
            print("No data available for plotting!")
            return {}
        
        saved_plots = {}
        
        # 1. Combined boxplot for AUC and Accuracy
        print("Creating combined boxplot...")
        try:
            fig1 = self.create_combined_boxplot(df, ['AUC', 'Accuracy'])
            plot1_path = os.path.join(self.output_dir, 'cv_performance_boxplot.png')
            fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            saved_plots['Combined Boxplot'] = plot1_path
            print(f"Saved: {plot1_path}")
        except Exception as e:
            print(f"Error creating combined boxplot: {e}")
        
        # 2. All metrics boxplot
        print("Creating all metrics boxplot...")
        try:
            fig2 = self.create_combined_boxplot(df, ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall'])
            plot2_path = os.path.join(self.output_dir, 'cv_performance_all_metrics.png')
            fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            saved_plots['All Metrics Boxplot'] = plot2_path
            print(f"Saved: {plot2_path}")
        except Exception as e:
            print(f"Error creating all metrics boxplot: {e}")
        
        # 3. Significance heatmaps
        for metric in ['AUC', 'Accuracy']:
            print(f"Creating significance heatmap for {metric}...")
            try:
                fig3 = self.create_performance_heatmap(df, metric)
                plot3_path = os.path.join(self.output_dir, f'cv_significance_heatmap_{metric.lower()}.png')
                fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
                plt.close(fig3)
                saved_plots[f'{metric} Significance Heatmap'] = plot3_path
                print(f"Saved: {plot3_path}")
            except Exception as e:
                print(f"Error creating {metric} heatmap: {e}")
        
        # 4. Detailed comparisons (if TAGT is available)
        available_models = df['Model'].unique()
        if 'TAGT' in available_models:
            for other_model in available_models:
                if other_model != 'TAGT':
                    print(f"Creating detailed comparison: TAGT vs {other_model}...")
                    try:
                        fig4 = self.create_detailed_comparison_plot(df, 'TAGT', other_model)
                        plot4_path = os.path.join(self.output_dir, f'cv_comparison_TAGT_vs_{other_model}.png')
                        fig4.savefig(plot4_path, dpi=300, bbox_inches='tight')
                        plt.close(fig4)
                        saved_plots[f'TAGT vs {other_model}'] = plot4_path
                        print(f"Saved: {plot4_path}")
                    except Exception as e:
                        print(f"Error creating TAGT vs {other_model} comparison: {e}")
        
        return saved_plots
    
    def save_summary_table(self) -> str:
        """
        Save a summary table of model performance statistics.
        
        Returns:
        --------
        str
            Path to saved summary table
        """
        df = self.prepare_plot_data()
        
        if len(df) == 0:
            print("No data available for summary table!")
            return ""
        
        # Calculate summary statistics
        summary_stats = []
        
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            
            for metric in ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']:
                metric_values = model_data[metric].dropna()
                
                if len(metric_values) > 0:
                    summary_stats.append({
                        'Model': model,
                        'Metric': metric,
                        'Mean': metric_values.mean(),
                        'Std': metric_values.std(),
                        'Min': metric_values.min(),
                        'Max': metric_values.max(),
                        'Median': metric_values.median(),
                        'N_Folds': len(metric_values)
                    })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save to CSV
        summary_path = os.path.join(self.output_dir, 'cv_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        
        # Also save a formatted version
        formatted_path = os.path.join(self.output_dir, 'cv_performance_summary_formatted.csv')
        
        # Pivot for better readability
        pivot_df = summary_df.pivot_table(
            index='Model', 
            columns='Metric', 
            values=['Mean', 'Std'], 
            aggfunc='first'
        )
        
        # Flatten column names
        pivot_df.columns = [f'{metric}_{stat}' for stat, metric in pivot_df.columns]
        pivot_df = pivot_df.round(4)
        
        pivot_df.to_csv(formatted_path)
        
        print(f"Saved summary table: {summary_path}")
        print(f"Saved formatted summary: {formatted_path}")
        
        return summary_path

def main():
    """
    Main function to generate all cross-validation performance plots.
    """
    print("=" * 60)
    print("CROSS-VALIDATION PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    # Initialize plotter
    plotter = CVPerformancePlotter()
    
    # Check data availability
    available_models = list_available_models()
    if not available_models:
        print("❌ No complete per-fold data found!")
        print("Please run compute_per_fold_results.py first.")
        return
    
    print(f"✅ Found complete data for: {', '.join(available_models)}")
    
        saved_plots = plotter.generate_all_plots()
    
    # Save summary table
    summary_path = plotter.save_summary_table()
    
    # Print summary
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    
    if saved_plots:
        print(f"📊 Generated {len(saved_plots)} plots:")
        for plot_name, path in saved_plots.items():
            print(f"  • {plot_name}: {path}")
    
    if summary_path:
        print(f"\n📋 Summary table: {summary_path}")
    
    print(f"\n📁 All files saved to: {plotter.output_dir}")
    print("\n🎯 Key features:")
    print("  • Boxplots with strip-plot overlays showing individual fold values")
    print("  • Statistical significance annotations (*, **, ***)")
    print("  • Significance heatmaps with p-value matrices")
    print("  • Detailed pairwise comparisons")
    print("  • Bootstrap hypothesis testing (10,000 resamples)")
    
    return saved_plots

if __name__ == "__main__":
    main()