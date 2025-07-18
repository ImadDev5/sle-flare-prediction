"""
Demonstration script for creating bar plots with bootstrap confidence interval error bars.

This script shows how to use the bootstrap confidence intervals calculated in step 4
to create publication-quality bar plots with error bars for model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')

def load_results_with_ci():
    """Load the results with bootstrap confidence intervals."""
    results_path = Path("results/results_summary_with_ci.csv")
    if not results_path.exists():
        print("Error: Results file not found. Please run the bootstrap confidence calculation first.")
        return None
    
    df = pd.read_csv(results_path, index_col='model')
    return df

def create_metric_comparison_plot(df, metric='auc', figsize=(12, 8)):
    """
    Create a bar plot comparing models for a specific metric with confidence intervals.
    
    Args:
        df: DataFrame with results and confidence intervals
        metric: Metric to plot ('auc', 'acc', 'f1', 'prec', 'recall', 'spec')
        figsize: Figure size tuple
    """
    # Filter out models without the metric
    valid_models = df.dropna(subset=[metric]).copy()
    
    if valid_models.empty:
        print(f"No valid data found for metric: {metric}")
        return None
    
    # Extract values and confidence intervals
    values = valid_models[metric]
    lower_col = f"{metric}_lower"
    upper_col = f"{metric}_upper"
    
    # Calculate error bars (asymmetric)
    if lower_col in valid_models.columns and upper_col in valid_models.columns:
        lower_errors = values - valid_models[lower_col].fillna(values)
        upper_errors = valid_models[upper_col].fillna(values) - values
        errors = [lower_errors, upper_errors]
        has_ci = True
    else:
        errors = None
        has_ci = False
    
        fig, ax = plt.subplots(figsize=figsize)
    
        bars = ax.bar(range(len(values)), values, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(values))),
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add error bars if available
    if has_ci and errors is not None:
        ax.errorbar(range(len(values)), values, 
                   yerr=errors, fmt='none', 
                   color='black', capsize=5, capthick=2)
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison: {metric.upper()} with 95% Bootstrap Confidence Intervals', 
                fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([name.replace('_', ' ').title() for name in values.index], 
                      rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis limits with some padding
    y_max = values.max()
    if has_ci and upper_col in valid_models.columns:
        y_max = max(y_max, valid_models[upper_col].max())
    ax.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    return fig, ax

def create_comprehensive_comparison_plot(df, metrics=['auc', 'acc', 'f1'], figsize=(15, 10)):
    """
    Create a comprehensive comparison plot with multiple metrics.
    
    Args:
        df: DataFrame with results and confidence intervals
        metrics: List of metrics to include
        figsize: Figure size tuple
    """
    # Filter models that have data for at least one metric
    valid_models = df.dropna(subset=metrics, how='all').copy()
    
    if valid_models.empty:
        print("No valid data found for any metrics")
        return None
    
        n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get data for this metric
        metric_data = valid_models.dropna(subset=[metric])
        
        if metric_data.empty:
            ax.text(0.5, 0.5, f'No data for {metric.upper()}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        values = metric_data[metric]
        lower_col = f"{metric}_lower"
        upper_col = f"{metric}_upper"
        
        # Calculate error bars
        if lower_col in metric_data.columns and upper_col in metric_data.columns:
            lower_errors = values - metric_data[lower_col].fillna(values)
            upper_errors = metric_data[upper_col].fillna(values) - values
            errors = [lower_errors, upper_errors]
        else:
            errors = None
        
                bars = ax.bar(range(len(values)), values,
                     color=plt.cm.Set3(np.linspace(0, 1, len(values))),
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add error bars
        if errors is not None:
            ax.errorbar(range(len(values)), values, yerr=errors, 
                       fmt='none', color='black', capsize=4, capthick=1.5)
        
        # Customize subplot
        ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([name.replace('_', ' ').title() for name in values.index], 
                          rotation=45, ha='right', fontsize=8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Grid and limits
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        y_max = values.max()
        if upper_col in metric_data.columns:
            y_max = max(y_max, metric_data[upper_col].max())
        ax.set_ylim(0, y_max * 1.15)
    
    fig.suptitle('Model Performance Comparison with 95% Bootstrap Confidence Intervals', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig, axes

def create_model_ranking_plot(df, metric='auc', figsize=(12, 8)):
    """
    Create a horizontal bar plot showing model ranking with confidence intervals.
    
    Args:
        df: DataFrame with results and confidence intervals
        metric: Metric to rank by
        figsize: Figure size tuple
    """
    # Filter and sort by metric
    valid_models = df.dropna(subset=[metric]).copy()
    valid_models = valid_models.sort_values(metric, ascending=True)
    
    if valid_models.empty:
        print(f"No valid data found for metric: {metric}")
        return None
    
    values = valid_models[metric]
    lower_col = f"{metric}_lower"
    upper_col = f"{metric}_upper"
    
    # Calculate error bars
    if lower_col in valid_models.columns and upper_col in valid_models.columns:
        left_errors = values - valid_models[lower_col].fillna(values)
        right_errors = valid_models[upper_col].fillna(values) - values
        errors = [left_errors, right_errors]
    else:
        errors = None
    
        fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(values))
    bars = ax.barh(y_pos, values, 
                   color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(values))),
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add error bars
    if errors is not None:
        ax.errorbar(values, y_pos, xerr=errors, 
                   fmt='none', color='black', capsize=4, capthick=1.5)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ').title() for name in values.index])
    ax.set_xlabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Ranking by {metric.upper()} (with 95% Bootstrap CI)', 
                fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    # Grid and formatting
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    x_max = values.max()
    if upper_col in valid_models.columns:
        x_max = max(x_max, valid_models[upper_col].max())
    ax.set_xlim(0, x_max * 1.15)
    
    plt.tight_layout()
    return fig, ax

def main():
    """Main function to demonstrate plotting with bootstrap confidence intervals."""
    print("Creating plots with bootstrap confidence intervals...")
    
    # Load results
    df = load_results_with_ci()
    if df is None:
        return
    
    print(f"Loaded results for {len(df)} models")
    print(f"Available metrics: {[col for col in df.columns if not col.endswith(('_lower', '_upper', 'split_type'))]}")
    
        plots_dir = Path("results/plots")
    plots_dir.mkdir(exist_ok=True)
    
        metrics_to_plot = ['auc', 'acc', 'f1', 'prec', 'recall']
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            try:
                print(f"Creating plot for {metric.upper()}...")
                fig, ax = create_metric_comparison_plot(df, metric)
                if fig is not None:
                    fig.savefig(plots_dir / f'{metric}_comparison_with_ci.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved: {metric}_comparison_with_ci.png")
            except Exception as e:
                print(f"  Error creating {metric} plot: {e}")
    
        try:
        print("Creating comprehensive comparison plot...")
        fig, axes = create_comprehensive_comparison_plot(df, ['auc', 'acc', 'f1'])
        if fig is not None:
            fig.savefig(plots_dir / 'comprehensive_model_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  Saved: comprehensive_model_comparison.png")
    except Exception as e:
        print(f"  Error creating comprehensive plot: {e}")
    
        try:
        print("Creating model ranking plot...")
        fig, ax = create_model_ranking_plot(df, 'auc')
        if fig is not None:
            fig.savefig(plots_dir / 'model_ranking_auc.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  Saved: model_ranking_auc.png")
    except Exception as e:
        print(f"  Error creating ranking plot: {e}")
    
    # Print summary of confidence intervals
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVAL SUMMARY")
    print("="*60)
    
    for metric in ['auc', 'acc', 'f1']:
        if metric in df.columns:
            lower_col = f"{metric}_lower"
            upper_col = f"{metric}_upper"
            
            print(f"\n{metric.upper()} Results:")
            print("-" * 20)
            
            valid_data = df.dropna(subset=[metric])
            for model in valid_data.index:
                value = valid_data.loc[model, metric]
                if lower_col in valid_data.columns and upper_col in valid_data.columns:
                    lower = valid_data.loc[model, lower_col]
                    upper = valid_data.loc[model, upper_col]
                    if not pd.isna(lower) and not pd.isna(upper):
                        print(f"{model:30s}: {value:.4f} [{lower:.4f}, {upper:.4f}]")
                    else:
                        print(f"{model:30s}: {value:.4f} [No CI]")
                else:
                    print(f"{model:30s}: {value:.4f} [No CI]")
    
    print(f"\nAll plots saved to: {plots_dir}")
    print("Bootstrap confidence interval calculation and plotting completed successfully!")

if __name__ == "__main__":
    main()