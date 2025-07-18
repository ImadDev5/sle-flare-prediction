"""
Script to create grouped horizontal bar charts for machine learning metrics.

This script generates professional-quality horizontal bar charts for AUC, Accuracy,
Precision, Recall, and F1 metrics with bootstrap 95% confidence interval error bars.
TAGT model bars are highlighted with bold outlines for emphasis.

The charts are saved to the figures directory as both PDF and PNG formats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    # Fallback to classic style if seaborn style not available
    plt.style.use('classic')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

try:
    sns.set_palette("husl")
except Exception:
    # Continue without custom palette if seaborn not available
    pass

def load_results_with_ci(results_path: str = "results/results_summary_with_ci.csv") -> pd.DataFrame:
    """
    Load the results DataFrame with bootstrap confidence intervals.
    
    Args:
        results_path: Path to the results CSV file
        
    Returns:
        DataFrame with results and confidence intervals
    """
    results_file = Path(results_path)
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.info("Please run the bootstrap confidence calculation first.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(results_file, index_col='model')
        logger.info(f"Loaded results for {len(df)} models")
        return df
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return pd.DataFrame()

def is_tagt_model(model_name: str) -> bool:
    """
    Check if a model is a TAGT variant.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model is a TAGT variant, False otherwise
    """
    model_lower = model_name.lower()
    tagt_keywords = ['tagt', 'breakthrough', 'ultimate', 'production_tagt']
    return any(keyword in model_lower for keyword in tagt_keywords)

def prepare_data_for_plotting(df: pd.DataFrame, metrics: List[str]) -> Dict:
    """
    Prepare data for plotting horizontal bar charts.
    
    Args:
        df: DataFrame with results and confidence intervals
        metrics: List of metrics to plot
        
    Returns:
        Dictionary with prepared data for each metric
    """
    plot_data = {}
    
    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in data")
            continue
            
        # Filter out models with missing data for this metric
        valid_models = df.dropna(subset=[metric]).copy()
        
        if valid_models.empty:
            logger.warning(f"No valid data for metric {metric}")
            continue
            
        # Sort by metric value (descending for horizontal bars)
        valid_models = valid_models.sort_values(metric, ascending=True)
        
        # Extract metric values and confidence intervals
        values = valid_models[metric].values
        model_names = valid_models.index.tolist()
        
        # Get confidence intervals
        lower_col = f"{metric}_lower"
        upper_col = f"{metric}_upper"
        
        lower_errors = np.zeros(len(values))
        upper_errors = np.zeros(len(values))
        
        if lower_col in valid_models.columns and upper_col in valid_models.columns:
            lower_bounds = valid_models[lower_col].values
            upper_bounds = valid_models[upper_col].values
            
            # Handle NaN values by replacing with original values
            lower_bounds = np.where(np.isnan(lower_bounds), values, lower_bounds)
            upper_bounds = np.where(np.isnan(upper_bounds), values, upper_bounds)
            
            # Calculate error bar lengths
            lower_errors = values - lower_bounds
            upper_errors = upper_bounds - values
            
            # Ensure no negative error bars
            lower_errors = np.maximum(lower_errors, 0)
            upper_errors = np.maximum(upper_errors, 0)
        
        # Identify TAGT models
        is_tagt = [is_tagt_model(name) for name in model_names]
        
        plot_data[metric] = {
            'values': values,
            'model_names': model_names,
            'lower_errors': lower_errors,
            'upper_errors': upper_errors,
            'is_tagt': is_tagt,
            'n_models': len(model_names)
        }
    
    return plot_data

def create_single_metric_chart(metric: str, data: Dict, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a horizontal bar chart for a single metric.
    
    Args:
        metric: Name of the metric
        data: Dictionary with plotting data for the metric
        figsize: Figure size tuple
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    values = data['values']
    model_names = data['model_names']
    lower_errors = data['lower_errors']
    upper_errors = data['upper_errors']
    is_tagt = data['is_tagt']
    n_models = data['n_models']
    
        y_pos = np.arange(n_models)
    
        colors = []
    edge_colors = []
    linewidths = []
    
    for i, tagt in enumerate(is_tagt):
        if tagt:
            colors.append('#FF6B6B')  # Red for TAGT models
            edge_colors.append('black')
            linewidths.append(3)  # Bold outline for TAGT
        else:
            colors.append('#4ECDC4')  # Teal for baseline models
            edge_colors.append('gray')
            linewidths.append(1)
    
        bars = ax.barh(y_pos, values, 
                   color=colors, 
                   edgecolor=edge_colors,
                   linewidth=linewidths,
                   alpha=0.8,
                   height=0.6)
    
    # Add error bars
    if np.any(lower_errors > 0) or np.any(upper_errors > 0):
        ax.errorbar(values, y_pos, 
                   xerr=[lower_errors, upper_errors],
                   fmt='none', 
                   color='black', 
                   capsize=4, 
                   capthick=2,
                   elinewidth=1.5)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ').title() for name in model_names])
    ax.set_xlabel(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Model Performance Comparison: {metric.upper()}\n'
                 f'(with 95% Bootstrap Confidence Intervals)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', 
                ha='left', va='center', 
                fontweight='bold', fontsize=10)
    
    # Add legend
    tagt_patch = patches.Patch(color='#FF6B6B', label='TAGT Models')
    baseline_patch = patches.Patch(color='#4ECDC4', label='Baseline Models')
    ax.legend(handles=[tagt_patch, baseline_patch], 
             loc='lower right', fontsize=12)
    
    # Grid and formatting
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Set x-axis limits with padding
    x_max = max(values) 
    if np.any(upper_errors > 0):
        x_max = max(x_max, np.max(values + upper_errors))
    ax.set_xlim(0, x_max * 1.15)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_grouped_metrics_chart(plot_data: Dict, figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
    """
    Create a grouped horizontal bar chart for all metrics.
    
    Args:
        plot_data: Dictionary with plotting data for all metrics
        figsize: Figure size tuple
        
    Returns:
        Matplotlib Figure object
    """
    metrics = list(plot_data.keys())
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        logger.error("No metrics to plot")
        return None
    
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    
    if n_metrics == 1:
        axes = [axes]
    
    # Get consistent y-axis (all unique model names)
    all_models = set()
    for metric in metrics:
        all_models.update(plot_data[metric]['model_names'])
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = plot_data[metric]
        
        values = data['values']
        model_names = data['model_names']
        lower_errors = data['lower_errors']
        upper_errors = data['upper_errors']
        is_tagt = data['is_tagt']
        n_models = data['n_models']
        
                y_pos = np.arange(n_models)
        
                colors = []
        edge_colors = []
        linewidths = []
        
        for j, tagt in enumerate(is_tagt):
            if tagt:
                colors.append('#FF6B6B')  # Red for TAGT models
                edge_colors.append('black')
                linewidths.append(3)  # Bold outline for TAGT
            else:
                colors.append('#4ECDC4')  # Teal for baseline models
                edge_colors.append('gray')
                linewidths.append(1)
        
                bars = ax.barh(y_pos, values, 
                       color=colors, 
                       edgecolor=edge_colors,
                       linewidth=linewidths,
                       alpha=0.8,
                       height=0.6)
        
        # Add error bars
        if np.any(lower_errors > 0) or np.any(upper_errors > 0):
            ax.errorbar(values, y_pos, 
                       xerr=[lower_errors, upper_errors],
                       fmt='none', 
                       color='black', 
                       capsize=3, 
                       capthick=1.5,
                       elinewidth=1)
        
        # Customize subplot
        ax.set_yticks(y_pos)
        if i == 0:  # Only show y-labels on leftmost subplot
            ax.set_yticklabels([name.replace('_', ' ').title() for name in model_names])
        else:
            ax.set_yticklabels([])
        
        ax.set_xlabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', 
                    ha='left', va='center', 
                    fontweight='bold', fontsize=9)
        
        # Grid and formatting
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        # Set x-axis limits with padding
        x_max = max(values)
        if np.any(upper_errors > 0):
            x_max = max(x_max, np.max(values + upper_errors))
        ax.set_xlim(0, x_max * 1.15)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add main title
    fig.suptitle('Model Performance Comparison Across All Metrics\n'
                 '(with 95% Bootstrap Confidence Intervals)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add legend
    tagt_patch = patches.Patch(color='#FF6B6B', label='TAGT Models')
    baseline_patch = patches.Patch(color='#4ECDC4', label='Baseline Models')
    fig.legend(handles=[tagt_patch, baseline_patch], 
              loc='upper right', fontsize=12, bbox_to_anchor=(0.98, 0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    return fig

def save_plots(fig: plt.Figure, filename: str, output_dir: str = "figures") -> None:
    """
    Save plots in both PDF and PNG formats.
    
    Args:
        fig: Matplotlib Figure object
        filename: Base filename (without extension)
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as PDF
    pdf_path = output_path / f"{filename}.pdf"
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    logger.info(f"Saved PDF: {pdf_path}")
    
    # Save as PNG
    png_path = output_path / f"{filename}.png"
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved PNG: {png_path}")

def main():
    """
    Main function to create and save metric bar charts.
    """
    logger.info("Starting metric bar chart generation...")
    
    # Load data
    df = load_results_with_ci()
    if df.empty:
        logger.error("No data available. Exiting.")
        return
    
    # Define metrics to plot
    metrics = ['auc', 'acc', 'prec', 'recall', 'f1']
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        logger.error("No target metrics found in data")
        return
    
    logger.info(f"Available metrics: {available_metrics}")
    
    # Prepare data for plotting
    plot_data = prepare_data_for_plotting(df, available_metrics)
    
    if not plot_data:
        logger.error("No valid data for plotting")
        return
    
        logger.info("Creating individual metric charts...")
    for metric in available_metrics:
        if metric in plot_data:
            try:
                fig = create_single_metric_chart(metric, plot_data[metric])
                save_plots(fig, f"{metric}_bars", "figures")
                plt.close(fig)
                logger.info(f"Created chart for {metric}")
            except Exception as e:
                logger.error(f"Error creating chart for {metric}: {e}")
    
        logger.info("Creating grouped metrics chart...")
    try:
        fig = create_grouped_metrics_chart(plot_data)
        if fig is not None:
            save_plots(fig, "metric_bars", "figures")
            plt.close(fig)
            logger.info("Created grouped metrics chart")
    except Exception as e:
        logger.error(f"Error creating grouped chart: {e}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("METRIC BAR CHART GENERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total models processed: {len(df)}")
    logger.info(f"Metrics included: {', '.join(available_metrics)}")
    
    # Count TAGT models
    tagt_count = sum(1 for model in df.index if is_tagt_model(model))
    logger.info(f"TAGT models highlighted: {tagt_count}")
    logger.info(f"Baseline models: {len(df) - tagt_count}")
    
    logger.info("\nFiles saved to figures/ directory:")
    logger.info("• Individual metric charts: {metric}_bars.pdf/png")
    logger.info("• Grouped metrics chart: metric_bars.pdf/png")
    
    logger.info("\nFeatures included:")
    logger.info("• Bootstrap 95% confidence interval error bars")
    logger.info("• Bold outlines for TAGT models")
    logger.info("• Color-coded model types")
    logger.info("• Value labels on bars")
    logger.info("• Professional styling")
    
    logger.info("\nMetric bar chart generation completed successfully!")

if __name__ == "__main__":
    main()