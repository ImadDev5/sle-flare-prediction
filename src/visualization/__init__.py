"""
Visualization module for TAGT research project.

This module provides utilities for creating publication-ready plots and figures
for the Temporal Attention-based Graph Transformer (TAGT) model analysis.

Available modules:
- plot_roc: ROC curve comparison plotting with confidence intervals
"""

from .plot_roc import ROCCurveAnalyzer, auto_discover_results_files

__version__ = "1.0.0"
__author__ = "TAGT Research Team"

__all__ = [
    'ROCCurveAnalyzer',
    'auto_discover_results_files'
]