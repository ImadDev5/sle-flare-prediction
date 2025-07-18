"""
Analysis module for model results consolidation and statistical analysis.
"""

from .collect_results import ResultsCollector
from .utils import (
    load_consolidated_results,
    get_model_summary,
    compare_models,
    get_best_performing_models,
    get_cross_validation_results,
    create_parquet_from_csv,
    print_results_overview
)
from .significance import (
    delong_auc_test,
    paired_bootstrap,
    mcnemar_test,
    create_significance_matrix,
    save_significance_results
)

__all__ = [
    'ResultsCollector',
    'load_consolidated_results',
    'get_model_summary',
    'compare_models',
    'get_best_performing_models',
    'get_cross_validation_results',
    'create_parquet_from_csv',
    'print_results_overview',
    'delong_auc_test',
    'paired_bootstrap',
    'mcnemar_test',
    'create_significance_matrix',
    'save_significance_results'
]