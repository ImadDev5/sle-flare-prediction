"""
Utility functions for loading per-fold results for statistical testing.
Provides convenient access to y_true, y_pred, y_prob data across all models and folds.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class PerFoldResultsLoader:
    """Easy access to per-fold results for statistical testing."""
    
    def __init__(self, results_dir="results/per_fold"):
        self.results_dir = results_dir
        self.models = ['Random_Forest', 'SVM_RBF', 'Logistic_Regression', 'Simple_LSTM', 'TAGT']
        self.n_folds = 5
        
    def load_model_fold(self, model_name: str, fold_idx: int) -> Optional[Dict]:
        """Load results for a specific model and fold."""
        filename = os.path.join(self.results_dir, f"{model_name}_fold_{fold_idx}.pkl")
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_all_folds(self, model_name: str) -> List[Dict]:
        """Load all folds for a specific model."""
        results = []
        for fold_idx in range(self.n_folds):
            fold_data = self.load_model_fold(model_name, fold_idx)
            if fold_data is not None:
                results.append(fold_data)
        return results
    
    def get_paired_predictions(self, model1: str, model2: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get paired predictions for two models across all folds.
        Returns: (y_true, model1_probs, model2_probs)
        """
        y_true_all = []
        model1_probs_all = []
        model2_probs_all = []
        
        for fold_idx in range(self.n_folds):
            fold1_data = self.load_model_fold(model1, fold_idx)
            fold2_data = self.load_model_fold(model2, fold_idx)
            
            if fold1_data is not None and fold2_data is not None:
                # Verify y_true is consistent
                if np.array_equal(fold1_data['y_true'], fold2_data['y_true']):
                    y_true_all.extend(fold1_data['y_true'])
                    model1_probs_all.extend(fold1_data['y_prob'])
                    model2_probs_all.extend(fold2_data['y_prob'])
                else:
                    print(f"Warning: y_true mismatch in fold {fold_idx} between {model1} and {model2}")
        
        return (np.array(y_true_all), 
                np.array(model1_probs_all), 
                np.array(model2_probs_all))
    
    def get_fold_predictions(self, fold_idx: int) -> Dict[str, Dict]:
        """Get predictions for all models in a specific fold."""
        fold_results = {}
        for model in self.models:
            data = self.load_model_fold(model, fold_idx)
            if data is not None:
                fold_results[model] = {
                    'y_true': data['y_true'],
                    'y_pred': data['y_pred'],
                    'y_prob': data['y_prob'],
                    'metrics': data['metrics']
                }
        return fold_results
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of all model performances."""
        summary_data = []
        
        for model in self.models:
            model_folds = self.load_all_folds(model)
            if len(model_folds) == self.n_folds:
                # Calculate metrics across folds
                fold_aucs = [fold['metrics']['auc'] for fold in model_folds]
                fold_accs = [fold['metrics']['accuracy'] for fold in model_folds]
                fold_f1s = [fold['metrics']['f1'] for fold in model_folds]
                
                summary_data.append({
                    'Model': model,
                    'Mean_AUC': np.mean(fold_aucs),
                    'Std_AUC': np.std(fold_aucs),
                    'Mean_Accuracy': np.mean(fold_accs),
                    'Std_Accuracy': np.std(fold_accs),
                    'Mean_F1': np.mean(fold_f1s),
                    'Std_F1': np.std(fold_f1s),
                    'Complete_Folds': len(model_folds)
                })
        
        return pd.DataFrame(summary_data)
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check which models have complete fold data."""
        availability = {}
        for model in self.models:
            complete_folds = 0
            for fold_idx in range(self.n_folds):
                if self.load_model_fold(model, fold_idx) is not None:
                    complete_folds += 1
            availability[model] = (complete_folds == self.n_folds)
        return availability

# Convenience functions for quick access
def load_comparison_data(model1: str, model2: str, results_dir="results/per_fold") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quick function to get paired predictions for two models."""
    loader = PerFoldResultsLoader(results_dir)
    return loader.get_paired_predictions(model1, model2)

def get_model_summary(results_dir="results/per_fold") -> pd.DataFrame:
    """Quick function to get summary table of all models."""
    loader = PerFoldResultsLoader(results_dir)
    return loader.create_summary_table()

def list_available_models(results_dir="results/per_fold") -> List[str]:
    """List models with complete fold data."""
    loader = PerFoldResultsLoader(results_dir)
    availability = loader.check_data_availability()
    return [model for model, complete in availability.items() if complete]

# Example usage functions
def example_delong_setup():
    """Example of how to set up data for DeLong test."""
    print("Example: Setting up DeLong test for TAGT vs Logistic Regression")
    
    # Load paired predictions
    y_true, tagt_probs, lr_probs = load_comparison_data('TAGT', 'Logistic_Regression')
    
    print(f"Samples: {len(y_true)}")
    print(f"TAGT AUC: {roc_auc_score(y_true, tagt_probs):.3f}")
    print(f"LR AUC: {roc_auc_score(y_true, lr_probs):.3f}")
    print(f"Ready for DeLong test with {len(y_true)} paired observations")
    
    return y_true, tagt_probs, lr_probs

def example_mcnemar_setup():
    """Example of how to set up data for McNemar test."""
    print("Example: Setting up McNemar test for TAGT vs Random Forest")
    
    loader = PerFoldResultsLoader()
    
    # Get binary predictions instead of probabilities
    y_true_all = []
    tagt_pred_all = []
    rf_pred_all = []
    
    for fold_idx in range(5):
        tagt_data = loader.load_model_fold('TAGT', fold_idx)
        rf_data = loader.load_model_fold('Random_Forest', fold_idx)
        
        if tagt_data and rf_data:
            y_true_all.extend(tagt_data['y_true'])
            tagt_pred_all.extend(tagt_data['y_pred'])
            rf_pred_all.extend(rf_data['y_pred'])
    
    y_true = np.array(y_true_all)
    tagt_pred = np.array(tagt_pred_all)
    rf_pred = np.array(rf_pred_all)
    
    print(f"Samples: {len(y_true)}")
    print(f"TAGT Accuracy: {np.mean(tagt_pred == y_true):.3f}")
    print(f"RF Accuracy: {np.mean(rf_pred == y_true):.3f}")
    print(f"Ready for McNemar test with {len(y_true)} paired predictions")
    
    return y_true, tagt_pred, rf_pred

if __name__ == "__main__":
    print("=" * 50)
    print("PER-FOLD RESULTS LOADER UTILITY")
    print("=" * 50)
    
    # Check data availability
    print("\n📊 Model Data Availability:")
    loader = PerFoldResultsLoader()
    availability = loader.check_data_availability()
    for model, complete in availability.items():
        status = "✅ Complete" if complete else "❌ Incomplete"
        print(f"  {model}: {status}")
    
    # Show summary table
    print("\n📈 Model Performance Summary:")
    summary_df = get_model_summary()
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # Show available models for comparison
    available_models = list_available_models()
    print(f"\n🔬 Available for paired testing: {', '.join(available_models)}")
    
    print("\n" + "=" * 50)
    print("Ready for statistical testing!")
    print("=" * 50)