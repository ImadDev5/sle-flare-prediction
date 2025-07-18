"""
Verify per-fold results for paired statistical testing.
This script checks that all required y_true, y_pred, y_prob data is available
for each model and each fold to enable paired statistical testing.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_fold_results(model_name, fold_idx):
    """Load results for a specific model and fold."""
    filename = f"results/per_fold/{model_name}_fold_{fold_idx}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def verify_data_structure(data, model_name, fold_idx):
    """Verify that fold data has required structure."""
    required_keys = ['y_true', 'y_pred', 'y_prob', 'fold_idx', 'model', 'metrics']
    
    if not all(key in data for key in required_keys):
        missing = [key for key in required_keys if key not in data]
        print(f"  ❌ Missing keys: {missing}")
        return False
    
    # Check array shapes
    y_true = data['y_true']
    y_pred = data['y_pred'] 
    y_prob = data['y_prob']
    
    if not (len(y_true) == len(y_pred) == len(y_prob)):
        print(f"  ❌ Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, y_prob={len(y_prob)}")
        return False
    
    # Check data types
    if not isinstance(y_true, np.ndarray):
        print(f"  ❌ y_true is not numpy array: {type(y_true)}")
        return False
    
    if not isinstance(y_pred, np.ndarray):
        print(f"  ❌ y_pred is not numpy array: {type(y_pred)}")
        return False
        
    if not isinstance(y_prob, np.ndarray):
        print(f"  ❌ y_prob is not numpy array: {type(y_prob)}")
        return False
    
    # Check value ranges
    if not np.all(np.isin(y_true, [0, 1])):
        print(f"  ❌ y_true contains values other than 0/1: {np.unique(y_true)}")
        return False
    
    if not np.all(np.isin(y_pred, [0, 1])):
        print(f"  ❌ y_pred contains values other than 0/1: {np.unique(y_pred)}")
        return False
    
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        print(f"  ❌ y_prob contains values outside [0,1]: min={y_prob.min():.3f}, max={y_prob.max():.3f}")
        return False
    
    print(f"  ✅ Data structure valid: {len(y_true)} samples")
    return True

def check_fold_consistency(all_results):
    """Check that all folds have the same test samples (same indices)."""
    print("\n📊 CHECKING FOLD CONSISTENCY")
    print("=" * 50)
    
    # Get the first model as reference
    reference_model = list(all_results.keys())[0]
    reference_folds = all_results[reference_model]
    
    for fold_idx in range(len(reference_folds)):
        ref_y_true = reference_folds[fold_idx]['y_true']
        ref_size = len(ref_y_true)
        
        print(f"\nFold {fold_idx}:")
        print(f"  Sample size: {ref_size}")
        print(f"  Positive rate: {np.mean(ref_y_true):.3f}")
        
        # Check that all models have the same y_true for this fold
        consistent = True
        for model_name, model_folds in all_results.items():
            if fold_idx < len(model_folds):
                model_y_true = model_folds[fold_idx]['y_true']
                if not np.array_equal(ref_y_true, model_y_true):
                    print(f"  ❌ {model_name} has different y_true for fold {fold_idx}")
                    consistent = False
                else:
                    print(f"  ✅ {model_name} has consistent y_true")
        
        if consistent:
            print(f"  ✅ All models have consistent test sets for fold {fold_idx}")

def main():
    print("=" * 80)
    print("VERIFYING PER-FOLD RESULTS FOR PAIRED STATISTICAL TESTING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define models to check
    models = ['Random_Forest', 'SVM_RBF', 'Logistic_Regression', 'Simple_LSTM', 'TAGT']
    n_folds = 5
    
    print(f"\nChecking {len(models)} models across {n_folds} folds")
    print(f"Models: {', '.join(models)}")
    
    # Storage for all results
    all_results = {}
    complete_models = []
    
    # Check each model
    for model_name in models:
        print(f"\n📁 CHECKING MODEL: {model_name}")
        print("-" * 40)
        
        model_results = []
        model_complete = True
        
        for fold_idx in range(n_folds):
            print(f"  Fold {fold_idx}:", end=" ")
            
            data = load_fold_results(model_name, fold_idx)
            if data is None:
                print(f"❌ File not found")
                model_complete = False
                continue
            
            if verify_data_structure(data, model_name, fold_idx):
                model_results.append(data)
            else:
                model_complete = False
        
        if model_complete:
            all_results[model_name] = model_results
            complete_models.append(model_name)
            print(f"  ✅ {model_name} complete with {len(model_results)} folds")
        else:
            print(f"  ❌ {model_name} incomplete")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    
    if complete_models:
                summary_data = []
        
        for model_name in complete_models:
            model_results = all_results[model_name]
            
            # Calculate overall metrics
            all_y_true = np.concatenate([fold['y_true'] for fold in model_results])
            all_y_pred = np.concatenate([fold['y_pred'] for fold in model_results])
            all_y_prob = np.concatenate([fold['y_prob'] for fold in model_results])
            
            # Get mean metrics across folds
            fold_aucs = [fold['metrics']['auc'] for fold in model_results]
            fold_accs = [fold['metrics']['accuracy'] for fold in model_results]
            
            summary_data.append({
                'Model': model_name,
                'Total Samples': len(all_y_true),
                'Mean AUC': np.mean(fold_aucs),
                'Std AUC': np.std(fold_aucs),
                'Mean Accuracy': np.mean(fold_accs),
                'Std Accuracy': np.std(fold_accs)
            })
        
                df = pd.DataFrame(summary_data)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Check fold consistency
        if len(complete_models) > 1:
            check_fold_consistency(all_results)
        
        # Paired testing readiness
        print(f"\n🔬 PAIRED STATISTICAL TESTING READINESS")
        print("=" * 50)
        print(f"✅ Complete models: {len(complete_models)}")
        print(f"✅ Folds per model: {n_folds}")
        print(f"✅ Total paired comparisons possible: {len(complete_models) * (len(complete_models) - 1) // 2}")
        
        if len(complete_models) >= 2:
            print(f"\n🎯 Ready for paired statistical tests:")
            print(f"   - DeLong test for AUC comparison")
            print(f"   - McNemar test for accuracy comparison")
            print(f"   - Wilcoxon signed-rank test for performance metrics")
            
            # Show example comparison pairs
            print(f"\nExample comparisons:")
            for i, model1 in enumerate(complete_models[:-1]):
                for model2 in complete_models[i+1:]:
                    print(f"   - {model1} vs {model2}")
        
        # File listing
        print(f"\n📂 GENERATED FILES SUMMARY")
        print("=" * 50)
        print(f"Location: results/per_fold/")
        
        per_fold_files = []
        for model_name in complete_models:
            for fold_idx in range(n_folds):
                filename = f"{model_name}_fold_{fold_idx}.pkl"
                per_fold_files.append(filename)
        
        print(f"Individual fold files: {len(per_fold_files)}")
        for file in sorted(per_fold_files):
            print(f"   - {file}")
        
        print(f"\nSummary files:")
        summary_files = ['all_fold_results.pkl', 'summary_results.pkl', 'tagt_all_fold_results.pkl', 'tagt_summary_results.pkl']
        for file in summary_files:
            if os.path.exists(f"results/per_fold/{file}"):
                print(f"   - {file}")
    
    else:
        print("❌ No complete models found!")
    
    print(f"\n" + "=" * 80)
    print("VERIFICATION COMPLETE!")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results

if __name__ == "__main__":
    results = main()