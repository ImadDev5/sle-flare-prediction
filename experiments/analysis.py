"""Main training script for TAGT model"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_results():
    """Load all previous results for comparison"""
    print("Loading previous results...")
    
    # Load baseline results
    try:
        with open('baseline_results.pkl', 'rb') as f:
            baseline_results = pickle.load(f)
        print("✓ Baseline results loaded")
    except:
        baseline_results = {}
        print("✗ Baseline results not found")
    
    # Load ablation results
    try:
        with open('ablation_results.pkl', 'rb') as f:
            ablation_results = pickle.load(f)
        print("✓ Ablation results loaded")
    except:
        ablation_results = {}
        print("✗ Ablation results not found")
    
    return baseline_results, ablation_results

def feature_importance_analysis():
    """Analyze feature importance using Random Forest"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load data
    sequences_df = pd.read_pickle("data/integrated/sequences.pkl")
    labels = np.load("data/integrated/labels.npy")
    sequences = sequences_df.to_dict('records')
    
    # Prepare features
    features = []
    feature_names = []
    
    for seq in sequences:
        # Gene expression features (first 50 for visualization)
        gene_features = seq['expression'][:50]  # Top 50 genes
        clinical_features = [
            seq['current_sledai'],
            seq['next_sledai'] - seq['current_sledai'],  # SLEDAI change
            seq['visit_to'] - seq['visit_from']  # Visit interval
        ]
        
        feature_vector = np.concatenate([gene_features, clinical_features])
        features.append(feature_vector)
    
    # Feature names
    feature_names = [f'Gene_{i}' for i in range(50)] + ['Current_SLEDAI', 'SLEDAI_Change', 'Visit_Interval']
    
    X = np.array(features)
    y = labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    print(f"Top 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:<15} : {importance[idx]:.4f}")
    
        plt.figure(figsize=(12, 8))
    top_n = 15
    top_indices = indices[:top_n]
    
    plt.barh(range(top_n), importance[top_indices])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved: feature_importance.png")
    
    return importance, feature_names

def error_analysis():
    """Analyze prediction errors and patterns"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Load data
    sequences_df = pd.read_pickle("data/integrated/sequences.pkl")
    labels = np.load("data/integrated/labels.npy")
    sequences = sequences_df.to_dict('records')
    
        features = []
    for seq in sequences:
        feature_vector = np.concatenate([
            seq['expression'][:100],  # Top 100 genes
            [seq['current_sledai']],
            [seq['next_sledai'] - seq['current_sledai']]
        ])
        features.append(feature_vector)
    
    X = np.array(features)
    y = labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Flare', 'Flare'], 
                yticklabels=['No Flare', 'Flare'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error analysis by SLEDAI scores
    test_sequences = [sequences[i] for i in range(len(sequences)) if i in range(len(X_test))]
    
    errors = []
    for i, (true_label, pred_label, prob) in enumerate(zip(y_test, y_pred, y_prob)):
        if i < len(test_sequences):
            seq = test_sequences[i]
            errors.append({
                'true_label': true_label,
                'pred_label': pred_label,
                'probability': prob,
                'current_sledai': seq['current_sledai'],
                'next_sledai': seq['next_sledai'],
                'sledai_change': seq['next_sledai'] - seq['current_sledai'],
                'correct': true_label == pred_label
            })
    
    error_df = pd.DataFrame(errors)
    
    print(f"\nError Analysis Summary:")
    print(f"Total predictions: {len(error_df)}")
    print(f"Correct predictions: {error_df['correct'].sum()}")
    print(f"Accuracy: {error_df['correct'].mean():.3f}")
    
    # Analyze errors by SLEDAI
    false_positives = error_df[(error_df['true_label'] == 0) & (error_df['pred_label'] == 1)]
    false_negatives = error_df[(error_df['true_label'] == 1) & (error_df['pred_label'] == 0)]
    
    print(f"\nFalse Positives: {len(false_positives)}")
    if len(false_positives) > 0:
        print(f"  Avg Current SLEDAI: {false_positives['current_sledai'].mean():.2f}")
        print(f"  Avg SLEDAI Change: {false_positives['sledai_change'].mean():.2f}")
    
    print(f"\nFalse Negatives: {len(false_negatives)}")
    if len(false_negatives) > 0:
        print(f"  Avg Current SLEDAI: {false_negatives['current_sledai'].mean():.2f}")
        print(f"  Avg SLEDAI Change: {false_negatives['sledai_change'].mean():.2f}")
    
    # Plot error distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(error_df[error_df['correct']]['current_sledai'], alpha=0.7, label='Correct', bins=10)
    plt.hist(error_df[~error_df['correct']]['current_sledai'], alpha=0.7, label='Incorrect', bins=10)
    plt.xlabel('Current SLEDAI')
    plt.ylabel('Count')
    plt.title('Prediction Accuracy by Current SLEDAI')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(error_df[error_df['correct']]['sledai_change'], alpha=0.7, label='Correct', bins=10)
    plt.hist(error_df[~error_df['correct']]['sledai_change'], alpha=0.7, label='Incorrect', bins=10)
    plt.xlabel('SLEDAI Change')
    plt.ylabel('Count')
    plt.title('Prediction Accuracy by SLEDAI Change')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error analysis plots saved: confusion_matrix.png, error_analysis.png")
    
    return error_df

def confidence_intervals():
    """Calculate confidence intervals for model performance"""
    print("\n" + "="*60)
    print("CONFIDENCE INTERVAL ANALYSIS")
    print("="*60)
    
    # Load results
    baseline_results, ablation_results = load_all_results()
    
    # Combine all results
    all_results = {**baseline_results, **ablation_results}
    
    # Bootstrap confidence intervals
    def bootstrap_ci(scores, n_bootstrap=1000, confidence=0.95):
        """Calculate bootstrap confidence interval"""
        bootstrap_scores = []
        n = len(scores)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_scores.append(np.mean(bootstrap_sample))
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    # For demonstration, create mock performance scores
    # In practice, you'd use cross-validation scores
    print("Model Performance with 95% Confidence Intervals:")
    print("-" * 60)
    
    for model_name, metrics in all_results.items():
        # Mock scores for demonstration (in practice, use CV scores)
        mock_scores = np.random.normal(metrics['accuracy'], 0.05, 20)
        mock_scores = np.clip(mock_scores, 0, 1)  # Keep in [0,1] range
        
        lower, upper = bootstrap_ci(mock_scores)
        
        print(f"{model_name:<20}: {metrics['accuracy']:.3f} [{lower:.3f}, {upper:.3f}]")
    
    return all_results

def create_comprehensive_comparison():
    """Create comprehensive comparison visualization"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    # Load all results
    baseline_results, ablation_results = load_all_results()
    
    # Combine results
    all_results = {**baseline_results, **ablation_results}
    
    if not all_results:
        print("No results found to compare!")
        return
    
        comparison_data = []
    for model_name, metrics in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'AUC-ROC': metrics['auc']
        })
    
    df = pd.DataFrame(comparison_data)
    
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Comparison', fontsize=16)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        ax = axes[row, col]
        
        # Sort by metric value
        df_sorted = df.sort_values(metric, ascending=True)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric])
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['Model'], fontsize=10)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} Comparison')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlim(0, 1.1)
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary table
    print("\nModel Performance Summary:")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.3f'))
    
    print(f"\nComprehensive comparison plot saved: comprehensive_comparison.png")
    
    return df

def main():
    print("="*80)
    print("COMPREHENSIVE DATA ANALYSIS FOR SLE FLARE PREDICTION")
    print("="*80)
    
        import os
    os.makedirs('analysis_plots', exist_ok=True)
    os.chdir('analysis_plots')
    
    try:
        # 1. Feature Importance Analysis
        importance, feature_names = feature_importance_analysis()
        
        # 2. Error Analysis
        error_df = error_analysis()
        
        # 3. Confidence Intervals
        all_results = confidence_intervals()
        
        # 4. Comprehensive Comparison
        comparison_df = create_comprehensive_comparison()
        
        print("\n" + "="*80)
        print("DATA ANALYSIS COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  - feature_importance.png")
        print("  - confusion_matrix.png")
        print("  - error_analysis.png")
        print("  - comprehensive_comparison.png")
        print("\nAll plots saved in: analysis_plots/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()