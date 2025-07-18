import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('real_data_validation/logs', exist_ok=True)
os.makedirs('real_data_validation/results', exist_ok=True)
os.makedirs('real_data_validation/figures', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_validation/logs/traditional_on_your_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TraditionalOnYourData:
    """Test traditional models on YOUR actual 1000-dimensional training data."""
    
    def __init__(self):
        self.results_path = Path("real_data_validation/results")
        self.figures_path = Path("real_data_validation/figures")
        
        # Data storage
        self.X_flat = None  # 1000-dimensional flattened data
        self.labels = None
        self.results = {}
        
    def load_your_actual_data(self):
        """Load YOUR actual 1000-dimensional training data."""
        logger.info("Loading YOUR actual 1000-dimensional training data...")
        
        try:
            # Load sequences (temporal data)
            with open('data/integrated/sequences_real.pkl', 'rb') as f:
                sequences = pickle.load(f)
            
            # Load labels
            self.labels = np.load('data/integrated/labels_real.npy')
            
                        self.X_flat = np.array([seq['expression'] for seq in sequences])
            
            logger.info(f"SUCCESS: Loaded YOUR actual training data:")
            logger.info(f"  - Samples: {self.X_flat.shape[0]}")
            logger.info(f"  - Features (genes): {self.X_flat.shape[1]}")
            logger.info(f"  - Flare rate: {np.mean(self.labels):.2%}")
            logger.info(f"  - SLE flares: {np.sum(self.labels)}")
            logger.info(f"  - Controls: {len(self.labels) - np.sum(self.labels)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading YOUR data: {e}")
            return False
    
    def test_traditional_models(self, cv_folds=5):
        """Test traditional models on YOUR 1000-dimensional data."""
        logger.info("Testing traditional models on YOUR 1000-dimensional data...")
        
                models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Testing {model_name} on YOUR 1000-dimensional data...")
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            cv_scores = {
                'auc_roc': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'specificity': []
            }
            
            all_y_true = []
            all_y_pred = []
            all_y_prob = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_flat, self.labels)):
                logger.info(f"  Fold {fold + 1}/{cv_folds}")
                
                X_train, X_val = self.X_flat[train_idx], self.X_flat[val_idx]
                y_train, y_val = self.labels[train_idx], self.labels[val_idx]
                
                # Scale for SVM and Logistic Regression
                if model_name in ['SVM', 'Logistic Regression']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_val_scaled)
                
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_val_scaled)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_prob = model.decision_function(X_val_scaled)
                    # Normalize to [0, 1]
                    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
                else:
                    y_prob = y_pred.astype(float)
                
                # Calculate metrics
                try:
                    auc_roc = roc_auc_score(y_val, y_prob)
                except:
                    auc_roc = 0.5
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                # Calculate specificity
                tn = np.sum((y_val == 0) & (y_pred == 0))
                fp = np.sum((y_val == 0) & (y_pred == 1))
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Store metrics
                cv_scores['auc_roc'].append(auc_roc)
                cv_scores['accuracy'].append(accuracy)
                cv_scores['precision'].append(precision)
                cv_scores['recall'].append(recall)
                cv_scores['f1_score'].append(f1)
                cv_scores['specificity'].append(specificity)
                
                # Store for overall analysis
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred)
                all_y_prob.extend(y_prob)
            
            # Calculate summary statistics
            model_results = {}
            for metric, scores in cv_scores.items():
                model_results[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores,
                    'ci_lower': np.percentile(scores, 2.5),
                    'ci_upper': np.percentile(scores, 97.5)
                }
            
            # Overall metrics
            model_results['overall'] = {
                'auc_roc': roc_auc_score(all_y_true, all_y_prob) if len(set(all_y_true)) > 1 else 0.5,
                'accuracy': accuracy_score(all_y_true, all_y_pred),
                'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
                'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
                'f1_score': f1_score(all_y_true, all_y_pred, zero_division=0)
            }
            
            # Store predictions for visualization
            model_results['predictions'] = {
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'y_prob': all_y_prob
            }
            
            results[model_name] = model_results
            
            logger.info(f"SUCCESS {model_name} Results on YOUR 1000-dimensional data:")
            logger.info(f"  - AUC-ROC: {model_results['auc_roc']['mean']:.4f} ± {model_results['auc_roc']['std']:.4f}")
            logger.info(f"  - Accuracy: {model_results['accuracy']['mean']:.4f} ± {model_results['accuracy']['std']:.4f}")
            logger.info(f"  - F1-Score: {model_results['f1_score']['mean']:.4f} ± {model_results['f1_score']['std']:.4f}")
        
        self.results = results
        return results
    
    def create_performance_visualization(self):
        """Create performance visualization."""
        logger.info("Creating performance visualization...")
        
        # Prepare data for plotting
        models = list(self.results.keys())
        auc_means = [self.results[model]['auc_roc']['mean'] for model in models]
        auc_stds = [self.results[model]['auc_roc']['std'] for model in models]
        acc_means = [self.results[model]['accuracy']['mean'] for model in models]
        acc_stds = [self.results[model]['accuracy']['std'] for model in models]
        f1_means = [self.results[model]['f1_score']['mean'] for model in models]
        f1_stds = [self.results[model]['f1_score']['std'] for model in models]
        
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Traditional Models Performance on YOUR 1000-Dimensional SLE Data', 
                     fontsize=16, fontweight='bold')
        
        # Colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # AUC-ROC
        bars1 = ax1.bar(models, auc_means, yerr=auc_stds, capsize=5, 
                       color=colors, alpha=0.8)
        ax1.set_title('AUC-ROC Performance', fontweight='bold')
        ax1.set_ylabel('AUC-ROC Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars1, auc_means, auc_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Accuracy
        bars2 = ax2.bar(models, acc_means, yerr=acc_stds, capsize=5,
                       color=colors, alpha=0.8)
        ax2.set_title('Accuracy Performance', fontweight='bold')
        ax2.set_ylabel('Accuracy Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars2, acc_means, acc_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # F1-Score
        bars3 = ax3.bar(models, f1_means, yerr=f1_stds, capsize=5,
                       color=colors, alpha=0.8)
        ax3.set_title('F1-Score Performance', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars3, f1_means, f1_stds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'traditional_models_on_your_1000d_data.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance visualization saved")
    
    def generate_results_report(self):
        """Generate comprehensive results report."""
        logger.info("Generating results report...")
        
        # Find best model
        best_auc_model = max(self.results.items(), key=lambda x: x[1]['auc_roc']['mean'])
        best_acc_model = max(self.results.items(), key=lambda x: x[1]['accuracy']['mean'])
        
        report = f"""
# Traditional Models on YOUR Actual 1000-Dimensional SLE Data

## Dataset Information
- **Source**: YOUR actual training data (sequences_real.pkl)
- **Samples**: {self.X_flat.shape[0]} temporal sequences
- **Features**: {self.X_flat.shape[1]} genes (1000-dimensional)
- **SLE Flares**: {np.sum(self.labels)} ({np.mean(self.labels):.1%})
- **Controls**: {len(self.labels) - np.sum(self.labels)} ({100 - np.mean(self.labels)*100:.1%})
- **Validation**: 5-fold cross-validation

## Performance Results

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score | Specificity |
|-------|---------|----------|-----------|--------|----------|-------------|
"""
        
        for model_name, results in self.results.items():
            auc = results['auc_roc']['mean']
            auc_std = results['auc_roc']['std']
            acc = results['accuracy']['mean']
            acc_std = results['accuracy']['std']
            prec = results['precision']['mean']
            prec_std = results['precision']['std']
            rec = results['recall']['mean']
            rec_std = results['recall']['std']
            f1 = results['f1_score']['mean']
            f1_std = results['f1_score']['std']
            spec = results['specificity']['mean']
            spec_std = results['specificity']['std']
            
            report += f"| **{model_name}** | {auc:.3f}±{auc_std:.3f} | {acc:.3f}±{acc_std:.3f} | {prec:.3f}±{prec_std:.3f} | {rec:.3f}±{rec_std:.3f} | {f1:.3f}±{f1_std:.3f} | {spec:.3f}±{spec_std:.3f} |\n"
        
        report += f"""

## Key Findings

### Best Performing Models
- **Best AUC-ROC**: {best_auc_model[0]} ({best_auc_model[1]['auc_roc']['mean']:.4f} ± {best_auc_model[1]['auc_roc']['std']:.4f})
- **Best Accuracy**: {best_acc_model[0]} ({best_acc_model[1]['accuracy']['mean']:.4f} ± {best_acc_model[1]['accuracy']['std']:.4f})

### Clinical Assessment
"""
        
        best_auc = best_auc_model[1]['auc_roc']['mean']
        if best_auc > 0.9:
            assessment = "EXCELLENT - Outstanding clinical utility"
        elif best_auc > 0.8:
            assessment = "GOOD - Strong clinical utility"
        elif best_auc > 0.7:
            assessment = "MODERATE - Acceptable clinical utility"
        else:
            assessment = "LIMITED - Requires improvement for clinical use"
        
        report += f"""
- **Performance Level**: {assessment}
- **Best AUC-ROC**: {best_auc:.4f}
- **Clinical Readiness**: {"Ready for clinical validation" if best_auc > 0.8 else "Needs optimization before clinical use"}

### Baseline for TAGT Comparison
These results provide the correct baseline for comparing YOUR TAGT model:
- **Target to Beat**: {best_auc:.4f} AUC-ROC ({best_auc_model[0]})
- **Dataset**: Same 1000-dimensional data YOUR TAGT was trained on
- **Fair Comparison**: Identical preprocessing and validation methodology

## Next Steps
1. **Fix TAGT Architecture**: Resolve model loading issues
2. **Test TAGT**: Compare against these baseline results
3. **Performance Analysis**: Determine if TAGT outperforms traditional models
4. **Research Paper**: Use these results as proper baselines

---
*This represents the CORRECT baseline comparison for YOUR TAGT model validation.*
"""
        
        # Save report
        with open(self.results_path / "traditional_on_your_1000d_data_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Results report generated")
        
        return report
    
    def run_validation(self):
        """Run the complete validation."""
        logger.info("STARTING TRADITIONAL MODELS VALIDATION ON YOUR 1000-DIMENSIONAL DATA")
        logger.info("=" * 80)
        
        # Load YOUR actual data
        if not self.load_your_actual_data():
            logger.error("Failed to load YOUR data")
            return False
        
        # Test traditional models
        results = self.test_traditional_models()
        
        # Save results
        with open(self.results_path / "traditional_on_your_1000d_data_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
                self.create_performance_visualization()
        
                report = self.generate_results_report()
        
        logger.info("SUCCESS: TRADITIONAL MODELS VALIDATION COMPLETED")
        logger.info("=" * 80)
        
        print("\n" + "="*80)
        print("TRADITIONAL MODELS ON YOUR 1000-DIMENSIONAL DATA - RESULTS")
        print("="*80)
        print(report)
        
        return True

def main():
    """Run traditional models validation on YOUR data."""
    validator = TraditionalOnYourData()
    return validator.run_validation()

if __name__ == "__main__":
    success = main()