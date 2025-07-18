"""
TAGT Validation Plan - Phase 1: Baseline Model Implementation
Implement and test the baseline models mentioned in documentation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModels:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_models(self):
        """Initialize baseline models as mentioned in documentation"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ),
            'lstm': SimpleLSTM()  # Will implement below
        }
        
    def evaluate_model(self, model, X, y, model_name):
        """Evaluate a single model with cross-validation"""
        logger.info(f"Evaluating {model_name}...")
        
        # 5-fold cross-validation as mentioned in docs
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Scale features for SVM and Logistic Regression
        if model_name in ['svm', 'logistic_regression']:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Cross-validation scores
        cv_scores = {
            'auc_roc': cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc'),
            'accuracy': cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(model, X_scaled, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(model, X_scaled, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        }
        
        # Calculate mean and std
        results = {}
        for metric, scores in cv_scores.items():
            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
        
        # Fit final model for feature importance (if available)
        model.fit(X_scaled, y)
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_.tolist()
        
        return results
    
    def run_baseline_comparison(self, X, y):
        """Run all baseline models and compare results"""
        logger.info("Running baseline model comparison...")
        
        self.prepare_models()
        
        for model_name, model in self.models.items():
            if model_name == 'lstm':
                # Handle LSTM separately (requires different data format)
                self.results[model_name] = self.evaluate_lstm(X, y)
            else:
                self.results[model_name] = self.evaluate_model(model, X, y, model_name)
        
        return self.results
    
    def evaluate_lstm(self, X, y):
        """Evaluate LSTM model (simplified version)"""
        logger.info("Evaluating LSTM...")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Reshape for LSTM (batch_size, seq_len, features)
        # For now, treat each sample as a sequence of length 1
        X_tensor = X_tensor.unsqueeze(1)
        
        # Simple cross-validation for LSTM
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {'auc_roc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
            
                        model = SimpleLSTM(input_size=X.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # Train
            model.train()
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            for epoch in range(50):  # Quick training
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).squeeze()
                val_probs = val_outputs.numpy()
                val_preds = (val_probs > 0.5).astype(int)
                
                # Calculate metrics
                cv_scores['auc_roc'].append(roc_auc_score(y_val.numpy(), val_probs))
                cv_scores['accuracy'].append(accuracy_score(y_val.numpy(), val_preds))
                cv_scores['precision'].append(precision_score(y_val.numpy(), val_preds, zero_division=0))
                cv_scores['recall'].append(recall_score(y_val.numpy(), val_preds, zero_division=0))
                cv_scores['f1'].append(f1_score(y_val.numpy(), val_preds, zero_division=0))
        
        # Calculate mean and std
        results = {}
        for metric, scores in cv_scores.items():
            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        return results
    
    def generate_comparison_report(self):
        """Generate baseline comparison report"""
        report = f"""
# BASELINE MODEL COMPARISON REPORT
Generated: {datetime.now()}

## Performance Summary (Mean ± Std)

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
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
            f1 = results['f1']['mean']
            f1_std = results['f1']['std']
            
            report += f"| {model_name.replace('_', ' ').title()} | {auc:.3f}±{auc_std:.3f} | {acc:.3f}±{acc_std:.3f} | {prec:.3f}±{prec_std:.3f} | {rec:.3f}±{rec_std:.3f} | {f1:.3f}±{f1_std:.3f} |\n"
        
        report += f"""

## Detailed Results

"""
        
        for model_name, results in self.results.items():
            report += f"""
### {model_name.replace('_', ' ').title()}
- **AUC-ROC**: {results['auc_roc']['mean']:.4f} ± {results['auc_roc']['std']:.4f}
- **Accuracy**: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}
- **Precision**: {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}
- **Recall**: {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}
- **F1-Score**: {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}

"""
        
        # Compare with documented claims
        report += """
## Comparison with Documented Claims

**Documented Claims:**
- Random Forest: 64.8% AUC-ROC
- SVM: 51.9% AUC-ROC  
- LSTM: 40.7% AUC-ROC
- TAGT: 96.3% AUC-ROC

**Actual Results:**
"""
        
        for model_name, results in self.results.items():
            documented_claims = {
                'random_forest': 0.648,
                'svm': 0.519,
                'lstm': 0.407,
                'tagt': 0.963
            }
            
            actual_auc = results['auc_roc']['mean']
            if model_name in documented_claims:
                claimed_auc = documented_claims[model_name]
                difference = actual_auc - claimed_auc
                report += f"- {model_name.replace('_', ' ').title()}: {actual_auc:.3f} (Claimed: {claimed_auc:.3f}, Difference: {difference:+.3f})\n"
        
        return report

class SimpleLSTM(nn.Module):
    """Simple LSTM model for baseline comparison"""
    def __init__(self, input_size=1000, hidden_size=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        
        return out

def create_synthetic_data(n_samples=200, n_features=1000, flare_rate=0.23):
    """Create synthetic data for testing (until real data is available)"""
    logger.info(f"Creating synthetic data: {n_samples} samples, {n_features} features")
    
    np.random.seed(42)
    
        X = np.random.randn(n_samples, n_features)
    
        n_flares = int(n_samples * flare_rate)
    y = np.zeros(n_samples)
    y[:n_flares] = 1
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Add some signal to make the problem learnable
    # Flare samples have higher expression in first 50 genes
    flare_indices = y == 1
    X[flare_indices, :50] += np.random.randn(np.sum(flare_indices), 50) * 0.5
    
    return X, y

def main():
    """Run baseline model validation"""
    logger.info("Starting baseline model validation...")
    
        X, y = create_synthetic_data()
    
    # Run baseline comparison
    baseline_models = BaselineModels()
    results = baseline_models.run_baseline_comparison(X, y)
    
        report = baseline_models.generate_comparison_report()
    
    # Save results
    import os
    os.makedirs('validation_plan/reports', exist_ok=True)
    
    with open('validation_plan/reports/baseline_comparison.md', 'w') as f:
        f.write(report)
    
    with open('validation_plan/reports/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(report)
    return results

if __name__ == "__main__":
    results = main()