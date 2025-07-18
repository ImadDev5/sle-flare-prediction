import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.makedirs('real_data_validation/logs', exist_ok=True)
os.makedirs('real_data_validation/results', exist_ok=True)
os.makedirs('real_data_validation/figures', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_validation/logs/traditional_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TraditionalModelsValidator:
    """Comprehensive traditional models validation on real data."""
    
    def __init__(self):
        self.results_path = Path("real_data_validation/results")
        self.figures_path = Path("real_data_validation/figures")
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
                log_path = Path("real_data_validation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.models = {}
        self.validation_results = {}
        
    def setup_traditional_models(self):
        """Setup traditional ML models with optimized hyperparameters."""
        logger.info("Setting up traditional ML models...")
        
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'name': 'Random Forest',
                'type': 'tree_based'
            },
            
            'svm': {
                'model': SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                ),
                'name': 'Support Vector Machine',
                'type': 'kernel_based'
            },
            
            'logistic_regression': {
                'model': LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    solver='liblinear',
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ),
                'name': 'Logistic Regression',
                'type': 'linear'
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'name': 'XGBoost',
                'type': 'gradient_boosting'
            }
        }
        
        # Add neural network models
        self.models.update({
            'lstm': {
                'model': SimpleLSTM,
                'name': 'LSTM',
                'type': 'neural_network'
            },
            
            'transformer': {
                'model': SimpleTransformer,
                'name': 'Standard Transformer',
                'type': 'neural_network'
            }
        })
        
        logger.info(f"Setup {len(self.models)} traditional models")
        
    def load_unified_datasets(self):
        """Load unified datasets prepared in Phase 1."""
        logger.info("Loading unified datasets...")
        
        try:
            with open(self.results_path / "unified_datasets.pkl", 'rb') as f:
                datasets = pickle.load(f)
            
            logger.info("Unified datasets loaded successfully")
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading unified datasets: {e}")
            return None
    
    def validate_sklearn_model(self, model_info, X, y, cv_folds=5):
        """Validate sklearn-based models."""
        model = model_info['model']
        model_name = model_info['name']
        
        logger.info(f"Validating {model_name}...")
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Metrics storage
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
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features for SVM and Logistic Regression
            if model_info['type'] in ['kernel_based', 'linear']:
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
        results = {}
        for metric, scores in cv_scores.items():
            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores,
                'ci_lower': np.percentile(scores, 2.5),
                'ci_upper': np.percentile(scores, 97.5)
            }
        
        # Overall metrics
        results['overall'] = {
            'auc_roc': roc_auc_score(all_y_true, all_y_prob) if len(set(all_y_true)) > 1 else 0.5,
            'accuracy': accuracy_score(all_y_true, all_y_pred),
            'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
            'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
            'f1_score': f1_score(all_y_true, all_y_pred, zero_division=0)
        }
        
        # Store predictions
        results['predictions'] = {
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_prob': all_y_prob
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        logger.info(f"{model_name} validation completed:")
        logger.info(f"  - AUC-ROC: {results['auc_roc']['mean']:.4f} ± {results['auc_roc']['std']:.4f}")
        logger.info(f"  - Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
        
        return results
    
    def validate_neural_model(self, model_class, model_name, X, y, cv_folds=5):
        """Validate neural network models."""
        logger.info(f"Validating {model_name}...")
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Metrics storage
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
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
                        input_dim = X.shape[1]
            model = model_class(input_dim=input_dim).to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            
            # Training loop
            model.train()
            for epoch in range(50):  # Quick training
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                y_prob = torch.sigmoid(val_outputs.squeeze()).cpu().numpy()
                y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics (same as sklearn models)
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
        
        # Calculate summary statistics (same as sklearn)
        results = {}
        for metric, scores in cv_scores.items():
            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores,
                'ci_lower': np.percentile(scores, 2.5),
                'ci_upper': np.percentile(scores, 97.5)
            }
        
        # Overall metrics
        results['overall'] = {
            'auc_roc': roc_auc_score(all_y_true, all_y_prob) if len(set(all_y_true)) > 1 else 0.5,
            'accuracy': accuracy_score(all_y_true, all_y_pred),
            'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
            'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
            'f1_score': f1_score(all_y_true, all_y_pred, zero_division=0)
        }
        
        # Store predictions
        results['predictions'] = {
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_prob': all_y_prob
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        logger.info(f"{model_name} validation completed:")
        logger.info(f"  - AUC-ROC: {results['auc_roc']['mean']:.4f} ± {results['auc_roc']['std']:.4f}")
        logger.info(f"  - Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
        
        return results
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation on all traditional models."""
        logger.info("Running comprehensive traditional models validation...")
        
                self.setup_traditional_models()
        
        # Load datasets
        datasets = self.load_unified_datasets()
        if datasets is None:
            logger.error("Failed to load datasets")
            return False
        
        # Validate on each dataset
        validation_results = {}
        
        for dataset_name, data in datasets.items():
            logger.info(f"Validating traditional models on {dataset_name}...")
            
            dataset_results = {}
            
            for model_key, model_info in self.models.items():
                logger.info(f"  - Testing {model_info['name']}...")
                
                try:
                    if model_info['type'] == 'neural_network':
                        results = self.validate_neural_model(
                            model_info['model'], 
                            model_info['name'],
                            data['X'], 
                            data['y']
                        )
                    else:
                        results = self.validate_sklearn_model(
                            model_info,
                            data['X'], 
                            data['y']
                        )
                    
                    if results is not None:
                        dataset_results[model_key] = results
                        dataset_results[model_key]['model_info'] = {
                            'name': model_info['name'],
                            'type': model_info['type']
                        }
                
                except Exception as e:
                    logger.error(f"Error validating {model_info['name']}: {e}")
                    continue
            
            validation_results[dataset_name] = {
                'models': dataset_results,
                'dataset_info': {
                    'n_samples': data['n_samples'],
                    'n_features': data['n_features'],
                    'description': data['description']
                }
            }
        
        self.validation_results = validation_results
        
        # Save results
        with open(self.results_path / "traditional_models_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info("Traditional models validation completed successfully")
        
        return True

# Simple neural network models for comparison
class SimpleLSTM(nn.Module):
    """Simple LSTM for baseline comparison."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Reshape for LSTM (batch_size, seq_len, features)
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        out = self.fc(lstm_out[:, -1, :])
        
        return out

class SimpleTransformer(nn.Module):
    """Simple Transformer for baseline comparison."""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # Reshape and project
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        out = self.classifier(x)
        
        return out

def main():
    """Main traditional models validation function."""
    logger.info("STARTING REAL DATA VALIDATION - PHASE 3: TRADITIONAL MODELS")
    logger.info("=" * 75)
    
    # Initialize validator
    validator = TraditionalModelsValidator()
    
    # Run comprehensive validation
    success = validator.run_comprehensive_validation()
    
    if success:
        logger.info("PHASE 3 COMPLETED SUCCESSFULLY")
        logger.info("=" * 75)
        return True
    else:
        logger.error("PHASE 3 FAILED")
        return False

if __name__ == "__main__":
    success = main()