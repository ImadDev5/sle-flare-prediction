"""
Compute per-fold predictions and probabilities for baseline models.
This script re-runs baseline models with cross-validation and saves
y_true, y_pred, y_prob for each fold to enable paired statistical testing.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LSTMDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
                features = np.concatenate([
            sequence['expression'],
            [sequence['current_sledai']]
        ])
        return torch.FloatTensor(features), torch.LongTensor([self.labels[idx]])

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Since we don't have true sequences, we'll use a simple approach
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode features
        encoded = self.feature_encoder(x)  # (batch_size, hidden_size)
        
        # Add sequence dimension (simulate temporal data)
        encoded = encoded.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # LSTM forward
        lstm_out, _ = self.lstm(encoded)  # (batch_size, 1, hidden_size)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classify
        output = self.classifier(last_output)
        
        return output

def prepare_data():
    """Load and prepare data for baseline comparisons"""
    print("Loading integrated dataset...")
    
    # Try to load real data first, then fallback to synthetic
    try:
        sequences = pd.read_pickle("data/integrated/sequences_real.pkl")
        labels = np.load("data/integrated/labels_real.npy")
        print("Loaded real data")
        # Real data is already a list of dicts
        if isinstance(sequences, pd.DataFrame):
            sequences = sequences.to_dict('records')
    except FileNotFoundError:
        print("Real data not found, using synthetic data")
        sequences_df = pd.read_pickle("data/integrated/sequences.pkl")
        labels = np.load("data/integrated/labels.npy")
        sequences = sequences_df.to_dict('records')
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Positive samples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    
    # Prepare features for sklearn models
    features = []
    for seq in sequences:
        # Combine expression + clinical features
        feature_vector = np.concatenate([
            seq['expression'],
            [seq['current_sledai']],
            [seq['next_sledai'] - seq['current_sledai']],  # SLEDAI change
            [seq['visit_to'] - seq['visit_from']]  # Visit interval
        ])
        features.append(feature_vector)
    
    X = np.array(features)
    y = labels
    
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y, sequences, labels

def train_sklearn_model_fold(model, X_train, X_test, y_train, y_test, fold_idx, model_name):
    """Train sklearn model for a single fold and save predictions"""
    print(f"Training {model_name} - Fold {fold_idx + 1}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Save fold results
    fold_results = {
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fold_idx': fold_idx,
        'model': model_name
    }
    
    # Calculate metrics for logging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    fold_results['metrics'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    print(f"  Fold {fold_idx + 1} - AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")
    
    return fold_results

def train_lstm_model_fold(sequences_train, sequences_test, y_train, y_test, fold_idx):
    """Train LSTM model for a single fold and save predictions"""
    print(f"Training Simple LSTM - Fold {fold_idx + 1}")
    
        train_dataset = LSTMDataset(sequences_train, y_train)
    test_dataset = LSTMDataset(sequences_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Model setup
    input_size = len(sequences_train[0]['expression']) + 1  # +1 for SLEDAI
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleLSTM(input_size=input_size)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (reduced epochs for efficiency)
    model.train()
    for epoch in range(20):  # Reduced epochs
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
    
    # Evaluation
    model.eval()
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            probs = torch.softmax(outputs, dim=1)
            
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())
    
    # Save fold results
    fold_results = {
        'y_true': y_test,
        'y_pred': np.array(y_pred),
        'y_prob': np.array(y_prob),
        'fold_idx': fold_idx,
        'model': 'Simple LSTM'
    }
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    fold_results['metrics'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    print(f"  Fold {fold_idx + 1} - AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")
    
    return fold_results

def main():
    print("=" * 80)
    print("COMPUTING PER-FOLD PREDICTIONS FOR BASELINE MODELS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
        os.makedirs("results/per_fold", exist_ok=True)
    
    # Load and prepare data
    X, y, sequences, labels = prepare_data()
    
    # Set up cross-validation
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\nUsing {n_splits}-fold cross-validation")
    print(f"Total samples: {len(X)}")
    
    # Define models
    models = {
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'SVM_RBF': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic_Regression': LogisticRegression(
            C=1.0,
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
    }
    
    # Storage for all results
    all_fold_results = {}
    
    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n" + "=" * 50)
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print("=" * 50)
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Split sequences for LSTM
        sequences_train = [sequences[i] for i in train_idx]
        sequences_test = [sequences[i] for i in test_idx]
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Train positive rate: {np.mean(y_train):.3f}, Test positive rate: {np.mean(y_test):.3f}")
        
        # Train sklearn models
        for model_name, model in models.items():
            fold_results = train_sklearn_model_fold(
                model, X_train, X_test, y_train, y_test, fold_idx, model_name
            )
            
            # Save individual fold results
            filename = f"results/per_fold/{model_name}_fold_{fold_idx}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(fold_results, f)
            
            # Store in overall results
            if model_name not in all_fold_results:
                all_fold_results[model_name] = []
            all_fold_results[model_name].append(fold_results)
        
        # Train LSTM model
        lstm_fold_results = train_lstm_model_fold(
            sequences_train, sequences_test, y_train, y_test, fold_idx
        )
        
        # Save LSTM fold results
        filename = f"results/per_fold/Simple_LSTM_fold_{fold_idx}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(lstm_fold_results, f)
        
        # Store in overall results
        if 'Simple_LSTM' not in all_fold_results:
            all_fold_results['Simple_LSTM'] = []
        all_fold_results['Simple_LSTM'].append(lstm_fold_results)
    
    # Compute overall statistics
    print("\n" + "=" * 80)
    print("OVERALL CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    summary_results = {}
    for model_name, fold_results in all_fold_results.items():
        # Calculate mean metrics across folds
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
        mean_metrics = {}
        
        for metric in metrics:
            values = [fold['metrics'][metric] for fold in fold_results]
            mean_metrics[metric] = np.mean(values)
        
        summary_results[model_name] = mean_metrics
        
        print(f"{model_name:<20} {mean_metrics['auc']:<10.3f} {mean_metrics['accuracy']:<10.3f} "
              f"{mean_metrics['precision']:<10.3f} {mean_metrics['recall']:<10.3f} {mean_metrics['f1']:<10.3f}")
    
    # Save summary results
    with open('results/per_fold/summary_results.pkl', 'wb') as f:
        pickle.dump(summary_results, f)
    
    # Save all fold results
    with open('results/per_fold/all_fold_results.pkl', 'wb') as f:
        pickle.dump(all_fold_results, f)
    
    print(f"\n✅ Per-fold results saved to: results/per_fold/")
    print(f"✅ Individual fold files: {model_name}_fold_k.pkl")
    print(f"✅ Summary results: summary_results.pkl")
    print(f"✅ All results: all_fold_results.pkl")
    
    print("\n" + "=" * 80)
    print("PER-FOLD COMPUTATION COMPLETE!")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List all generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk("results/per_fold"):
        for file in files:
            if file.endswith('.pkl'):
                print(f"  - {os.path.join(root, file)}")
    
    return all_fold_results

if __name__ == "__main__":
    results = main()