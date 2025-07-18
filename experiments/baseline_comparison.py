"""Main training script for TAGT model"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
    
    # Load data
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

def train_sklearn_models(X_train, X_test, y_train, y_test):
    """Train and evaluate sklearn baseline models"""
    print("\n" + "="*60)
    print("TRAINING SKLEARN BASELINE MODELS")
    print("="*60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  AUC-ROC:   {auc:.3f}")
    
    return results

def train_lstm_model(sequences_train, sequences_test, y_train, y_test):
    """Train and evaluate LSTM baseline model"""
    print("\n" + "="*60)
    print("TRAINING LSTM BASELINE MODEL")
    print("="*60)
    
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
    
    print(f"LSTM Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    for epoch in range(30):  # Fewer epochs for baseline
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
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}")
    
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
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    print(f"\nLSTM Results:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main():
    print("="*80)
    print("BASELINE MODEL COMPARISONS FOR SLE FLARE PREDICTION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and prepare data
    X, y, sequences, labels = prepare_data()
    
    # Split data (same split as TAGT model)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split sequences for LSTM
    sequences_train, sequences_test = train_test_split(
        sequences, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Train sklearn models
    sklearn_results = train_sklearn_models(X_train, X_test, y_train, y_test)
    
    # Train LSTM model
    lstm_results = train_lstm_model(sequences_train, sequences_test, y_train, y_test)
    
    # Combine all results
    all_results = {**sklearn_results, 'Simple LSTM': lstm_results}
    
    # Print comparison table
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 80)
    
    for model_name, metrics in all_results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['auc']:<10.3f}")
    
    # Save results
    with open('baseline_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nResults saved to: baseline_results.pkl")
    print("\n" + "="*80)
    print("BASELINE COMPARISONS COMPLETE!")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    results = main()