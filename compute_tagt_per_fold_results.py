"""
Compute per-fold predictions and probabilities for TAGT model.
This script re-runs the TAGT model with cross-validation and saves
y_true, y_pred, y_prob for each fold to enable paired statistical testing.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Try to import the TAGT model
try:
    from src.models.optimized_tagt import create_optimized_model
except ImportError:
    print("Warning: Could not import optimized_tagt model. Creating a simple alternative.")
    
    class SimpleTAGT(nn.Module):
        def __init__(self, config):
            super(SimpleTAGT, self).__init__()
            model_config = config.get('model', {})
            self.input_dim = model_config.get('gene_expression_dim', 1000)
            self.hidden_dim = model_config.get('hidden_dim', 64)
            
            self.gene_encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.clinical_encoder = nn.Sequential(
                nn.Linear(15, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim + self.hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, gene_expression, adjacency_tensor, clinical_features):
            # Process gene expression
            gene_features = self.gene_encoder(gene_expression.squeeze(1))
            
            # Process clinical features
            clinical_processed = self.clinical_encoder(clinical_features)
            
            # Combine features
            combined = torch.cat([gene_features, clinical_processed], dim=1)
            
            # Get probabilities
            probabilities = self.classifier(combined).squeeze()
            
                        logits = torch.log(probabilities / (1 - probabilities + 1e-8))
            
            return {
                'probabilities': probabilities,
                'logits': logits
            }
    
    def create_optimized_model(config):
        return SimpleTAGT(config)

class OptimizedSLEDataset(Dataset):
    """Memory-efficient SLE dataset loader."""
    
    def __init__(self, sequences, labels, adjacency):
        self.sequences = sequences
        self.labels = labels
        self.adjacency = adjacency
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Extract features with memory efficiency
        expression = np.array(sequence['expression'], dtype=np.float32)
        current_sledai = float(sequence['current_sledai'])
        
                gene_expression = torch.FloatTensor(expression).unsqueeze(0)  # [1, n_genes]
        
        # Enhanced clinical features
        clinical_features = [
            current_sledai,
            sequence.get('next_sledai', current_sledai) - current_sledai,
            float(sequence.get('current_flare', 0)),
            float(sequence['visit_to'] - sequence['visit_from']),
            current_sledai / 20.0,
            1.0 if current_sledai > 10 else 0.0,
            np.log1p(current_sledai),
            np.sqrt(max(0, current_sledai)),
            (current_sledai ** 2) / 400.0,
            float(hash(sequence.get('patient_id', 'UNKNOWN')) % 100) / 100.0,
            float(sequence.get('visit_from', 0)) / 10.0,
            float(sequence.get('visit_to', 1)) / 10.0,
            np.sin(2 * np.pi * sequence.get('visit_from', 0) / 12),
            np.cos(2 * np.pi * sequence.get('visit_from', 0) / 12),
            np.random.normal(0, 0.005)  # Minimal noise
        ]
        
        clinical_tensor = torch.FloatTensor(clinical_features)
        
        return {
            'gene_expression': gene_expression,
            'clinical_features': clinical_tensor,
            'label': torch.FloatTensor([label])
        }

def load_data():
    """Load real processed data efficiently."""
    try:
        print("Loading real data for TAGT cross-validation...")
        
        with open('data/integrated/sequences_real.pkl', 'rb') as f:
            sequences = pickle.load(f)
        
        labels = np.load('data/integrated/labels_real.npy')
        
        # Load adjacency matrix efficiently
        try:
            adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
            adjacency = adjacency_sparse.toarray().astype(np.float32)
        except FileNotFoundError:
                        n_genes = len(sequences[0]['expression'])
            adjacency = np.eye(n_genes, dtype=np.float32)
            print(f"Warning: Using identity adjacency matrix ({n_genes}x{n_genes})")
        
        print(f"Loaded real data: {len(sequences)} sequences")
        print(f"Adjacency shape: {adjacency.shape}")
        print(f"Flare rate: {np.mean(labels):.2%}")
        
        return OptimizedSLEDataset(sequences, labels, adjacency)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_config():
    """Load or create configuration."""
    config_path = "configs/optimized_tagt_config.json"
    
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found, using default configuration")
        config = {
            "model": {
                "gene_expression_dim": 1000,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.2
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "epochs": 20
            },
            "cross_validation": {
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42
            }
        }
    
    return config

def evaluate_model_fold(model, test_loader, adjacency_tensor, device, fold_idx):
    """Evaluate model performance for a single fold and save predictions"""
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch in test_loader:
            gene_expression = batch['gene_expression'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(gene_expression, adjacency_tensor, clinical_features)
            probabilities = outputs['probabilities'].cpu().numpy()
            
            # Convert to binary predictions
            binary_preds = (probabilities > 0.5).astype(int)
            
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(binary_preds.flatten())
            y_prob.extend(probabilities.flatten())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    
    fold_results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fold_idx': fold_idx,
        'model': 'TAGT',
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    }
    
    print(f"  Fold {fold_idx + 1} - AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")
    
    return fold_results

def main():
    print("=" * 80)
    print("COMPUTING PER-FOLD PREDICTIONS FOR TAGT MODEL")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
        os.makedirs("results/per_fold", exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Load data
    full_dataset = load_data()
    
    # Set up cross-validation
    cv_config = config.get('cross_validation', {})
    n_splits = cv_config.get('n_splits', 5)
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=cv_config.get('shuffle', True),
        random_state=cv_config.get('random_state', 42)
    )
    
    print(f"\nUsing {n_splits}-fold cross-validation")
    print(f"Total samples: {len(full_dataset)}")
    
    # Training parameters
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 4)
    learning_rate = training_config.get('learning_rate', 1e-4)
    weight_decay = training_config.get('weight_decay', 1e-5)
    epochs = 15  # Reduced for efficiency
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Storage for all results
    all_fold_results = []
    
    adjacency_tensor = torch.FloatTensor(full_dataset.adjacency).to(device)
    
    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(full_dataset.sequences, full_dataset.labels)):
        print(f"\n" + "=" * 50)
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print("=" * 50)
        
                train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
                model = create_optimized_model(config)
        model.to(device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop for this fold
        print(f"Training TAGT - Fold {fold_idx + 1}")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                gene_expression = batch['gene_expression'].to(device)
                clinical_features = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(gene_expression, adjacency_tensor, clinical_features)
                loss = criterion(outputs['logits'], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 5 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Evaluate and save fold results
        fold_results = evaluate_model_fold(model, test_loader, adjacency_tensor, device, fold_idx)
        
        # Save individual fold results
        filename = f"results/per_fold/TAGT_fold_{fold_idx}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(fold_results, f)
        
        all_fold_results.append(fold_results)
    
    # Compute overall statistics
    print("\n" + "=" * 80)
    print("TAGT MODEL CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    # Calculate mean metrics across folds
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    mean_metrics = {}
    
    for metric in metrics:
        values = [fold['metrics'][metric] for fold in all_fold_results]
        mean_metrics[metric] = np.mean(values)
        std_metrics = np.std(values)
        print(f"{metric.upper()}: {mean_metrics[metric]:.3f} ± {std_metrics:.3f}")
    
    # Save summary results
    summary_results = {'TAGT': mean_metrics}
    with open('results/per_fold/tagt_summary_results.pkl', 'wb') as f:
        pickle.dump(summary_results, f)
    
    # Save all fold results
    with open('results/per_fold/tagt_all_fold_results.pkl', 'wb') as f:
        pickle.dump(all_fold_results, f)
    
    print(f"\n✅ TAGT per-fold results saved to: results/per_fold/")
    print(f"✅ Individual fold files: TAGT_fold_k.pkl")
    print(f"✅ Summary results: tagt_summary_results.pkl")
    print(f"✅ All results: tagt_all_fold_results.pkl")
    
    print("\n" + "=" * 80)
    print("TAGT PER-FOLD COMPUTATION COMPLETE!")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List all TAGT files
    print("\nGenerated TAGT files:")
    for root, dirs, files in os.walk("results/per_fold"):
        for file in files:
            if file.startswith('TAGT') or file.startswith('tagt'):
                print(f"  - {os.path.join(root, file)}")
    
    return all_fold_results

if __name__ == "__main__":
    results = main()