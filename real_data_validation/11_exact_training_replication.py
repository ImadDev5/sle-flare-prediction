import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.optimized_tagt import create_optimized_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SLEDataset(Dataset):
    """Exact same dataset class as training."""
    
    def __init__(self, sequences, labels, indices):
        self.sequences = [sequences[i] for i in indices]
        self.labels = labels[indices]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Clinical features (exactly as in training)
        clinical_features = np.zeros(15)
        clinical_features[0] = seq.get('current_sledai', 0)
        clinical_features[1] = seq.get('next_sledai', 0)
        clinical_features[2] = seq.get('current_flare', 0)
        clinical_features[3] = seq.get('next_flare', 0)
        
        return {
            'gene_expression': torch.FloatTensor(seq['expression']).unsqueeze(0),  # Add seq dim
            'clinical_features': torch.FloatTensor(clinical_features),
            'label': torch.FloatTensor([self.labels[idx]])
        }

def evaluate_model_exactly_like_training(model, test_loader, adjacency_tensor, device):
    """Exact same evaluation function as training."""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            gene_expression = batch['gene_expression'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(gene_expression, adjacency_tensor, clinical_features)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(outputs['probabilities'].cpu().numpy().flatten())
    
    # Calculate metrics exactly as in training
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
        
        # Binary predictions with optimal threshold
        threshold = 0.5
        binary_preds = (all_preds > threshold).astype(int)
        
        accuracy = accuracy_score(all_labels, binary_preds)
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
    else:
        auc = accuracy = precision = recall = f1 = 0.0
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    """Replicate training evaluation exactly."""
    logger.info("REPLICATING TRAINING EVALUATION EXACTLY")
    logger.info("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data exactly as in training
    logger.info("Loading data...")
    with open('data/integrated/sequences_real.pkl', 'rb') as f:
        sequences = pickle.load(f)
    
    labels = np.load('data/integrated/labels_real.npy')
    
    adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
    adjacency = adjacency_sparse.toarray().astype(np.float32)
    
    logger.info(f"Data loaded: {len(sequences)} sequences, {labels.shape[0]} labels")
    
        indices = np.arange(len(sequences))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42,
        stratify=labels
    )
    
    logger.info(f"Split created: {len(train_idx)} train, {len(test_idx)} test")
    
        test_dataset = SLEDataset(sequences, labels, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
        adjacency_tensor = torch.FloatTensor(adjacency).unsqueeze(0).to(device)
    
    # Load model exactly as in training
    logger.info("Loading model...")
    if os.path.exists("configs/optimized_tagt_config.json"):
        with open("configs/optimized_tagt_config.json", 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model_architecture': {
                'n_genes': 1000,
                'hidden_dim': 256,
                'num_graph_layers': 3,
                'num_heads': 8,
                'temporal_hidden_dim': 128,
                'clinical_dim': 15,
                'dropout': 0.1
            }
        }
    
    model = create_optimized_model(config)
    
    # Load weights
    model_path = "results/best_optimized_model.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Evaluate exactly as in training
    logger.info("Evaluating model...")
    metrics = evaluate_model_exactly_like_training(model, test_loader, adjacency_tensor, device)
    
    logger.info("RESULTS FROM EXACT TRAINING REPLICATION:")
    logger.info(f"  AUC-ROC: {metrics['auc']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")
    
    # Compare with training logs
    training_auc = 0.9715
    training_acc = 0.8816
    
    logger.info("=" * 50)
    logger.info("COMPARISON WITH TRAINING LOGS:")
    logger.info(f"Training AUC-ROC: {training_auc:.4f}")
    logger.info(f"Current AUC-ROC:  {metrics['auc']:.4f}")
    logger.info(f"Difference:       {metrics['auc'] - training_auc:+.4f}")
    logger.info("")
    logger.info(f"Training Accuracy: {training_acc:.4f}")
    logger.info(f"Current Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Difference:        {metrics['accuracy'] - training_acc:+.4f}")
    
    if abs(metrics['auc'] - training_auc) < 0.01:
        logger.info("✅ PERFECT MATCH! Results match training exactly!")
    elif abs(metrics['auc'] - training_auc) < 0.05:
        logger.info("⚠️ CLOSE MATCH! Results are reasonably close.")
    else:
        logger.info("❌ MISMATCH! Results differ significantly.")
        logger.info("Possible causes:")
        logger.info("1. Model saving bug - wrong epoch saved")
        logger.info("2. Model architecture changed after training")
        logger.info("3. Data preprocessing difference")
        logger.info("4. Random seed difference")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()