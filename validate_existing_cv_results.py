import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.models.optimized_tagt import create_optimized_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SLEDataset(Dataset):
    """Dataset class matching cross-validation setup."""
    
    def __init__(self, sequences, labels, indices):
        self.sequences = [sequences[i] for i in indices]
        self.labels = labels[indices]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Clinical features
        clinical_features = np.zeros(15)
        clinical_features[0] = seq.get('current_sledai', 0)
        clinical_features[1] = seq.get('next_sledai', 0)
        clinical_features[2] = seq.get('current_flare', 0)
        clinical_features[3] = seq.get('next_flare', 0)
        
        return {
            'gene_expression': torch.FloatTensor(seq['expression']).unsqueeze(0),
            'clinical_features': torch.FloatTensor(clinical_features),
            'label': torch.FloatTensor([self.labels[idx]])
        }

def evaluate_fold_model(model, test_loader, adjacency_tensor, device):
    """Evaluate a fold model."""
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
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
        binary_preds = (all_preds > 0.5).astype(int)
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
    """Validate existing cross-validation results."""
    logger.info("VALIDATING EXISTING CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    with open('data/integrated/sequences_real.pkl', 'rb') as f:
        sequences = pickle.load(f)
    
    labels = np.load('data/integrated/labels_real.npy')
    
    adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
    adjacency = adjacency_sparse.toarray().astype(np.float32)
    adjacency_tensor = torch.FloatTensor(adjacency).unsqueeze(0).to(device)
    
    logger.info(f"Data loaded: {len(sequences)} sequences")
    
    # Load config
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
    
    # Load existing CV results for comparison
    with open('results/cross_validation_results.json', 'r') as f:
        existing_results = json.load(f)
    
    logger.info("EXISTING CV RESULTS:")
    logger.info(f"  AUC-ROC: {existing_results['auc']['mean']:.4f} ± {existing_results['auc']['std']:.4f}")
    logger.info(f"  Accuracy: {existing_results['accuracy']['mean']:.4f} ± {existing_results['accuracy']['std']:.4f}")
    
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Validate each fold
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(sequences, labels)):
        logger.info(f"\nValidating Fold {fold + 1}/5...")
        
        # Check if fold model exists
        fold_model_path = f"results/per_fold/TAGT_fold_{fold}.pkl"
        
        if os.path.exists(fold_model_path):
            try:
                # Load fold model
                model = create_optimized_model(config)
                
                # Load fold-specific weights
                with open(fold_model_path, 'rb') as f:
                    fold_data = pickle.load(f)
                
                if 'model_state_dict' in fold_data:
                    model.load_state_dict(fold_data['model_state_dict'], strict=False)
                elif 'model' in fold_data:
                    model = fold_data['model']
                else:
                    logger.warning(f"Unexpected fold data structure for fold {fold}")
                    continue
                
                model.to(device)
                
                                test_dataset = SLEDataset(sequences, labels, val_idx)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # Evaluate
                metrics = evaluate_fold_model(model, test_loader, adjacency_tensor, device)
                fold_results.append(metrics)
                
                logger.info(f"  Fold {fold + 1} Results:")
                logger.info(f"    AUC-ROC: {metrics['auc']:.4f}")
                logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
                
                # Compare with existing results
                existing_auc = existing_results['auc']['values'][fold]
                existing_acc = existing_results['accuracy']['values'][fold]
                
                logger.info(f"  Comparison with existing:")
                logger.info(f"    AUC diff: {metrics['auc'] - existing_auc:+.4f}")
                logger.info(f"    Acc diff: {metrics['accuracy'] - existing_acc:+.4f}")
                
            except Exception as e:
                logger.error(f"Error validating fold {fold}: {e}")
                continue
        else:
            logger.warning(f"Fold model not found: {fold_model_path}")
    
    if fold_results:
        # Calculate overall statistics
        auc_values = [r['auc'] for r in fold_results]
        acc_values = [r['accuracy'] for r in fold_results]
        
        mean_auc = np.mean(auc_values)
        std_auc = np.std(auc_values)
        mean_acc = np.mean(acc_values)
        std_acc = np.std(acc_values)
        
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS SUMMARY:")
        logger.info(f"  Validated AUC-ROC: {mean_auc:.4f} ± {std_auc:.4f}")
        logger.info(f"  Validated Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        logger.info("\nCOMPARISON WITH EXISTING RESULTS:")
        logger.info(f"  Existing AUC-ROC: {existing_results['auc']['mean']:.4f} ± {existing_results['auc']['std']:.4f}")
        logger.info(f"  Existing Accuracy: {existing_results['accuracy']['mean']:.4f} ± {existing_results['accuracy']['std']:.4f}")
        
        auc_diff = mean_auc - existing_results['auc']['mean']
        acc_diff = mean_acc - existing_results['accuracy']['mean']
        
        logger.info(f"  AUC-ROC Difference: {auc_diff:+.4f}")
        logger.info(f"  Accuracy Difference: {acc_diff:+.4f}")
        
        if abs(auc_diff) < 0.01:
            logger.info("✅ PERFECT MATCH! Validation confirms existing results!")
        elif abs(auc_diff) < 0.05:
            logger.info("✅ GOOD MATCH! Results are very close to existing.")
        else:
            logger.info("⚠️ DIFFERENCE DETECTED! Results differ from existing.")
        
        logger.info("=" * 60)
    else:
        logger.error("No fold models could be validated!")

if __name__ == "__main__":
    main()