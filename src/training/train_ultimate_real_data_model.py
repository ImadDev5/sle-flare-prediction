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
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.models.ultimate_tagt import create_ultimate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SLEDataLoader(Dataset):
    """Data loader for SLE datasets"""
    
    def __init__(self, sequences, labels, adjacency):
        self.sequences = sequences
        self.labels = labels
        self.adjacency = torch.FloatTensor(adjacency)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Extract features
        expression = sequence['expression']
        current_sledai = sequence['current_sledai']
        
        # Data format transformation
        expression_tensor = torch.FloatTensor(expression).unsqueeze(0)  # [1, n_genes]
        
                clinical_features = [
            current_sledai,
            sequence.get('next_sledai', current_sledai) - current_sledai,  # SLEDAI change
            sequence.get('current_flare', 0),
            sequence['visit_to'] - sequence['visit_from'],
            current_sledai / 20.0,  # Normalized SLEDAI
            1.0 if current_sledai > 10 else 0.0,  # High SLEDAI indicator
            np.log1p(current_sledai),  # Log SLEDAI
            np.sqrt(current_sledai),  # Sqrt SLEDAI
            current_sledai ** 2 / 400.0,  # Squared SLEDAI (normalized)
            sequence.get('patient_id', 'UNKNOWN').count('_'),  # Patient encoding
            float(sequence.get('visit_from', 0)),  # Visit number
            float(sequence.get('visit_to', 1)),  # Next visit
            np.sin(2 * np.pi * sequence.get('visit_from', 0) / 12),  # Seasonal encoding
            np.cos(2 * np.pi * sequence.get('visit_from', 0) / 12),  # Seasonal encoding
            np.random.normal(0, 0.01)  # Small noise for regularization
        ]
        
        clinical_tensor = torch.FloatTensor(clinical_features)
        
        return {
            'gene_expression': expression_tensor,
            'clinical_features': clinical_tensor,
            'label': torch.FloatTensor([label]),
            'adjacency': self.adjacency
        }

class UltimateTrainer:
    """Trainer class for the Ultimate TAGT model."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 1e-5)
        self.epochs = self.config.get('epochs', 50)
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
    def load_data(self) -> SLEDataLoader:
        """Load real processed data for training."""
        try:
            with open('data/integrated/sequences_real.pkl', 'rb') as f:
                sequences = pickle.load(f)
            
            labels = np.load('data/integrated/labels_real.npy')
            adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
            adjacency = adjacency_sparse.toarray()
            
            logger.info(f"Loaded real data: {len(sequences)} sequences, adjacency shape: {adjacency.shape}")
            logger.info(f"Flare rate: {np.mean(labels):.2%}")
            
            return SLEDataLoader(sequences, labels, adjacency)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_optimizer(self, model: nn.Module):
        """Create optimizer for model training."""
        return optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def train_fold(self, fold: int, train_idx: np.array, val_idx: np.array, full_dataset: SLEDataLoader) -> Dict:
        """Train one fold of the cross-validation."""
        logger.info(f"Starting fold {fold+1} of cross-validation...")
        
                train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        # Initialize model
        model = create_ultimate_model(self.config)
        model.to(self.device)
        
                optimizer = self.create_optimizer(model)
        
        # Training loop
        best_val_auc = 0.0
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            all_train_labels = []
            all_train_preds = []
            
            for batch in train_loader:
                gene_expression = batch['gene_expression'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                labels = batch['label'].to(self.device)
                adjacency = batch['adjacency'][0].to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = model(gene_expression, adjacency, clinical_features)
                    loss = nn.BCEWithLogitsLoss()(outputs['logits'], labels)
                
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                all_train_labels.extend(labels.cpu().numpy().flatten())
                all_train_preds.extend(outputs['probabilities'].cpu().numpy().flatten())
            
            # Validation
            val_metrics = self.validate_epoch(model, val_loader)
            
            # Logging
            val_auc = val_metrics['auc']
            logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                        f"Train Loss: {total_loss/len(train_loader):.4f}, "
                        f"Validation AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                logger.info("New best model, saving...")
                torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
            
        return {'best_val_auc': best_val_auc}
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Dict:
        """Validate model on validation set."""
        model.eval()
        all_val_labels = []
        all_val_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                gene_expression = batch['gene_expression'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                labels = batch['label'].to(self.device)
                adjacency = batch['adjacency'][0].to(self.device)
                
                outputs = model(gene_expression, adjacency, clinical_features)
                all_val_labels.extend(labels.cpu().numpy().flatten())
                all_val_preds.extend(outputs['probabilities'].cpu().numpy().flatten())
        
        auc = roc_auc_score(all_val_labels, all_val_preds) if len(set(all_val_labels)) > 1 else 0.0
        
        return {'auc': auc}

    def cross_validate(self) -> None:
        """Perform cross-validation training."""
        # Load data
        full_dataset = self.load_data()
        
        kfold = StratifiedKFold(n_splits=self.config.get('n_splits', 5), shuffle=True, random_state=42)
        
        results = {}
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X=full_dataset.sequences, y=full_dataset.labels)):
            fold_result = self.train_fold(fold, train_idx, val_idx, full_dataset)
            results[f'fold_{fold+1}'] = fold_result
        
        logger.info(f"Cross-validation results: {results}")
        
if __name__ == "__main__":
    # Load configuration
    config_path = "configs/ultimate_tagt_config.json"
    
        trainer = UltimateTrainer(config_path)
    trainer.cross_validate()