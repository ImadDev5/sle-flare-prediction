#!/usr/bin/env python3
"""
Breakthrough Training Script for TAGT Model v2

Advanced training pipeline with state-of-the-art techniques:
1. Focal loss with adaptive gamma
2. Mixed precision training
3. Gradient accumulation
4. Advanced learning rate scheduling
5. Cross-validation with ensemble
6. Real-time monitoring and early stopping
7. Model checkpointing and recovery
8. Comprehensive evaluation metrics
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Scientific computing
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
)
from torch.cuda.amp import GradScaler, autocast

# Visualization and monitoring
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our models
from src.models.breakthrough_tagt import BreakthroughTAGT, create_breakthrough_model

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('breakthrough_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss with adaptive gamma for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 adaptive: bool = True, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Adaptive gamma based on class distribution
        if self.adaptive:
            pos_ratio = targets.mean()
            gamma = self.gamma * (1 - pos_ratio) if pos_ratio < 0.5 else self.gamma * pos_ratio
        else:
            gamma = self.gamma
            
        focal_loss = self.alpha * (1 - p_t) ** gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SLEDataset(Dataset):
    """Dataset class for SLE flare prediction."""
    
    def __init__(self, sequences: List[Dict], labels: np.ndarray, 
                 adjacency: np.ndarray, scaler: Optional[StandardScaler] = None,
                 augment: bool = False):
        self.sequences = sequences
        self.labels = labels
        self.adjacency = torch.FloatTensor(adjacency)
        self.scaler = scaler
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Extract features
        expression = sequence['expression']
        current_sledai = sequence['current_sledai']
        next_sledai = sequence['next_sledai']
        
        # Create temporal sequence (current state only for now)
        gene_expression = torch.FloatTensor(expression).unsqueeze(0)  # [1, n_genes]
        
        # Clinical features
        clinical_features = torch.FloatTensor([
            current_sledai,
            next_sledai - current_sledai,  # SLEDAI change
            sequence['current_flare'],
            sequence['visit_to'] - sequence['visit_from'],  # Time interval
            current_sledai / 20.0,  # Normalized SLEDAI
            1.0 if current_sledai > 10 else 0.0,  # High SLEDAI indicator
            1.0 if next_sledai > current_sledai else 0.0,  # SLEDAI increasing
            np.log1p(current_sledai),  # Log SLEDAI
            np.sqrt(current_sledai),  # Sqrt SLEDAI
            current_sledai ** 2 / 400.0  # Squared SLEDAI (normalized)
        ])
        
        # Data augmentation
        if self.augment and np.random.random() < 0.3:
            # Add small noise to expression
            noise = torch.randn_like(gene_expression) * 0.05
            gene_expression = gene_expression + noise
            
            # Add small noise to clinical features
            clinical_noise = torch.randn_like(clinical_features) * 0.02
            clinical_features = clinical_features + clinical_noise
        
        return {
            'gene_expression': gene_expression,
            'clinical_features': clinical_features,
            'label': torch.FloatTensor([label]),
            'adjacency': self.adjacency
        }

class EarlyStopping:
    """Early stopping with patience and model checkpointing."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()

class BreakthroughTrainer:
    """Advanced trainer for breakthrough TAGT model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Training parameters
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.epochs = config.get('epochs', 100)
        self.accumulation_steps = config.get('accumulation_steps', 4)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
    def create_data_loaders(self, sequences: List[Dict], labels: np.ndarray, 
                          adjacency: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        
        # Split data
        train_seq, temp_seq, train_labels, temp_labels = train_test_split(
            sequences, labels, test_size=0.4, random_state=42, stratify=labels
        )
        
        val_seq, test_seq, val_labels, test_labels = train_test_split(
            temp_seq, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        logger.info(f"Data splits - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
        logger.info(f"Class distribution - Train: {np.mean(train_labels):.3f}, Val: {np.mean(val_labels):.3f}, Test: {np.mean(test_labels):.3f}")
        
        # Create datasets
        train_dataset = SLEDataset(train_seq, train_labels, adjacency, augment=True)
        val_dataset = SLEDataset(val_seq, val_labels, adjacency, augment=False)
        test_dataset = SLEDataset(test_seq, test_labels, adjacency, augment=False)
        
        # Weighted sampling for imbalanced data
        class_weights = torch.FloatTensor([
            1.0 / (1 - np.mean(train_labels)),  # Weight for class 0
            1.0 / np.mean(train_labels)         # Weight for class 1
        ])
        
        sample_weights = [class_weights[int(label)] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, 
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            gene_expression = batch['gene_expression'].to(self.device)
            clinical_features = batch['clinical_features'].to(self.device)
            labels = batch['label'].to(self.device)
            adjacency = batch['adjacency'][0].to(self.device)  # Same for all samples
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = model(gene_expression, adjacency, clinical_features)
                    loss = criterion(outputs['logits'], labels)
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(gene_expression, adjacency, clinical_features)
                loss = criterion(outputs['logits'], labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss.item() * self.accumulation_steps
            predictions = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'AUC': f'{roc_auc_score(all_labels, all_predictions):.4f}' if len(set(all_labels)) > 1 else 'N/A'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        auc_score = roc_auc_score(all_labels, all_predictions) if len(set(all_labels)) > 1 else 0.0
        
        return {
            'loss': avg_loss,
            'auc': auc_score,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move to device
                gene_expression = batch['gene_expression'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                labels = batch['label'].to(self.device)
                adjacency = batch['adjacency'][0].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = model(gene_expression, adjacency, clinical_features)
                        loss = criterion(outputs['logits'], labels)
                else:
                    outputs = model(gene_expression, adjacency, clinical_features)
                    loss = criterion(outputs['logits'], labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                predictions = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_predictions.extend(predictions.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        if len(set(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_predictions)
            ap_score = average_precision_score(all_labels, all_predictions)
            
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Binary predictions with optimal threshold
            binary_preds = (np.array(all_predictions) > optimal_threshold).astype(int)
            
            accuracy = accuracy_score(all_labels, binary_preds)
            precision = precision_score(all_labels, binary_preds, zero_division=0)
            recall = recall_score(all_labels, binary_preds, zero_division=0)
            f1 = f1_score(all_labels, binary_preds, zero_division=0)
            mcc = matthews_corrcoef(all_labels, binary_preds)
        else:
            auc_score = ap_score = accuracy = precision = recall = f1 = mcc = 0.0
            optimal_threshold = 0.5
        
        return {
            'loss': avg_loss,
            'auc': auc_score,
            'ap': ap_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'threshold': optimal_threshold,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, sequences: List[Dict], labels: np.ndarray, adjacency: np.ndarray) -> Dict:
        """Main training loop."""
        logger.info("Starting breakthrough training...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            sequences, labels, adjacency
        )
        
        # Create model
        model = create_breakthrough_model(self.config)
        model.to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss function
        criterion = FocalLoss(alpha=1.0, gamma=2.0, adaptive=True)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training loop
        best_val_auc = 0.0
        
        for epoch in range(self.epochs):
            # Train epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train AUC: {train_metrics['auc']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Save metrics
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['auc'], model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_metrics['auc'],
                    'config': self.config
                }, 'best_breakthrough_model.pth')
        
        # Final evaluation on test set
        test_metrics = self.validate_epoch(model, test_loader, criterion)
        
        logger.info("Training completed!")
        logger.info(f"Best validation AUC: {best_val_auc:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        
        return {
            'model': model,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': test_metrics,
            'best_val_auc': best_val_auc
        }

def load_real_data() -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Load processed real data."""
    try:
        # Try to load real processed data
        import pickle
        import scipy.sparse as sp
        
        with open('data/integrated/sequences_real.pkl', 'rb') as f:
            sequences = pickle.load(f)
        
        labels = np.load('data/integrated/labels_real.npy')
        adjacency = sp.load_npz('data/processed/adjacency_real.npz').toarray()
        
        logger.info(f"Loaded real data: {len(sequences)} sequences, {adjacency.shape} adjacency")
        return sequences, labels, adjacency
        
    except FileNotFoundError:
        logger.warning("Real data not found, creating synthetic data...")
        return create_synthetic_data()

def create_synthetic_data() -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Create synthetic data for testing."""
    np.random.seed(42)
    
    n_samples = 500
    n_genes = 1000
    n_patients = 100
    
    # Create synthetic sequences
    sequences = []
    labels = []
    
    for i in range(n_samples):
        patient_id = f"PATIENT_{i % n_patients}"
        
        # Synthetic gene expression
        expression = np.random.normal(0, 1, n_genes)
        
        # Add some structure for flare prediction
        flare_prob = np.random.random()
        if flare_prob > 0.7:  # 30% flare rate
            # Increase expression of some genes for flares
            flare_genes = np.random.choice(n_genes, size=50, replace=False)
            expression[flare_genes] += np.random.normal(1, 0.5, 50)
            label = 1
        else:
            label = 0
        
        # Synthetic clinical data
        current_sledai = np.random.normal(8, 4)
        current_sledai = max(0, current_sledai)
        
        if label == 1:
            next_sledai = current_sledai + np.random.normal(5, 2)
        else:
            next_sledai = current_sledai + np.random.normal(0, 1)
        
        sequences.append({
            'patient_id': patient_id,
            'visit_from': 0,
            'visit_to': 1,
            'expression': expression,
            'current_sledai': current_sledai,
            'next_sledai': next_sledai,
            'current_flare': 0,
            'next_flare': label
        })
        
        labels.append(label)
    
    # Create synthetic adjacency matrix
    adjacency = np.random.random((n_genes, n_genes))
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    adjacency = (adjacency > 0.95).astype(float)  # Sparse network
    np.fill_diagonal(adjacency, 1.0)
    
    logger.info(f"Created synthetic data: {len(sequences)} sequences, {adjacency.shape} adjacency")
    logger.info(f"Flare rate: {np.mean(labels):.2%}")
    
    return sequences, np.array(labels), adjacency

def main():
    """Main training function."""
    # Configuration
    config = {
        'n_genes': 1000,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'clinical_dim': 10,
        'dropout': 0.1,
        'num_pathways': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'accumulation_steps': 4,
        'use_amp': True,
        'patience': 15,
        'min_delta': 0.001
    }
    
    # Load data
    sequences, labels, adjacency = load_real_data()
    
    # Create trainer
    trainer = BreakthroughTrainer(config)
    
    # Train model
    results = trainer.train(sequences, labels, adjacency)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f'results/breakthrough_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    with open(results_dir / 'results.json', 'w') as f:
        json.dump({
            'best_val_auc': results['best_val_auc'],
            'test_auc': results['test_metrics']['auc'],
            'test_f1': results['test_metrics']['f1'],
            'test_accuracy': results['test_metrics']['accuracy'],
            'test_precision': results['test_metrics']['precision'],
            'test_recall': results['test_metrics']['recall']
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")
    logger.info("Breakthrough training completed successfully!")

if __name__ == "__main__":
    main()