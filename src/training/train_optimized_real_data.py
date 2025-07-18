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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from typing import Dict
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.optimized_tagt import create_optimized_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/optimized_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedSLEDataset(Dataset):
    """Memory-efficient SLE dataset loader."""
    
    def __init__(self, sequences, labels, adjacency):
        self.sequences = sequences
        self.labels = labels
        self.adjacency = adjacency  # Keep as numpy array to save memory
        
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
            'label': torch.FloatTensor([label]),
            'idx': idx  # For adjacency lookup
        }

class OptimizedTrainer:
    """Memory-efficient trainer for RTX 3050."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Force CPU for now to avoid CUDA issues
        self.device = torch.device('cpu')  # Will switch to CUDA once working
        logger.info(f"Using device: {self.device}")
        
        # Optimized training parameters for RTX 3050
        training_config = self.config.get('training', {})
        self.batch_size = training_config.get('batch_size', 4)  # Small batch size
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        self.epochs = training_config.get('epochs', 50)  # Reduced epochs for testing
        self.accumulation_steps = training_config.get('accumulation_steps', 8)  # Gradient accumulation
        
        logger.info(f"Training config - Batch: {self.batch_size}, LR: {self.learning_rate}, Epochs: {self.epochs}")
        
    def load_data(self):
        """Load real processed data efficiently."""
        try:
            logger.info("Loading real data...")
            
            with open('data/integrated/sequences_real.pkl', 'rb') as f:
                sequences = pickle.load(f)
            
            labels = np.load('data/integrated/labels_real.npy')
            
            # Load adjacency matrix efficiently
            adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
            adjacency = adjacency_sparse.toarray().astype(np.float32)
            
            logger.info(f"Loaded real data: {len(sequences)} sequences")
            logger.info(f"Adjacency shape: {adjacency.shape}")
            logger.info(f"Flare rate: {np.mean(labels):.2%}")
            
            return OptimizedSLEDataset(sequences, labels, adjacency)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def cross_validate(self):
        """Perform cross-validation training and evaluation."""
        logger.info("Starting cross-validation...")
        
        # Load data
        full_dataset = self.load_data()
        
        # Set up cross-validation
        kfold = StratifiedKFold(n_splits=self.config['cross_validation']['n_splits'],
                                shuffle=self.config['cross_validation']['shuffle'],
                                random_state=self.config['cross_validation']['random_state'])

        results = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for fold, (train_idx, test_idx) in enumerate(kfold.split(full_dataset.sequences, full_dataset.labels)):
            logger.info(f"Fold {fold + 1}:")

            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)

            model = create_optimized_model(self.config)
            model.to(self.device)

            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            criterion = nn.BCEWithLogitsLoss()

            best_fold_auc = 0.0
            adjacency_tensor = torch.FloatTensor(full_dataset.adjacency).to(self.device)

            for epoch in range(self.epochs):
                model.train()
                total_loss = 0.0

                for batch in train_loader:
                    gene_expression = batch['gene_expression'].to(self.device)
                    clinical_features = batch['clinical_features'].to(self.device)
                    labels = batch['label'].to(self.device)

                    optimizer.zero_grad()
                    outputs = model(gene_expression, adjacency_tensor, clinical_features)
                    loss = criterion(outputs['logits'], labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                fold_metrics = self.evaluate_model(model, test_loader, adjacency_tensor)
                logger.info(f"Fold {fold + 1} Epoch {epoch + 1}: Loss: {total_loss:.4f}, AUC: {fold_metrics['auc']:.4f}")

                if fold_metrics['auc'] > best_fold_auc:
                    best_fold_auc = fold_metrics['auc']

            results['auc'].append(best_fold_auc)
            results['accuracy'].append(fold_metrics['accuracy'])
            results['precision'].append(fold_metrics['precision'])
            results['recall'].append(fold_metrics['recall'])
            results['f1'].append(fold_metrics['f1'])

        logger.info(f"CV AUC: {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
        logger.info(f"CV Accuracy: {np.mean(results['accuracy']):.4f}")
        logger.info(f"CV Precision: {np.mean(results['precision']):.4f}")
        logger.info(f"CV Recall: {np.mean(results['recall']):.4f}")
        logger.info(f"CV F1: {np.mean(results['f1']):.4f}")
        return results

    def create_data_loaders(self, dataset):
        """Create optimized data loaders."""
        # Simple train/test split for now
        indices = np.arange(len(dataset))
        labels = dataset.labels
        
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Data loaders created - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return train_loader, test_loader
    
    def train_model(self):
        """Main training function."""
        logger.info("Starting optimized training...")
        
        # Load data
        dataset = self.load_data()
        train_loader, test_loader = self.create_data_loaders(dataset)
        
                model = create_optimized_model(self.config)
        model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Optimizer and loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_test_auc = 0.0
        adjacency_tensor = torch.FloatTensor(dataset.adjacency).to(self.device)
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            for batch_idx, batch in enumerate(train_loader):
                gene_expression = batch['gene_expression'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = model(gene_expression, adjacency_tensor, clinical_features)
                loss = criterion(outputs['logits'], labels)
                loss = loss / self.accumulation_steps  # Normalize for accumulation
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradients
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.accumulation_steps
                num_batches += 1
                
                # Memory cleanup
                del outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Validation
            test_metrics = self.evaluate_model(model, test_loader, adjacency_tensor)
            
            avg_loss = total_loss / num_batches
            test_auc = test_metrics['auc']
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                       f"Loss: {avg_loss:.4f}, "
                       f"Test AUC: {test_auc:.4f}, "
                       f"Test Acc: {test_metrics['accuracy']:.4f}")
            
            # Save best model
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                torch.save(model.state_dict(), 'results/best_optimized_model.pth')
                logger.info(f"New best model saved! AUC: {test_auc:.4f}")
        
        logger.info(f"Training completed! Best AUC: {best_test_auc:.4f}")
        
        # Final evaluation
        model.load_state_dict(torch.load('results/best_optimized_model.pth'))
        final_metrics = self.evaluate_model(model, test_loader, adjacency_tensor)
        
        logger.info("FINAL RESULTS:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Save results
        results = {
            'best_auc': best_test_auc,
            'final_metrics': final_metrics,
            'model_params': total_params,
            'config': self.config
        }
        
        with open('results/optimized_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_model(self, model, test_loader, adjacency_tensor):
        """Evaluate model performance."""
        model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                gene_expression = batch['gene_expression'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(gene_expression, adjacency_tensor, clinical_features)
                
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(outputs['probabilities'].cpu().numpy().flatten())
        
        # Calculate metrics
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
    """Main training function."""
    logger.info("OPTIMIZED TAGT TRAINING")
    logger.info("=" * 60)
    
    # Update config for optimized model
    config_path = "configs/ultimate_tagt_config.json"
    
    # Update config for memory efficiency
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Optimize for RTX 3050
    config['model_architecture'].update({
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 3
    })
    
    config['training'].update({
        'batch_size': 4,
        'epochs': 30,
        'accumulation_steps': 8
    })
    
    # Save optimized config
    optimized_config_path = "configs/optimized_tagt_config.json"
    with open(optimized_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
        trainer = OptimizedTrainer(optimized_config_path)
    results = trainer.train_model()
    
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()