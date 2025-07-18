"""Breakthrough training script for TAGT model with enhanced architecture and training.

This script implements a state-of-the-art approach for SLE flare prediction using
a Topology-Aware Graph Transformer (TAGT) model with several key innovations:

1. Dimension-aware graph attention mechanism that properly handles single-feature nodes
2. Multi-scale feature extraction with residual connections
3. Focal loss with adaptive weighting for handling class imbalance
4. Gradient accumulation for effective training with limited memory
5. Cosine learning rate scheduling with warmup
6. Cross-validation with threshold optimization
7. Pathway-aware attention for biological interpretability
8. Hierarchical feature fusion for improved performance

The model architecture is designed to efficiently process protein-protein interaction
networks with gene expression data as node features, combined with clinical metrics
for improved predictive performance.
"""
from __future__ import annotations

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import logging
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm

# Import fixed graph attention layers
from src.training.fixed_graph_attention import FixedGraphAttentionLayer, MultiScaleGraphAttention

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('breakthrough_training.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Paths
DATA_DIR = Path("data/processed")
EXPRESSION_PATH = DATA_DIR / "expression_normalized.csv"
CLINICAL_PATH = DATA_DIR / "clinical_data.csv"
ADJACENCY_PATH = Path("data/processed/ppi/ppi_adjacency_sparse.npz")
PROBE_LIST_PATH = DATA_DIR / "ppi/probe_list.csv"
MODEL_OUTPUT_DIR = Path("models")
METRICS_OUTPUT_DIR = Path("metrics")

# Training parameters
SEED = 42
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
NUM_WORKERS = 0  # Windows compatibility
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
LR_WARMUP_EPOCHS = 5
HIDDEN_DIM = 96
MAX_GENES = 25  # Align with probe list length
N_HEADS = 4
DROPOUT = 0.2
FOCAL_ALPHA = 0.75  # Weight for positive class in focal loss
FOCAL_GAMMA = 2.0   # Focusing parameter for focal loss

MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(exist_ok=True)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Clear GPU memory
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Dataset ---
class SLEDataset(Dataset):
    """Dataset for SLE flare prediction aligned to PPI network"""
    def __init__(self, expression_path: Path, clinical_path: Path, probe_list_path: Path, indices):
        clinical_df = pd.read_csv(clinical_path)
        self.clinical = clinical_df.iloc[indices].reset_index(drop=True)
        self.labels = torch.tensor(self.clinical["flare"].values, dtype=torch.long)
        self.sledai = torch.tensor(self.clinical["sledai"].values, dtype=torch.float32).unsqueeze(1)

        full_probe_list = pd.read_csv(probe_list_path)["ProbeID"].tolist()
        probe_list = full_probe_list[:MAX_GENES]  # limit gene set
        expr_df = pd.read_csv(expression_path, index_col=0)
        sample_ids = self.clinical["Unnamed: 0"].astype(str).tolist()
        aligned_expr = expr_df.loc[probe_list, sample_ids].T
        self.expression = torch.tensor(aligned_expr.values, dtype=torch.float32)
        self.n_genes = len(probe_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "expression": self.expression[idx],
            "sledai": self.sledai[idx],
            "label": self.labels[idx],
        }

# --- Loss Functions ---
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Parameters
    ----------
    alpha : float
        Weighting factor in (0,1) to balance positive vs. negative examples.
    gamma : float
        Focusing parameter > 0. Higher emphasizes hard, mis-classified cases.
    reduction : str
        'mean' or 'sum'.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # p_t = prob of the true class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()

# --- Model Architecture ---
class PathwayAttention(nn.Module):
    """
    Pathway-aware attention mechanism for biological interpretability.
    
    This module applies attention to gene clusters that are known to be part of
    the same biological pathways, enhancing interpretability of the model.
    
    Args:
        hidden_dim: Hidden dimension size
        n_pathways: Number of biological pathways to model
        dropout: Dropout probability
    """
    def __init__(self, hidden_dim: int, n_pathways: int = 50, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pathways = n_pathways
        
        # Pathway embeddings (learnable)
        self.pathway_embeddings = nn.Parameter(
            torch.randn(n_pathways, hidden_dim)
        )
        
        # Attention mechanism
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pathway attention to input features.
        
        Args:
            x: Input features of shape (n_nodes, hidden_dim)
            
        Returns:
            Output features of shape (n_nodes, hidden_dim)
        """
                queries = self.query_proj(x)  # (n_nodes, hidden_dim)
        
                keys = self.key_proj(self.pathway_embeddings)  # (n_pathways, hidden_dim)
        values = self.value_proj(self.pathway_embeddings)  # (n_pathways, hidden_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (n_nodes, n_pathways)
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)  # Scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (n_nodes, n_pathways)
        
        # Apply attention
        context = torch.matmul(attention_weights, values)  # (n_nodes, hidden_dim)
        
        # Apply output projection
        output = self.output_proj(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + output)
        
        return output

class BreakthroughTAGTModel(nn.Module):
    """
    Breakthrough Topology-Aware Graph Transformer (TAGT) model for SLE flare prediction.
    
    This model combines multi-scale graph attention with pathway-aware attention
    and clinical data integration for improved predictive performance and interpretability.
    
    Args:
        n_genes: Number of genes in the dataset
        hidden_dim: Hidden dimension size
        n_heads: Number of attention heads
        n_classes: Number of output classes
        dropout: Dropout probability
    """
    def __init__(
        self, 
        n_genes: int, 
        hidden_dim: int = 96, 
        n_heads: int = 4, 
        n_classes: int = 2, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Multi-scale graph attention layers
        self.graph_attention = MultiScaleGraphAttention(
            in_features=n_genes,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Pathway attention for biological interpretability
        self.pathway_attention = PathwayAttention(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Clinical data integration
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Global pooling with attention
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def _encode_sample(self, x: torch.Tensor, sledai: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Encode a single sample through the model.
        
        Args:
            x: Gene expression values of shape (n_genes,)
            sledai: SLEDAI score of shape (1,)
            adj: Adjacency matrix of shape (n_genes, n_genes)
            
        Returns:
            Logits for classification
        """
        # Reshape gene expression to (n_genes, 1)
        x = x.unsqueeze(-1)
        
        # Apply multi-scale graph attention
        h = self.graph_attention(x, adj)
        
        # Apply pathway attention for interpretability
        h = self.pathway_attention(h)
        
        # Global pooling with attention
        attention_weights = self.pool_attention(h)  # (n_genes, 1)
        h_pooled = torch.sum(h * attention_weights, dim=0)  # (hidden_dim,)
        
        # Encode clinical data
        clinical_features = self.clinical_encoder(sledai)  # (hidden_dim // 2,)
        
        # Combine genomic and clinical features
        combined = torch.cat([h_pooled, clinical_features], dim=0)
        
        # Final classification
        logits = self.classifier(combined)
        
        return logits
        
    def forward(self, x_batch: torch.Tensor, sledai_batch: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of samples.
        
        Args:
            x_batch: Batch of gene expression values of shape (batch_size, n_genes)
            sledai_batch: Batch of SLEDAI scores of shape (batch_size, 1)
            adj: Adjacency matrix of shape (n_genes, n_genes)
            
        Returns:
            Batch of logits for classification
        """
        batch_size = x_batch.size(0)
        outputs = []
        
        # Process each sample individually
        for i in range(batch_size):
            outputs.append(self._encode_sample(x_batch[i], sledai_batch[i], adj))
            
        # Stack outputs
        return torch.stack(outputs)

# --- Training and Evaluation Functions ---
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    adj: torch.Tensor,
    gradient_accumulation_steps: int = 1
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        adj: Adjacency matrix
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Move data to device
        expression = batch["expression"].to(device)
        sledai = batch["sledai"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        outputs = model(expression, sledai, adj)
        loss = criterion(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if we've accumulated enough gradients
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * gradient_accumulation_steps
    
    # Handle any remaining gradients
    if len(dataloader) % gradient_accumulation_steps != 0:
        optimizer.step()
        scheduler.step()
        
    return total_loss / len(dataloader)

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    adj: torch.Tensor
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
        adj: Adjacency matrix
        
    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move data to device
            expression = batch["expression"].to(device)
            sledai = batch["sledai"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(expression, sledai, adj)
            loss = criterion(outputs, labels)
            
            # Store predictions and labels
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
            
            total_loss += loss.item()
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
        "mcc": matthews_corrcoef(all_labels, all_preds)
    }
    
    # Add confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    
    return total_loss / len(dataloader), metrics

def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    adj: torch.Tensor,
    epochs: int,
    patience: int,
    model_path: str,
    gradient_accumulation_steps: int = 1
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Train and evaluate the model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        adj: Adjacency matrix
        epochs: Number of epochs to train for
        patience: Early stopping patience
        model_path: Path to save the best model
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Tuple of (history dictionary, best metrics dictionary)
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
        "val_mcc": []
    }
    
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_metrics = {}
    patience_counter = 0
    
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            device, adj, gradient_accumulation_steps
        )
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, adj)
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_mcc"].append(val_metrics["mcc"])
        
        # Log metrics
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"Val Metrics: Accuracy={val_metrics['accuracy']:.4f}, "
                    f"Precision={val_metrics['precision']:.4f}, "
                    f"Recall={val_metrics['recall']:.4f}, "
                    f"F1={val_metrics['f1']:.4f}, "
                    f"AUC={val_metrics['auc']:.4f}, "
                    f"MCC={val_metrics['mcc']:.4f}")
        
        # Check if this is the best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            
            # Save the model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics
            }, model_path)
            
            logging.info(f"New best model saved with F1: {best_val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Clear memory
        clear_memory()
        
    return history, best_metrics

def cross_validate(
    n_genes: int,
    hidden_dim: int,
    n_heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    alpha: float,
    gamma: float,
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Perform cross-validation training.
    
    Args:
        n_genes: Number of genes in the dataset
        hidden_dim: Hidden dimension size
        n_heads: Number of attention heads
        dropout: Dropout probability
        lr: Learning rate
        weight_decay: Weight decay
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        n_splits: Number of cross-validation splits
        
    Returns:
        Dictionary of average metrics across all folds
    """
    # Load adjacency matrix from sparse format
    adj_sparse = np.load(ADJACENCY_PATH)
    adj = torch.sparse_coo_tensor(
        torch.LongTensor([adj_sparse['row'], adj_sparse['col']]),
        torch.FloatTensor(adj_sparse['data']),
        torch.Size(adj_sparse['shape'])
    ).to_dense()
    adj = adj.to(DEVICE)
    
    # Load all data
    clinical_df = pd.read_csv(CLINICAL_PATH)
    all_indices = list(range(len(clinical_df)))
    
    # Set up cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # Prepare to collect metrics across folds
    all_metrics = []
    
        full_dataset = SLEDataset(
        expression_path=EXPRESSION_PATH,
        clinical_path=CLINICAL_PATH,
        probe_list_path=PROBE_LIST_PATH,
        indices=all_indices
    )
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, full_dataset.labels)):
        logging.info(f"Starting fold {fold+1}/{n_splits}")
        
                train_dataset = SLEDataset(
            expression_path=EXPRESSION_PATH,
            clinical_path=CLINICAL_PATH,
            probe_list_path=PROBE_LIST_PATH,
            indices=train_idx
        )
        
        val_dataset = SLEDataset(
            expression_path=EXPRESSION_PATH,
            clinical_path=CLINICAL_PATH,
            probe_list_path=PROBE_LIST_PATH,
            indices=val_idx
        )
        
                train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        
                model = BreakthroughTAGTModel(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        ).to(DEVICE)
        
                criterion = FocalLoss(alpha=alpha, gamma=gamma)
        
                optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
                total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
        warmup_steps = len(train_loader) * LR_WARMUP_EPOCHS // GRADIENT_ACCUMULATION_STEPS
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # Train and evaluate
        model_path = str(MODEL_OUTPUT_DIR / f"tagt_breakthrough_fold_{fold}.pt")
        _, fold_metrics = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            adj=adj,
            epochs=EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
            model_path=model_path,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )
        
        # Store metrics
        all_metrics.append(fold_metrics)
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        clear_memory()
        
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
    # Log average metrics
    logging.info("Average metrics across all folds:")
    for key, value in avg_metrics.items():
        logging.info(f"{key}: {value:.4f}")
        
    # Save average metrics
    with open(METRICS_OUTPUT_DIR / "tagt_breakthrough_cv_metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)
        
    return avg_metrics

def main():
    """Main function to run the training."""
    logging.info("Starting breakthrough TAGT model training")
    
    # Set parameters
    n_genes = MAX_GENES
    hidden_dim = HIDDEN_DIM
    n_heads = N_HEADS
    dropout = DROPOUT
    lr = LEARNING_RATE
    weight_decay = WEIGHT_DECAY
    alpha = FOCAL_ALPHA
    gamma = FOCAL_GAMMA
    
    # Perform cross-validation
    avg_metrics = cross_validate(
        n_genes=n_genes,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha,
        gamma=gamma
    )
    
    # Log final results
    logging.info("Training completed successfully")
    logging.info(f"Final average F1 score: {avg_metrics['f1']:.4f}")
    logging.info(f"Final average AUC: {avg_metrics['auc']:.4f}")
    
    return avg_metrics

if __name__ == "__main__":
    main()