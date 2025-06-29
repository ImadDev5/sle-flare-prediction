#!/usr/bin/env python3
"""Cleaned training script for TAGT model (no duplicate definitions)."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import os
import gc
from tqdm import tqdm
import json
from pathlib import Path
import logging

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='w'),
        logging.StreamHandler()
    ]
)

DATA_DIR = Path("D:/SLE_data/processed")
EXPRESSION_PATH = DATA_DIR / "expression_normalized.csv"
CLINICAL_PATH = DATA_DIR / "clinical_data.csv"
ADJACENCY_PATH = DATA_DIR / "ppi/adjacency_matrix.pt"
PROBE_LIST_PATH = DATA_DIR / "ppi/probe_list.csv"
MODEL_OUTPUT_DIR = Path("models")
METRICS_OUTPUT_DIR = Path("metrics")

BATCH_SIZE = 8
NUM_WORKERS = 0  # Windows compatibility
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 128
MAX_GENES = 3000  # adjustable gene subset size for memory
N_HEADS = 4
DROPOUT = 0.1

MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

class GraphAttentionLayer(nn.Module):
    """Memory-efficient Graph Attention Layer (dense adjacency).

    Avoids O(N^2 · F) intermediate tensors by computing attention scores
    via additive attention (LeakyReLU(a^T W h_i + a'^T W h_j)). Still
    allocates an (N×N) attention matrix, but that is acceptable (≈1 GB for
    16 k genes) compared with the previous 60 + GB temporary allocation.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Use multiple attention heads
        self.n_heads = 4
        self.head_dim = out_features // self.n_heads
        self.all_head_size = out_features
        
        # Weight matrix for each head
        self.W = nn.Parameter(torch.zeros(size=(1, self.head_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention vectors for each head
        self.a_src = nn.Parameter(torch.zeros(size=(self.head_dim, 1)))
        self.a_dst = nn.Parameter(torch.zeros(size=(self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # Skip connection
        self.skip_connection = nn.Linear(1, self.all_head_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.all_head_size)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply graph attention mechanism with multiple heads to input features h.
        
        Args:
            h: Input features of shape (n_nodes, 1)  # Each node has 1 expression value
            adj: Adjacency matrix of shape (n_nodes, n_nodes)
            
        Returns:
            Output features of shape (n_nodes, out_features)
        """
        # Apply attention heads
        Wh = torch.matmul(h, self.W)  # Use matmul instead of mm for better broadcasting
        Wh = Wh.unsqueeze(1).expand(-1, self.n_heads, -1)  # (n_nodes, n_heads, head_dim)
        
        f1 = torch.matmul(Wh, self.a_src).squeeze(-1)  # (n_nodes, n_heads)
        f2 = torch.matmul(Wh, self.a_dst).squeeze(-1)  # (n_nodes, n_heads)
        
        # Compute attention scores for each head
        e = self.leakyrelu(f1.unsqueeze(1) + f2.unsqueeze(0))  # (n_nodes, n_nodes, n_heads)
        
        # Mask attention with adjacency matrix
        zero_vec = torch.full_like(e, -9e15)
        attention = torch.where(adj.unsqueeze(-1) > 0, e, zero_vec)  # (n_nodes, n_nodes, n_heads)
        
        # Apply softmax across nodes for each head
        attention = torch.softmax(attention, dim=1)  # (n_nodes, n_nodes, n_heads)
        attention = nn.functional.dropout(attention, self.dropout, self.training)
        
        # Apply attention
        h_prime = torch.bmm(attention.transpose(0, 1), Wh)  # (n_heads, n_nodes, head_dim)
        h_prime = h_prime.transpose(0, 1).contiguous().view(-1, self.all_head_size)  # (n_nodes, out_features)
        
        # Apply skip connection after attention
        skip = self.skip_connection(h)
        h_prime = h_prime + skip
        h_prime = self.layer_norm(h_prime)
        
        return nn.functional.elu(h_prime)

class TAGTModel(nn.Module):
    """Temporal Attention Graph Transformer model for SLE flare prediction.
    
    Processes gene expression data as a graph where each gene is a node with
    a single feature (expression value). Incorporates clinical data and temporal
    attention for improved prediction.
    """
    def __init__(self, n_genes, hidden_dim=128, n_heads=4, dropout=0.1):
        super(TAGTModel, self).__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Gene expression encoder (1 -> hidden_dim)
        self.gene_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head graph attention
        head_dim = hidden_dim // n_heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, head_dim, dropout=dropout)
            for _ in range(n_heads)
        ])
        
        # Clinical feature encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def _encode_sample(self, expr_vec, sledai, adj):
        """Process a single sample through the model.
        
        Args:
            expr_vec: Gene expression vector (n_genes,)
            sledai: SLEDAI score (scalar)
            adj: Adjacency matrix (n_genes, n_genes)
            
        Returns:
            Classification output (2,)
        """
        # Reshape expression vector to have feature dimension
        expr_vec = expr_vec.view(-1, 1)  # (n_genes, 1)
        
        # Encode gene expression
        h = self.gene_encoder(expr_vec)  # (n_genes, hidden_dim)
        
        # Apply multi-head graph attention
        head_outputs = []
        for att in self.attention_heads:
            head_output = att(h, adj)
            head_outputs.append(head_output)
        
        # Concatenate attention heads
        multi_head = torch.cat(head_outputs, dim=1)  # (n_genes, hidden_dim)
        
        # Global pooling to get graph-level representation
        graph_repr = torch.mean(multi_head, dim=0)  # (hidden_dim,)
        
        # Encode clinical features
        clinical_repr = self.clinical_encoder(sledai.view(1, 1)).squeeze(0)  # (hidden_dim,)
        
        # Combine graph and clinical representations
        combined = torch.cat([graph_repr, clinical_repr], dim=0)  # (hidden_dim*2,)
        combined = self.layer_norm(combined)
        
        # Classify
        return self.classifier(combined.unsqueeze(0))  # (1, 2)

    def forward(self, x_batch, sledai_batch, adj):
        """Process a batch of samples.
        
        Args:
            x_batch: Batch of gene expression vectors (batch_size, n_genes)
            sledai_batch: Batch of SLEDAI scores (batch_size,)
            adj: Adjacency matrix (n_genes, n_genes)
            
        Returns:
            Classification outputs (batch_size, 2)
        """
        outputs = []
        for i in range(x_batch.size(0)):
            outputs.append(self._encode_sample(x_batch[i], sledai_batch[i], adj))
        
        return torch.cat(outputs, dim=0)  # (batch_size, 2)

# --- Utility functions ---

def save_metrics(metrics: dict, epoch: int, phase: str):
    path = METRICS_OUTPUT_DIR / f"{phase}_epoch_{epoch}.json"
    with path.open("w") as f:
        json.dump(metrics, f, indent=4)


def evaluate(model, loader, criterion, adj, epoch: int, phase: str = "val", threshold: float | None = None):
    model.eval()
    loss_sum, preds, labels, probs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["expression"].to(device), batch["sledai"].to(device), adj)
            loss = criterion(out, batch["label"].to(device))
            if torch.isnan(loss):
                continue
            loss_sum += loss.item()
            p = torch.softmax(out, 1)[:, 1]
            p = torch.nan_to_num(p, nan=0.0)
            if threshold is None:
                preds.extend(torch.argmax(out, 1).cpu())
            else:
                preds.extend((p >= threshold).long().cpu())
            labels.extend(batch["label"].cpu())
            probs.extend(p.cpu())

    metrics = {
        "epoch": epoch,
        "phase": phase,
        "loss": loss_sum / len(loader),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }
    save_metrics(metrics, epoch, phase)
    if phase != "silent":
        logging.info(f"Epoch {epoch} {phase.upper()} | F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} Loss={metrics['loss']:.3f}")
    return metrics["f1"]

def find_best_threshold(model, loader, adj) -> float:
    """Grid-search F1 score to find best probability threshold."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["expression"].to(device), batch["sledai"].to(device), adj)
            p = torch.softmax(out, 1)[:, 1]
            probs.extend(p.cpu())
            labels.extend(batch["label"].cpu())
    probs = torch.stack(probs).numpy()
    labels = torch.tensor(labels).numpy()
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    logging.info(f"Best validation F1 {best_f1:.3f} at threshold {best_t:.2f}")
    return best_t

# --- Training loop ---

def train_model():
    logging.info("=== Preparing data ===")
    clin_df = pd.read_csv(CLINICAL_PATH)
    indices = np.arange(len(clin_df))
    train_val, test = train_test_split(indices, test_size=0.15, random_state=42, stratify=clin_df["flare"])
    train_idx, val_idx = train_test_split(train_val, test_size=0.15, random_state=42, stratify=clin_df.iloc[train_val]["flare"])

    ds_train = SLEDataset(EXPRESSION_PATH, CLINICAL_PATH, PROBE_LIST_PATH, train_idx)
    ds_val = SLEDataset(EXPRESSION_PATH, CLINICAL_PATH, PROBE_LIST_PATH, val_idx)
    ds_test = SLEDataset(EXPRESSION_PATH, CLINICAL_PATH, PROBE_LIST_PATH, test)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    full_adj = torch.load(ADJACENCY_PATH)
    adj = full_adj[:MAX_GENES, :MAX_GENES].clone()
    adj.fill_diagonal_(1)
    adj = adj.to(device)
    model = TAGTModel(ds_train.n_genes, HIDDEN_DIM, N_HEADS, DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=5, factor=0.5)
    # Compute class weights to handle imbalance
    class_counts = torch.bincount(ds_train.labels)
    class_weights = class_counts.float().sum() / (2 * class_counts.float().clamp(min=1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1, patience = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(dl_train, desc=f"Epoch {epoch}"):
            opt.zero_grad()
            out = model(batch["expression"].to(device), batch["sledai"].to(device), adj)
            loss = criterion(out, batch["label"].to(device))
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch} TRAIN | Loss={epoch_loss/len(dl_train):.3f}")

        val_f1 = evaluate(model, dl_val, criterion, adj, epoch, "val")
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "best_model.pt")
            logging.info("Saved new best model.")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered.")
                break
        clear_memory()

    logging.info("=== Final evaluation with threshold tuning ===")
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "best_model.pt"))
    best_thresh = find_best_threshold(model, dl_val, adj)
    evaluate(model, dl_test, criterion, adj, epoch, "test_tuned", threshold=best_thresh)
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "best_model.pt"))
    evaluate(model, dl_test, criterion, adj, epoch, "test")

if __name__ == "__main__":
    train_model()
