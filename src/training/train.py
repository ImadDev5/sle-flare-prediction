#!/usr/bin/env python3
"""Main training script for TAGT model"""
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
import pickle
import scipy.sparse as sp

# --- Configuration ---
# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Paths
DATA_DIR = Path("C:/Users/ADMIN/OneDrive/Desktop/SLE/data/processed")
EXPRESSION_PATH = DATA_DIR / "expression_real.csv"
ADJACENCY_PATH = DATA_DIR / "adjacency_real.npz"
GENE_LIST_PATH = DATA_DIR / "gene_list_real.pkl"
SEQUENCES_PATH = Path("C:/Users/ADMIN/OneDrive/Desktop/SLE/data/integrated/sequences_real.pkl")
LABELS_PATH = Path("C:/Users/ADMIN/OneDrive/Desktop/SLE/data/integrated/labels_real.npy")
MODEL_OUTPUT_DIR = Path("models")
METRICS_OUTPUT_DIR = Path("metrics")

# Model & Training Hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 0 # Set to 0 for Windows compatibility
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 64
N_HEADS = 4
DROPOUT = 0.2

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(exist_ok=True)

# --- Memory management ---
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Dataset Class ---
class SLEDataset(Dataset):
    """
    Dataset class for SLE flare prediction.
    Loads and aligns expression data, clinical data, and PPI network information.
    """
    def __init__(self, sequences, labels):
        logging.info("Initializing SLEDataset...")
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sledai_scores = torch.tensor([seq['current_sledai'] for seq in sequences], dtype=torch.float32).unsqueeze(1)
        self.expression_data = torch.tensor([seq['expression'] for seq in sequences], dtype=torch.float32)
        self.n_genes = self.expression_data.shape[1]
        
        logging.info(f"Dataset initialized for {len(self.sequences)} samples.")
        logging.info(f"Expression data shape: {self.expression_data.shape}")
        logging.info(f"Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'expression': self.expression_data[idx],
            'sledai': self.sledai_scores[idx],
            'label': self.labels[idx]
        }

# --- Model Architecture ---
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        B, N, C = input.shape
        h = torch.matmul(input, self.W) # [B, N, out_features]

        # Attention mechanism
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).expand(B, -1, -1)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=-1)
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.bmm(attention, h)

        return nn.functional.elu(h_prime)

class TAGTModel(nn.Module):
    def __init__(self, n_genes, hidden_dim=64, n_heads=4, dropout=0.2):
        super(TAGTModel, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(1, hidden_dim, dropout=dropout, alpha=0.2) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)

        self.out_att = GraphAttentionLayer(hidden_dim * n_heads, 2, dropout=dropout, alpha=0.2)
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2 + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, sledai, adj):
        x_expr = x.unsqueeze(-1)  # Reshape to [B, N, 1]
        x_gat = nn.functional.dropout(x_expr, self.dropout, training=self.training)
        x_gat = torch.cat([att(x_gat, adj) for att in self.attentions], dim=2)
        x_gat = nn.functional.dropout(x_gat, self.dropout, training=self.training)
        x_gat = nn.functional.elu(self.out_att(x_gat, adj))
        
        graph_out = x_gat.mean(1) # Global mean pooling
        
        clinical_out = self.clinical_encoder(sledai)
        
        combined = torch.cat([graph_out, clinical_out], dim=1)
        
        return self.classifier(combined)

# --- Training and Evaluation ---
def save_metrics(metrics, epoch, phase):
    """Saves metrics to a JSON file."""
    metrics_file = METRICS_OUTPUT_DIR / f"{phase}_metrics_epoch_{epoch}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved {phase} metrics for epoch {epoch} to {metrics_file}")

def evaluate_model(model, dataloader, criterion, adj_matrix, epoch, phase='validation'):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch} ({phase})"):
            expression = batch['expression'].to(device)
            sledai = batch['sledai'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(expression, sledai, adj_matrix)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            scores = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    metrics = {
        'epoch': epoch,
        'phase': phase,
        'avg_loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0,
        'mcc': matthews_corrcoef(all_labels, all_preds)
    }
    
    logging.info(f"Epoch {epoch} [{phase.upper()}] - F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}, Loss: {avg_loss:.4f}")
    save_metrics(metrics, epoch, phase)
    
    return metrics['f1_score'], avg_loss

def train_model():
    """Main function to orchestrate the training and evaluation process."""
    logging.info("--- Starting Model Training ---")

    # 1. Load data and create datasets
    logging.info("1. Loading data and creating datasets...")
    with open(SEQUENCES_PATH, 'rb') as f:
        sequences = pickle.load(f)
    labels = np.load(LABELS_PATH)

    # Split indices for train, validation, test
    train_val_indices, test_indices = train_test_split(np.arange(len(sequences)), test_size=0.15, random_state=42, stratify=labels)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.15, random_state=42, stratify=labels[train_val_indices])

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    test_sequences = [sequences[i] for i in test_indices]
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    train_dataset = SLEDataset(train_sequences, train_labels)
    val_dataset = SLEDataset(val_sequences, val_labels)
    test_dataset = SLEDataset(test_sequences, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Load the single adjacency matrix to be used for all samples
    adj_matrix = sp.load_npz(ADJACENCY_PATH).toarray()
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    logging.info(f"Adjacency matrix loaded to device. Shape: {adj_matrix.shape}")

    # 2. Initialize model, optimizer, and loss function
    logging.info("2. Initializing model, optimizer, and loss function...")
    n_genes = train_dataset.n_genes
    model = TAGTModel(n_genes=n_genes, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # 3. Training loop
    logging.info("3. Starting training loop...")
    best_val_f1 = -1
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch in progress_bar:
            expression = batch['expression'].to(device)
            sledai = batch['sledai'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            
            outputs = model(expression, sledai, adj_matrix)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # Debugging prints
            # print(f"Outputs: {outputs.detach().cpu().numpy()}")
            # print(f"Labels: {labels.detach().cpu().numpy()}")
            # print(f"Loss: {loss.item()}")
        
        avg_train_loss = train_loss / len(train_loader)
        logging.info(f"Epoch {epoch} [TRAIN] - Avg Loss: {avg_train_loss:.4f}")

        # Validation
        val_f1, val_loss = evaluate_model(model, val_loader, criterion, adj_matrix, epoch, 'validation')
        scheduler.step(val_loss)

        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "best_model.pt")
            logging.info(f"Epoch {epoch}: New best model saved with F1-score: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            logging.info(f"Epoch {epoch}: No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                break
        
        clear_memory()

    # 4. Final evaluation on test set
    logging.info("4. Final evaluation on test set...")
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "best_model.pt"))
    evaluate_model(model, test_loader, criterion, adj_matrix, EPOCHS, 'test')

if __name__ == "__main__":
    train_model()