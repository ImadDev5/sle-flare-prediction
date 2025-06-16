#!/usr/bin/env python3
"""Main training script for TAGT model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import os

class SLEDataset(Dataset):
    def __init__(self, sequences, labels, adjacency_matrix):
        self.sequences = sequences
        self.labels = labels
        self.adjacency_matrix = torch.FloatTensor(adjacency_matrix)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        expression = torch.FloatTensor(sequence['expression'])
        current_sledai = torch.FloatTensor([sequence['current_sledai']])
        label = torch.LongTensor([self.labels[idx]])
        
        return {
            'expression': expression,
            'current_sledai': current_sledai,
            'adjacency_matrix': self.adjacency_matrix,
            'label': label
        }

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, h, adj):
        batch_size = h.size(0)
        n_nodes = h.size(1) if len(h.shape) == 3 else adj.size(0)

        if len(h.shape) == 2:  # Single sample
            Wh = torch.mm(h, self.W)
            e = self._prepare_attentional_mechanism_input(Wh)

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = torch.softmax(attention, dim=1)
            attention = torch.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)

            return torch.relu(h_prime)
        else:  # Batch processing
            outputs = []
            for i in range(batch_size):
                Wh = torch.mm(h[i], self.W)
                e = self._prepare_attentional_mechanism_input(Wh)

                zero_vec = -9e15*torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
                attention = torch.softmax(attention, dim=1)
                attention = torch.dropout(attention, self.dropout, training=self.training)
                h_prime = torch.matmul(attention, Wh)
                outputs.append(torch.relu(h_prime))

            return torch.stack(outputs)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return torch.mm(all_combinations_matrix, self.a).view(N, N)

class TAGTModel(nn.Module):
    def __init__(self, n_genes, hidden_dim=128, n_heads=4, dropout=0.1):
        super(TAGTModel, self).__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # Gene expression encoder with graph-aware processing
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Graph convolution layer (simplified)
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Clinical data integration
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, expression, current_sledai, adjacency_matrix):
        batch_size = expression.size(0)

        # Encode gene expression
        gene_features = self.gene_encoder(expression)  # (batch_size, hidden_dim)

        # Apply simplified graph convolution
        # Since we have gene expression features (batch_size, hidden_dim) and adjacency (n_genes, n_genes)

        # For simplicity, let's use the adjacency matrix to create a graph-aware representation
        # by computing a weighted average of the gene features based on the graph structure

        # Normalize adjacency matrix for each row (make it a transition matrix)
        row_sums = adjacency_matrix.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_adj = adjacency_matrix / row_sums

        # Apply graph convolution: for each sample, use the normalized adjacency to aggregate features
        # Since gene_features is (batch_size, hidden_dim), we'll apply a simple transformation
        graph_features = self.graph_conv(gene_features)  # (batch_size, hidden_dim)

        # Add temporal dimension for attention
        graph_features = graph_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Apply temporal attention (self-attention in this case)
        attended_features, _ = self.temporal_attention(
            graph_features, graph_features, graph_features
        )
        attended_features = attended_features.squeeze(1)  # (batch_size, hidden_dim)

        # Encode clinical data
        clinical_features = self.clinical_encoder(current_sledai)

        # Combine features
        combined_features = torch.cat([attended_features, clinical_features], dim=1)

        # Classify
        output = self.classifier(combined_features)

        return output

def train_model():
    print("="*80)
    print("TRAINING TAGT MODEL FOR SLE FLARE PREDICTION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load integrated dataset
    print("\n1. Loading integrated dataset...")
    sequences_df = pd.read_pickle("data/integrated/sequences.pkl")
    labels = np.load("data/integrated/labels.npy")
    adjacency_matrix = np.load("data/integrated/adjacency_matrix.npy")
    
    sequences = sequences_df.to_dict('records')
    
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Positive samples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"   Gene features: {len(sequences[0]['expression'])}")
    print(f"   PPI network size: {adjacency_matrix.shape}")
    
    # Split data
    print("\n2. Splitting data...")
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=0.25, random_state=42, stratify=train_labels
    )
    
    print(f"   Train: {len(train_sequences)} samples")
    print(f"   Validation: {len(val_sequences)} samples") 
    print(f"   Test: {len(test_sequences)} samples")
    
    # Create datasets and dataloaders
    train_dataset = SLEDataset(train_sequences, train_labels, adjacency_matrix)
    val_dataset = SLEDataset(val_sequences, val_labels, adjacency_matrix)
    test_dataset = SLEDataset(test_sequences, test_labels, adjacency_matrix)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    print("\n3. Initializing TAGT model...")
    n_genes = len(sequences[0]['expression'])
    model = TAGTModel(n_genes=n_genes, hidden_dim=128, n_heads=4)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n4. Training model...")
    best_val_f1 = 0
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            expression = batch['expression'].to(device)
            current_sledai = batch['current_sledai'].to(device)
            adjacency_matrix = batch['adjacency_matrix'][0].to(device)  # Same for all samples
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(expression, current_sledai, adjacency_matrix)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                expression = batch['expression'].to(device)
                current_sledai = batch['current_sledai'].to(device)
                adjacency_matrix = batch['adjacency_matrix'][0].to(device)
                labels = batch['label'].squeeze().to(device)
                
                outputs = model(expression, current_sledai, adjacency_matrix)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_f1 = f1_score(train_targets, train_preds)
        val_f1 = f1_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch:2d}: Train F1={train_f1:.3f}, Val F1={val_f1:.3f}, Val Acc={val_acc:.3f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "models/best_tagt_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"   Early stopping at epoch {epoch}")
                break
    
    # Load best model and evaluate on test set
    print("\n5. Evaluating on test set...")
    model.load_state_dict(torch.load("models/best_tagt_model.pt"))
    model.eval()
    
    test_preds = []
    test_targets = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            expression = batch['expression'].to(device)
            current_sledai = batch['current_sledai'].to(device)
            adjacency_matrix = batch['adjacency_matrix'][0].to(device)
            labels = batch['label'].squeeze().to(device)
            
            outputs = model(expression, current_sledai, adjacency_matrix)
            probs = torch.softmax(outputs, dim=1)
            
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())  # Probability of flare
    
    # Calculate final metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds)
    test_recall = recall_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds)
    test_auc = roc_auc_score(test_targets, test_probs)
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS:")
    print("="*80)
    print(f"Accuracy:  {test_acc:.3f}")
    print(f"Precision: {test_precision:.3f}")
    print(f"Recall:    {test_recall:.3f}")
    print(f"F1-Score:  {test_f1:.3f}")
    print(f"AUC-ROC:   {test_auc:.3f}")
    print("="*80)
    
    # Save final model
    torch.save(model.state_dict(), "models/final_tagt_model.pt")
    
    print(f"\nModel saved to: models/final_tagt_model.pt")
    print("Training completed successfully!")

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    train_model()
