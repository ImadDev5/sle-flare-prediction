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
import warnings
warnings.filterwarnings('ignore')

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

# Original TAGT Model (for reference)
class TAGTModel(nn.Module):
    def __init__(self, n_genes, hidden_dim=128, n_heads=4, dropout=0.1):
        super(TAGTModel, self).__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, expression, current_sledai, adjacency_matrix):
        batch_size = expression.size(0)
        
        # Encode gene expression
        gene_features = self.gene_encoder(expression)
        
        # Apply graph convolution
        row_sums = adjacency_matrix.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        normalized_adj = adjacency_matrix / row_sums
        graph_features = self.graph_conv(gene_features)
        
        # Temporal attention
        graph_features = graph_features.unsqueeze(1)
        attended_features, _ = self.temporal_attention(
            graph_features, graph_features, graph_features
        )
        attended_features = attended_features.squeeze(1)
        
        # Clinical features
        clinical_features = self.clinical_encoder(current_sledai)
        
        # Combine and classify
        combined_features = torch.cat([attended_features, clinical_features], dim=1)
        output = self.classifier(combined_features)
        
        return output

# Ablation 1: No Graph Component
class TAGTNoGraph(nn.Module):
    def __init__(self, n_genes, hidden_dim=128, n_heads=4, dropout=0.1):
        super(TAGTNoGraph, self).__init__()
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, expression, current_sledai, adjacency_matrix):
        # Encode gene expression (no graph processing)
        gene_features = self.gene_encoder(expression)
        
        # Temporal attention
        gene_features = gene_features.unsqueeze(1)
        attended_features, _ = self.temporal_attention(
            gene_features, gene_features, gene_features
        )
        attended_features = attended_features.squeeze(1)
        
        # Clinical features
        clinical_features = self.clinical_encoder(current_sledai)
        
        # Combine and classify
        combined_features = torch.cat([attended_features, clinical_features], dim=1)
        output = self.classifier(combined_features)
        
        return output

# Ablation 2: No Attention Component
class TAGTNoAttention(nn.Module):
    def __init__(self, n_genes, hidden_dim=128, dropout=0.1):
        super(TAGTNoAttention, self).__init__()
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, expression, current_sledai, adjacency_matrix):
        # Encode gene expression
        gene_features = self.gene_encoder(expression)
        
        # Apply graph convolution (no attention)
        graph_features = self.graph_conv(gene_features)
        
        # Clinical features
        clinical_features = self.clinical_encoder(current_sledai)
        
        # Combine and classify
        combined_features = torch.cat([graph_features, clinical_features], dim=1)
        output = self.classifier(combined_features)
        
        return output

# Ablation 3: No Temporal Component (same as No Attention in this case)
class TAGTNoTemporal(nn.Module):
    def __init__(self, n_genes, hidden_dim=128, dropout=0.1):
        super(TAGTNoTemporal, self).__init__()
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, expression, current_sledai, adjacency_matrix):
        # Encode gene expression
        gene_features = self.gene_encoder(expression)
        
        # Apply graph convolution
        graph_features = self.graph_conv(gene_features)
        
        # Clinical features
        clinical_features = self.clinical_encoder(current_sledai)
        
        # Combine and classify
        combined_features = torch.cat([graph_features, clinical_features], dim=1)
        output = self.classifier(combined_features)
        
        return output

# Ablation 4: Clinical Only
class ClinicalOnly(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
        super(ClinicalOnly, self).__init__()
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
    def forward(self, expression, current_sledai, adjacency_matrix):
        # Only use clinical features
        clinical_features = self.clinical_encoder(current_sledai)
        output = self.classifier(clinical_features)
        return output

def train_and_evaluate_model(model, train_loader, test_loader, model_name, device, epochs=30):
    """Train and evaluate a model variant"""
    print(f"\nTraining {model_name}...")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            expression = batch['expression'].to(device)
            current_sledai = batch['current_sledai'].to(device)
            adjacency_matrix = batch['adjacency_matrix'][0].to(device)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(expression, current_sledai, adjacency_matrix)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}")
    
    # Evaluation
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
            test_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(test_targets, test_probs)
    except:
        auc = 0.5
    
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main():
    print("="*80)
    print("ABLATION STUDIES FOR TAGT MODEL")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading integrated dataset...")
    sequences_df = pd.read_pickle("data/integrated/sequences.pkl")
    labels = np.load("data/integrated/labels.npy")
    adjacency_matrix = np.load("data/integrated/adjacency_matrix.npy")
    
    sequences = sequences_df.to_dict('records')
    n_genes = len(sequences[0]['expression'])
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Gene features: {n_genes}")
    
    # Split data
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = SLEDataset(train_sequences, train_labels, adjacency_matrix)
    test_dataset = SLEDataset(test_sequences, test_labels, adjacency_matrix)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Define model variants
    models = {
        'TAGT (Full)': TAGTModel(n_genes=n_genes),
        'TAGT (No Graph)': TAGTNoGraph(n_genes=n_genes),
        'TAGT (No Attention)': TAGTNoAttention(n_genes=n_genes),
        'TAGT (No Temporal)': TAGTNoTemporal(n_genes=n_genes),
        'Clinical Only': ClinicalOnly()
    }
    
    # Train and evaluate each variant
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = train_and_evaluate_model(
            model, train_loader, test_loader, model_name, device
        )
    
    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Model Variant':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['auc']:<10.3f}")
    
    # Save results
    with open('ablation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: ablation_results.pkl")
    print("\n" + "="*80)
    print("ABLATION STUDIES COMPLETE!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = main()
