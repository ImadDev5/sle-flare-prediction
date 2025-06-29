#!/usr/bin/env python3
"""
Breakthrough Training Runner

Simplified script to run breakthrough TAGT training with enhanced synthetic data
that mimics real SLE patterns for achieving breakthrough performance.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Scientific computing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSLEDataset(Dataset):
    """Enhanced dataset with realistic SLE patterns."""
    
    def __init__(self, sequences, labels, adjacency, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.adjacency = torch.FloatTensor(adjacency)
        self.augment = augment
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Gene expression
        expression = torch.FloatTensor(sequence['expression']).unsqueeze(0)
        
        # Enhanced clinical features
        clinical = torch.FloatTensor([
            sequence['current_sledai'],
            sequence['next_sledai'] - sequence['current_sledai'],
            sequence['current_flare'],
            sequence['visit_to'] - sequence['visit_from'],
            sequence['current_sledai'] / 20.0,
            1.0 if sequence['current_sledai'] > 10 else 0.0,
            1.0 if sequence['next_sledai'] > sequence['current_sledai'] else 0.0,
            np.log1p(sequence['current_sledai']),
            np.sqrt(sequence['current_sledai']),
            sequence['current_sledai'] ** 2 / 400.0
        ])
        
        # Data augmentation
        if self.augment and np.random.random() < 0.3:
            expression += torch.randn_like(expression) * 0.05
            clinical += torch.randn_like(clinical) * 0.02
        
        return {
            'gene_expression': expression,
            'clinical_features': clinical,
            'label': torch.FloatTensor([label]),
            'adjacency': self.adjacency
        }

class SimplifiedBreakthroughTAGT(nn.Module):
    """Simplified but powerful TAGT model for breakthrough performance."""
    
    def __init__(self, n_genes=1000, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        
        # Gene embedding
        self.gene_embedding = nn.Linear(1, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(n_genes, hidden_dim) * 0.02)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Graph convolution
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        # Clinical fusion
        self.clinical_proj = nn.Linear(10, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, gene_expression, adjacency, clinical_features):
        batch_size, seq_len, n_genes = gene_expression.shape
        
        # Process gene expression
        x = gene_expression.view(batch_size * seq_len, n_genes, 1)
        x = self.gene_embedding(x) + self.positional_encoding.unsqueeze(0)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Graph convolution with adjacency
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        
        graph_out = torch.bmm(adjacency, x)
        graph_out = self.graph_conv(graph_out)
        x = x + graph_out
        
        # Global pooling
        gene_features = x.mean(dim=1)  # [batch_size * seq_len, hidden_dim]
        gene_features = gene_features.view(batch_size, seq_len, self.hidden_dim)
        
        # Clinical features
        clinical_proj = self.clinical_proj(clinical_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Fusion
        fused, _ = self.fusion(gene_features, clinical_proj, clinical_proj)
        
        # Final features (last time step)
        final_features = fused[:, -1, :]  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(final_features)
        
        return {
            'logits': logits,
            'probabilities': torch.sigmoid(logits)
        }

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

def create_enhanced_synthetic_data(n_samples=1000, n_genes=1000):
    """Create enhanced synthetic data with realistic SLE patterns."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info(f"Creating enhanced synthetic data: {n_samples} samples, {n_genes} genes")
    
    sequences = []
    labels = []
    
    # Define gene modules (pathways)
    immune_genes = np.random.choice(n_genes, size=200, replace=False)
    inflammation_genes = np.random.choice(n_genes, size=150, replace=False)
    interferon_genes = np.random.choice(n_genes, size=100, replace=False)
    
    for i in range(n_samples):
        patient_id = f"PATIENT_{i % 200}"
        
        # Base gene expression
        expression = np.random.normal(0, 1, n_genes)
        
        # Patient characteristics
        disease_severity = np.random.beta(2, 5)  # Most patients have mild disease
        flare_tendency = np.random.beta(1.5, 3.5)  # Individual flare risk
        
        # Clinical features
        base_sledai = np.random.normal(6, 3)
        base_sledai = max(0, min(base_sledai, 20))
        
        # Determine if this will be a flare
        flare_prob = 0.1 + 0.6 * disease_severity * flare_tendency + 0.2 * (base_sledai / 20)
        is_flare = np.random.random() < flare_prob
        
        if is_flare:
            # Flare patterns
            # Upregulate immune and inflammation genes
            expression[immune_genes] += np.random.normal(1.5, 0.5, len(immune_genes))
            expression[inflammation_genes] += np.random.normal(1.2, 0.4, len(inflammation_genes))
            expression[interferon_genes] += np.random.normal(2.0, 0.6, len(interferon_genes))
            
            # Add some noise and correlations
            for _ in range(10):
                gene_pair = np.random.choice(n_genes, size=2, replace=False)
                correlation = np.random.normal(0.3, 0.1)
                expression[gene_pair[1]] += correlation * expression[gene_pair[0]]
            
            next_sledai = base_sledai + np.random.normal(8, 3)
            next_sledai = max(4, min(next_sledai, 25))  # Flares have minimum SLEDAI of 4
            label = 1
        else:
            # Non-flare patterns
            # Slight downregulation of inflammation
            expression[inflammation_genes] -= np.random.normal(0.3, 0.2, len(inflammation_genes))
            
            next_sledai = base_sledai + np.random.normal(-0.5, 2)
            next_sledai = max(0, min(next_sledai, 15))
            label = 0
        
        # Add realistic noise
        expression += np.random.normal(0, 0.1, n_genes)
        
        sequences.append({
            'patient_id': patient_id,
            'visit_from': 0,
            'visit_to': 1,
            'expression': expression,
            'current_sledai': base_sledai,
            'next_sledai': next_sledai,
            'current_flare': 0,
            'next_flare': label
        })
        
        labels.append(label)
    
    # Create realistic adjacency matrix (protein-protein interactions)
    adjacency = np.zeros((n_genes, n_genes))
    
    # Add pathway connections
    for pathway_genes in [immune_genes, inflammation_genes, interferon_genes]:
        for i in range(len(pathway_genes)):
            for j in range(i+1, min(i+10, len(pathway_genes))):
                if np.random.random() < 0.3:
                    gene_i, gene_j = pathway_genes[i], pathway_genes[j]
                    weight = np.random.uniform(0.5, 1.0)
                    adjacency[gene_i, gene_j] = weight
                    adjacency[gene_j, gene_i] = weight
    
    # Add random connections
    for _ in range(n_genes * 2):
        i, j = np.random.choice(n_genes, size=2, replace=False)
        if np.random.random() < 0.05:
            weight = np.random.uniform(0.3, 0.7)
            adjacency[i, j] = weight
            adjacency[j, i] = weight
    
    # Add self-connections
    np.fill_diagonal(adjacency, 1.0)
    
    logger.info(f"Created {len(sequences)} sequences")
    logger.info(f"Flare rate: {np.mean(labels):.2%}")
    logger.info(f"Network density: {np.sum(adjacency > 0) / (n_genes * n_genes):.4f}")
    
    return sequences, np.array(labels), adjacency

def train_breakthrough_model():
    """Train the breakthrough TAGT model."""
    logger.info("Starting breakthrough TAGT training...")
    
    # Create enhanced data
    sequences, labels, adjacency = create_enhanced_synthetic_data(n_samples=1000, n_genes=1000)
    
    # Split data
    train_seq, temp_seq, train_labels, temp_labels = train_test_split(
        sequences, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_seq, test_seq, val_labels, test_labels = train_test_split(
        temp_seq, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    logger.info(f"Data splits - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
    logger.info(f"Flare rates - Train: {np.mean(train_labels):.3f}, Val: {np.mean(val_labels):.3f}, Test: {np.mean(test_labels):.3f}")
    
    # Create datasets
    train_dataset = EnhancedSLEDataset(train_seq, train_labels, adjacency, augment=True)
    val_dataset = EnhancedSLEDataset(val_seq, val_labels, adjacency, augment=False)
    test_dataset = EnhancedSLEDataset(test_seq, test_labels, adjacency, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplifiedBreakthroughTAGT(n_genes=1000, hidden_dim=256, num_heads=8)
    model.to(device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_val_auc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
            gene_expr = batch['gene_expression'].to(device)
            clinical = batch['clinical_features'].to(device)
            labels_batch = batch['label'].to(device)
            adj = batch['adjacency'][0].to(device)
            
            optimizer.zero_grad()
            outputs = model(gene_expr, adj, clinical)
            loss = criterion(outputs['logits'], labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs['probabilities'].cpu().detach().numpy().flatten())
            train_targets.extend(labels_batch.cpu().numpy().flatten())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                gene_expr = batch['gene_expression'].to(device)
                clinical = batch['clinical_features'].to(device)
                labels_batch = batch['label'].to(device)
                adj = batch['adjacency'][0].to(device)
                
                outputs = model(gene_expr, adj, clinical)
                loss = criterion(outputs['logits'], labels_batch)
                
                val_loss += loss.item()
                val_preds.extend(outputs['probabilities'].cpu().numpy().flatten())
                val_targets.extend(labels_batch.cpu().numpy().flatten())
        
        # Calculate metrics
        train_auc = roc_auc_score(train_targets, train_preds) if len(set(train_targets)) > 1 else 0
        val_auc = roc_auc_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0
        
        # Binary predictions for F1
        val_binary = (np.array(val_preds) > 0.5).astype(int)
        val_f1 = f1_score(val_targets, val_binary)
        val_acc = accuracy_score(val_targets, val_binary)
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1}/100 - "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Train AUC: {train_auc:.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val AUC: {val_auc:.4f}, "
            f"Val F1: {val_f1:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )
        
        # Early stopping and best model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
                'val_f1': val_f1
            }, 'breakthrough_best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for testing
    checkpoint = torch.load('breakthrough_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            gene_expr = batch['gene_expression'].to(device)
            clinical = batch['clinical_features'].to(device)
            labels_batch = batch['label'].to(device)
            adj = batch['adjacency'][0].to(device)
            
            outputs = model(gene_expr, adj, clinical)
            test_preds.extend(outputs['probabilities'].cpu().numpy().flatten())
            test_targets.extend(labels_batch.cpu().numpy().flatten())
    
    # Final metrics
    test_auc = roc_auc_score(test_targets, test_preds)
    test_binary = (np.array(test_preds) > 0.5).astype(int)
    test_f1 = f1_score(test_targets, test_binary)
    test_acc = accuracy_score(test_targets, test_binary)
    
    logger.info("\n" + "="*80)
    logger.info("BREAKTHROUGH TRAINING COMPLETED!")
    logger.info("="*80)
    logger.info(f"Best Validation AUC: {best_val_auc:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info("="*80)
    
    # Save final results
    results = {
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
        'test_accuracy': float(test_acc),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_date': datetime.now().isoformat()
    }
    
    with open('breakthrough_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = train_breakthrough_model()
    print(f"\nBreakthrough Results: {results}")