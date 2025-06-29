#!/usr/bin/env python3
"""
🚀 OPTIMIZED TAGT MODEL - RTX 3050 EDITION
==========================================

Memory-efficient version of the Ultimate TAGT model specifically optimized 
for RTX 3050 and real data. This version maintains breakthrough performance
while being much more memory efficient.

Key Optimizations:
- Efficient Graph Attention (no memory explosion)
- Gradient checkpointing
- Memory-efficient temporal processing
- Optimized for 4GB VRAM
- Real data only - No compromises on performance!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
from torch.nn import MultiheadAttention, LayerNorm, Dropout
from torch.utils.checkpoint import checkpoint

class EfficientGraphAttentionLayer(nn.Module):
    """Memory-efficient Graph Attention Layer optimized for RTX 3050."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 8, 
                 dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.alpha = alpha
        
        assert out_features % num_heads == 0
        
        # Efficient projections
        self.W_q = nn.Linear(in_features, out_features, bias=False)
        self.W_k = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        self.W_o = nn.Linear(out_features, out_features)
        
        # Attention parameters
        self.scale = math.sqrt(self.head_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(out_features)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, h, adj, edge_features=None):
        """
        Efficient forward pass with controlled memory usage.
        """
        batch_size, N, _ = h.shape
        
        # Multi-head projections
        q = self.W_q(h).view(batch_size, N, self.num_heads, self.head_dim)
        k = self.W_k(h).view(batch_size, N, self.num_heads, self.head_dim)
        v = self.W_v(h).view(batch_size, N, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply adjacency mask efficiently
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        elif adj.dim() == 3:
            adj = adj.unsqueeze(1)  # [batch, 1, N, N]
        
        # Mask and softmax
        scores = scores.masked_fill(adj == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [batch, heads, N, head_dim]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, N, self.out_features)
        
        # Output projection
        out = self.W_o(out)
        
        # Residual connection and layer norm
        if self.in_features == self.out_features:
            out = out + h
        
        return self.layer_norm(out)

class MemoryEfficientTemporalEncoder(nn.Module):
    """Memory-efficient temporal encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Single bidirectional LSTM for efficiency
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, lengths=None):
        """Forward pass with memory efficiency."""
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention for temporal modeling
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Project back to input dimension
        output = self.output_proj(attended_out)
        
        return output

class OptimizedCrossModalFusion(nn.Module):
    """Memory-efficient cross-modal fusion."""
    
    def __init__(self, genomic_dim: int, clinical_dim: int, fusion_dim: int = 256):
        super().__init__()
        self.genomic_dim = genomic_dim
        self.clinical_dim = clinical_dim
        self.fusion_dim = fusion_dim
        
        # Efficient encoders
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Simple fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, genomic_features, clinical_features):
        """Efficient cross-modal fusion."""
        batch_size, seq_len, _ = genomic_features.shape
        
        # Encode modalities
        genomic_encoded = self.genomic_encoder(genomic_features)
        clinical_encoded = self.clinical_encoder(clinical_features)
        
        # Expand clinical features
        clinical_expanded = clinical_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Simple concatenation and fusion
        fused = torch.cat([genomic_encoded, clinical_expanded], dim=-1)
        output = self.fusion_layer(fused)
        
        return output

class OptimizedTAGT(nn.Module):
    """🚀 Optimized TAGT Model - RTX 3050 Ready!"""
    
    def __init__(self, 
                 n_genes: int = 1000,
                 hidden_dim: int = 256,  # Reduced for memory efficiency
                 num_heads: int = 8,     # Reduced for memory efficiency
                 num_layers: int = 3,    # Reduced for memory efficiency
                 clinical_dim: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Gene expression embedding
        self.gene_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Learnable positional encoding (more memory efficient)
        self.positional_encoding = nn.Parameter(torch.randn(1, n_genes, hidden_dim) * 0.02)
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            EfficientGraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Temporal encoder
        self.temporal_encoder = MemoryEfficientTemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # Cross-modal fusion
        self.cross_modal_fusion = OptimizedCrossModalFusion(
            genomic_dim=hidden_dim,
            clinical_dim=clinical_dim,
            fusion_dim=hidden_dim
        )
        
        # Efficient classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, gene_expression, adjacency, clinical_features):
        """
        Memory-efficient forward pass.
        
        Args:
            gene_expression: [batch_size, seq_len, n_genes]
            adjacency: [n_genes, n_genes] or [batch_size, n_genes, n_genes]
            clinical_features: [batch_size, clinical_dim]
        """
        batch_size, seq_len, n_genes = gene_expression.shape
        
        # Process each time step separately to save memory
        temporal_features = []
        
        for t in range(seq_len):
            # Get gene expression at time t
            x_t = gene_expression[:, t, :]  # [batch_size, n_genes]
            
            # Embed gene expressions
            x_embedded = self.gene_embedding(x_t.unsqueeze(-1))  # [batch_size, n_genes, hidden_dim]
            
            # Add positional encoding
            x_embedded = x_embedded + self.positional_encoding
            
            # Apply graph attention layers with gradient checkpointing for memory efficiency
            for layer in self.graph_layers:
                if self.training:
                    x_embedded = checkpoint(layer, x_embedded, adjacency, use_reentrant=False)
                else:
                    x_embedded = layer(x_embedded, adjacency)
            
            # Pool gene features to get sequence representation
            sequence_repr = x_embedded.mean(dim=1)  # [batch_size, hidden_dim]
            temporal_features.append(sequence_repr)
        
        # Stack temporal features
        temporal_sequence = torch.stack(temporal_features, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Apply temporal encoding
        temporal_encoded = self.temporal_encoder(temporal_sequence)
        
        # Cross-modal fusion with clinical data
        fused_features = self.cross_modal_fusion(temporal_encoded, clinical_features)
        
        # Final prediction (use last time step)
        final_features = fused_features[:, -1, :]  # [batch_size, hidden_dim]
        logits = self.classifier(final_features)
        
        return {
            'logits': logits,
            'probabilities': torch.sigmoid(logits),
            'temporal_features': temporal_encoded,
            'final_features': final_features
        }

def create_optimized_model(config: Dict) -> OptimizedTAGT:
    """Create optimized TAGT model from configuration."""
    model_config = config.get('model_architecture', {})
    
    return OptimizedTAGT(
        n_genes=model_config.get('n_genes', 1000),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_heads=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 3),
        clinical_dim=model_config.get('clinical_dim', 15),
        dropout=model_config.get('dropout', 0.1)
    )

if __name__ == "__main__":
    # Test the optimized model
    config = {
        'model_architecture': {
            'n_genes': 1000,
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 3,
            'clinical_dim': 15,
            'dropout': 0.1
        }
    }
    
    model = create_optimized_model(config)
    
    # Test forward pass
    batch_size, seq_len, n_genes = 4, 3, 1000
    
    gene_expression = torch.randn(batch_size, seq_len, n_genes)
    adjacency = torch.rand(n_genes, n_genes)
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    clinical_features = torch.randn(batch_size, 15)
    
    output = model(gene_expression, adjacency, clinical_features)
    
    print("🚀 OPTIMIZED TAGT MODEL READY!")
    print(f"Model output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("RTX 3050 OPTIMIZED - REAL DATA ONLY! 💪")
