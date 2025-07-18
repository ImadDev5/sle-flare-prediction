"""
Breakthrough TAGT Model Implementation

State-of-the-art Temporal Attention Graph Transformer for SLE Flare Prediction
with advanced components for breakthrough performance:

1. Multi-scale Graph Attention with edge features
2. Hierarchical temporal modeling
3. Cross-modal attention fusion
4. Pathway-aware attention mechanisms
5. Advanced regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math

class MultiScaleGraphAttention(nn.Module):
    """Multi-scale graph attention with edge features and adaptive weights."""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, edge_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Multi-head projections
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        
        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Multi-scale convolutions
        self.conv1 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=1)
        self.conv3 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=7, padding=3)
        
        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [batch_size, num_nodes, in_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
            edge_attr: Edge attributes [batch_size, num_nodes, num_nodes, edge_dim]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Multi-scale feature extraction
        x_conv = x.transpose(1, 2)  # [batch, in_dim, num_nodes]
        conv1_out = self.conv1(x_conv)
        conv3_out = self.conv3(x_conv)
        conv5_out = self.conv5(x_conv)
        conv7_out = self.conv7(x_conv)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([conv1_out, conv3_out, conv5_out, conv7_out], dim=1)
        multi_scale = multi_scale.transpose(1, 2)  # [batch, num_nodes, out_dim]
        
        # Multi-head attention
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / self.temperature
        
        # Add edge features if available
        if edge_attr is not None:
            edge_scores = self.edge_proj(edge_attr)  # [batch, num_nodes, num_nodes, num_heads]
            edge_scores = edge_scores.permute(0, 3, 1, 2)  # [batch, num_heads, num_nodes, num_nodes]
            scores = scores + edge_scores
        
        # Apply adjacency mask
        adj_mask = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(adj_mask == 0, float('-inf'))
        
        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bmhd->bnhd', attn_weights, v)
        out = out.contiguous().view(batch_size, num_nodes, self.out_dim)
        
        # Combine with multi-scale features
        out = out + multi_scale
        
        # Output projection and residual connection
        out = self.out_proj(out)
        out = self.layer_norm(out + x if x.shape[-1] == self.out_dim else out)
        
        return out

class PathwayAttention(nn.Module):
    """Pathway-aware attention mechanism for biological interpretability."""
    
    def __init__(self, feature_dim: int, pathway_dim: int = 64, num_pathways: int = 50):
        super().__init__()
        self.feature_dim = feature_dim
        self.pathway_dim = pathway_dim
        self.num_pathways = num_pathways
        
        # Pathway embeddings (learnable)
        self.pathway_embeddings = nn.Parameter(torch.randn(num_pathways, pathway_dim))
        
        # Feature to pathway projection
        self.feature_to_pathway = nn.Linear(feature_dim, pathway_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(pathway_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(pathway_dim, feature_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Gene features [batch_size, num_genes, feature_dim]
        Returns:
            pathway_features: Pathway-level features [batch_size, num_pathways, feature_dim]
            attention_weights: Pathway attention weights [batch_size, num_genes, num_pathways]
        """
        batch_size, num_genes, _ = x.shape
        
        # Project features to pathway space
        gene_pathway = self.feature_to_pathway(x)  # [batch, num_genes, pathway_dim]
        
        # Expand pathway embeddings for batch
        pathway_emb = self.pathway_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: genes attend to pathways
        pathway_features, attn_weights = self.attention(
            query=pathway_emb,
            key=gene_pathway,
            value=gene_pathway
        )
        
        # Project back to feature space
        pathway_features = self.output_proj(pathway_features)
        
        return pathway_features, attn_weights.transpose(1, 2)

class HierarchicalTemporalEncoder(nn.Module):
    """Hierarchical temporal encoding with multiple time scales."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Multi-scale temporal encoders
        self.short_term_encoder = nn.LSTM(feature_dim, hidden_dim // 2, num_layers, 
                                        batch_first=True, bidirectional=True)
        self.medium_term_encoder = nn.LSTM(feature_dim, hidden_dim // 2, num_layers,
                                         batch_first=True, bidirectional=True)
        self.long_term_encoder = nn.LSTM(feature_dim, hidden_dim // 2, num_layers,
                                       batch_first=True, bidirectional=True)
        
        # Temporal fusion
        self.temporal_fusion = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Temporal features [batch_size, seq_len, feature_dim]
            seq_lengths: Actual sequence lengths [batch_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-scale temporal encoding
        short_out, _ = self.short_term_encoder(x)
        
        # Medium-term: subsample by 2
        medium_x = x[:, ::2, :] if seq_len > 1 else x
        medium_out, _ = self.medium_term_encoder(medium_x)
        medium_out = F.interpolate(medium_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        
        # Long-term: subsample by 4
        long_x = x[:, ::4, :] if seq_len > 3 else x
        long_out, _ = self.long_term_encoder(long_x)
        long_out = F.interpolate(long_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        
        # Combine multi-scale features
        multi_scale = torch.stack([short_out, medium_out, long_out], dim=2)  # [batch, seq_len, 3, hidden_dim]
        multi_scale = multi_scale.view(batch_size, seq_len * 3, self.hidden_dim)
        
        # Temporal fusion with attention
        fused, _ = self.temporal_fusion(multi_scale, multi_scale, multi_scale)
        
        # Aggregate temporal information
        temporal_features = fused.view(batch_size, seq_len, 3, self.hidden_dim).mean(dim=2)
        
        # Output projection
        output = self.output_proj(temporal_features)
        
        return output

class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion for genomic and clinical data."""
    
    def __init__(self, genomic_dim: int, clinical_dim: int, fusion_dim: int = 256):
        super().__init__()
        self.genomic_dim = genomic_dim
        self.clinical_dim = clinical_dim
        self.fusion_dim = fusion_dim
        
        # Modality projections
        self.genomic_proj = nn.Linear(genomic_dim, fusion_dim)
        self.clinical_proj = nn.Linear(clinical_dim, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, genomic_features: torch.Tensor, 
                clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            genomic_features: [batch_size, seq_len, genomic_dim]
            clinical_features: [batch_size, clinical_dim]
        """
        batch_size, seq_len, _ = genomic_features.shape
        
        # Project to fusion space
        genomic_proj = self.genomic_proj(genomic_features)
        clinical_proj = self.clinical_proj(clinical_features).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Cross-modal attention
        attended_genomic, _ = self.cross_attention(
            query=genomic_proj,
            key=clinical_proj,
            value=clinical_proj
        )
        
        attended_clinical, _ = self.cross_attention(
            query=clinical_proj,
            key=genomic_proj,
            value=genomic_proj
        )
        
        # Concatenate and fuse
        fused = torch.cat([attended_genomic, attended_clinical], dim=-1)
        output = self.fusion_layers(fused)
        
        return output

class BreakthroughTAGT(nn.Module):
    """Breakthrough TAGT model with state-of-the-art components."""
    
    def __init__(self, 
                 n_genes: int = 1000,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 clinical_dim: int = 10,
                 dropout: float = 0.1,
                 num_pathways: int = 50):
        super().__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projections
        self.gene_embedding = nn.Linear(1, hidden_dim)  # Single gene expression to hidden_dim
        self.positional_encoding = nn.Parameter(torch.randn(n_genes, hidden_dim))
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            MultiScaleGraphAttention(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Pathway attention
        self.pathway_attention = PathwayAttention(
            feature_dim=hidden_dim,
            num_pathways=num_pathways
        )
        
        # Temporal encoder
        self.temporal_encoder = HierarchicalTemporalEncoder(
            feature_dim=hidden_dim + num_pathways,  # Gene + pathway features
            hidden_dim=hidden_dim
        )
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(
            genomic_dim=hidden_dim,
            clinical_dim=clinical_dim,
            fusion_dim=hidden_dim
        )
        
        # Output layers
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
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, std=0.02)
    
    def forward(self, 
                gene_expression: torch.Tensor,
                adjacency: torch.Tensor,
                clinical_features: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            gene_expression: [batch_size, seq_len, n_genes]
            adjacency: [n_genes, n_genes] or [batch_size, n_genes, n_genes]
            clinical_features: [batch_size, clinical_dim]
            edge_attr: [n_genes, n_genes, edge_dim] (optional)
        """
        batch_size, seq_len, n_genes = gene_expression.shape
        
        # Ensure adjacency has batch dimension
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process each time step
        temporal_features = []
        pathway_attentions = []
        
        for t in range(seq_len):
            # Get gene expression at time t
            x_t = gene_expression[:, t, :]  # [batch_size, n_genes]
            
            # Embed gene expressions
            x_embedded = self.gene_embedding(x_t.unsqueeze(-1))  # [batch_size, n_genes, hidden_dim]
            
            # Add positional encoding
            x_embedded = x_embedded + self.positional_encoding.unsqueeze(0)
            
            # Apply graph attention layers
            for layer in self.graph_layers:
                x_embedded = layer(x_embedded, adjacency, edge_attr)
            
            # Pathway attention
            pathway_features, pathway_attn = self.pathway_attention(x_embedded)
            pathway_attentions.append(pathway_attn)
            
            # Combine gene and pathway features
            gene_pooled = x_embedded.mean(dim=1)  # [batch_size, hidden_dim]
            pathway_pooled = pathway_features.mean(dim=1)  # [batch_size, hidden_dim]
            
            combined_features = torch.cat([gene_pooled, pathway_pooled], dim=-1)
            temporal_features.append(combined_features)
        
        # Stack temporal features
        temporal_sequence = torch.stack(temporal_features, dim=1)  # [batch_size, seq_len, hidden_dim*2]
        
        # Temporal encoding
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
            'pathway_attention': torch.stack(pathway_attentions, dim=1),
            'final_features': final_features
        }
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for interpretability."""
        attention_weights = {}
        
        for i, layer in enumerate(self.graph_layers):
            if hasattr(layer, 'last_attention_weights'):
                attention_weights[f'graph_layer_{i}'] = layer.last_attention_weights
        
        return attention_weights

def create_breakthrough_model(config: Dict) -> BreakthroughTAGT:
    """Create breakthrough TAGT model from configuration."""
    return BreakthroughTAGT(
        n_genes=config.get('n_genes', 1000),
        hidden_dim=config.get('hidden_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        clinical_dim=config.get('clinical_dim', 10),
        dropout=config.get('dropout', 0.1),
        num_pathways=config.get('num_pathways', 50)
    )

if __name__ == "__main__":
    # Test the model
    config = {
        'n_genes': 1000,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'clinical_dim': 10,
        'dropout': 0.1,
        'num_pathways': 50
    }
    
    model = create_breakthrough_model(config)
    
    # Test forward pass
    batch_size, seq_len, n_genes = 4, 3, 1000
    
    gene_expression = torch.randn(batch_size, seq_len, n_genes)
    adjacency = torch.rand(n_genes, n_genes)
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    clinical_features = torch.randn(batch_size, 10)
    
    output = model(gene_expression, adjacency, clinical_features)
    
    print(f"Model output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")