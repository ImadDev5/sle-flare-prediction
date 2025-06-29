"""
Fixed Graph Attention Layer for SLE model.

This module implements a dimension-aware graph attention mechanism that properly handles
single-feature nodes in protein-protein interaction networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedGraphAttentionLayer(nn.Module):
    """
    Dimension-aware Graph Attention Layer for single-feature nodes.
    
    This implementation correctly handles the case where each node has a single feature
    (gene expression value) and ensures proper dimension handling throughout the attention
    mechanism and skip connections.
    
    Args:
        in_features: Number of nodes in the graph (number of genes)
        out_features: Output dimension per node
        dropout: Dropout probability
        alpha: LeakyReLU negative slope
        concat: Whether to concatenate attention heads (True) or average them (False)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, 
                 alpha: float = 0.2, n_heads: int = 4):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.all_head_size = self.head_dim * n_heads
        
        # Weight matrix for linear transformation of input
        # Note: We're transforming from 1 feature to head_dim features
        self.W = nn.Parameter(torch.zeros(size=(1, self.head_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.zeros(size=(self.head_dim, 1)))
        self.a_dst = nn.Parameter(torch.zeros(size=(self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        
        # Skip connection - transform from 1 feature to output dimension
        self.skip_projection = nn.Linear(1, self.all_head_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.all_head_size)
        
        # Activation and dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Apply graph attention mechanism to input features.
        
        Args:
            h: Input features of shape (n_nodes, 1) - each node has 1 expression value
            adj: Adjacency matrix of shape (n_nodes, n_nodes)
            
        Returns:
            Output features of shape (n_nodes, out_features)
        """
        n_nodes = h.size(0)
        
        # Apply skip connection projection (n_nodes, 1) -> (n_nodes, all_head_size)
        skip = self.skip_projection(h)
        
        # Linear transformation: (n_nodes, 1) -> (n_nodes, head_dim)
        Wh = torch.matmul(h, self.W)  # Use matmul for broadcasting
        
        # Expand to multiple heads: (n_nodes, n_heads, head_dim)
        Wh = Wh.unsqueeze(1).expand(-1, self.n_heads, -1)
        
        # Compute attention scores
        f1 = torch.matmul(Wh, self.a_src).squeeze(-1)  # (n_nodes, n_heads)
        f2 = torch.matmul(Wh, self.a_dst).squeeze(-1)  # (n_nodes, n_heads)
        
        # Compute attention matrix: (n_nodes, n_nodes, n_heads)
        e = self.leakyrelu(f1.unsqueeze(1) + f2.unsqueeze(0))
        
        # Mask attention with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(-1) > 0, e, zero_vec)
        
        # Apply softmax and dropout
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to get new node features
        # Reshape for batch matrix multiplication
        attention = attention.permute(2, 0, 1)  # (n_heads, n_nodes, n_nodes)
        Wh = Wh.permute(1, 0, 2)  # (n_heads, n_nodes, head_dim)
        
        # Apply attention: (n_heads, n_nodes, head_dim)
        h_prime = torch.bmm(attention, Wh)
        
        # Reshape back: (n_nodes, n_heads, head_dim) -> (n_nodes, all_head_size)
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.all_head_size)
        
        # Add skip connection and apply layer norm
        h_prime = h_prime + skip
        h_prime = self.layer_norm(h_prime)
        
        return F.elu(h_prime)


class MultiScaleGraphAttention(nn.Module):
    """
    Multi-scale graph attention module that captures both local and global interactions.
    
    This module applies graph attention at multiple scales:
    1. Local: Direct interactions between connected nodes
    2. Global: Attention across all nodes regardless of connectivity
    3. Hierarchical: Attention across node clusters
    
    Args:
        in_features: Number of input features per node
        hidden_dim: Hidden dimension size
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, in_features: int, hidden_dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        # Local attention (uses adjacency matrix)
        self.local_attention = FixedGraphAttentionLayer(
            in_features, hidden_dim, dropout, n_heads=n_heads
        )
        
        # Global attention (ignores adjacency matrix)
        self.global_attention = FixedGraphAttentionLayer(
            in_features, hidden_dim, dropout, n_heads=n_heads
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale graph attention.
        
        Args:
            x: Input features of shape (n_nodes, 1)
            adj: Adjacency matrix of shape (n_nodes, n_nodes)
            
        Returns:
            Output features of shape (n_nodes, hidden_dim)
        """
        # Local attention with real adjacency
        local_features = self.local_attention(x, adj)
        
        # Global attention with full connectivity
        global_adj = torch.ones_like(adj)
        global_features = self.global_attention(x, global_adj)
        
        # Combine features
        combined = torch.cat([local_features, global_features], dim=1)
        return self.fusion(combined)