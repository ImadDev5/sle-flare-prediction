"""
TAGT Model Implementation
Temporal Attention Graph Transformer for SLE Flare Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """Graph convolution layer for protein interaction modeling"""
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return self.dropout(F.relu(output))

class TemporalAttention(nn.Module):
    """Multi-head attention for temporal sequence modeling"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return self.norm(attended + x)

class TAGTModel(nn.Module):
    """
    Temporal Attention Graph Transformer for SLE Flare Prediction
    
    Combines graph neural networks for protein interaction modeling
    with temporal attention mechanisms for disease progression analysis.
    """
    
    def __init__(self, n_genes, hidden_dim=128, n_heads=4, dropout=0.1):
        super(TAGTModel, self).__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        
        # Gene expression encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution for PPI modeling
        self.graph_conv = GraphConvolution(hidden_dim, hidden_dim, dropout)
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_dim, n_heads, dropout)
        
        # Clinical data encoder
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
            nn.Linear(64, 2)
        )
        
    def forward(self, expression, clinical, adjacency):
        batch_size = expression.size(0)
        
        # Encode gene expression
        gene_features = self.gene_encoder(expression)
        
        # Apply graph convolution
        if adjacency is not None:
            # Normalize adjacency matrix
            row_sums = adjacency.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1
            normalized_adj = adjacency / row_sums
            
            # Graph convolution
            graph_features = self.graph_conv(gene_features, normalized_adj)
        else:
            graph_features = gene_features
        
        # Temporal attention
        graph_features = graph_features.unsqueeze(1)
        attended_features = self.temporal_attention(graph_features)
        attended_features = attended_features.squeeze(1)
        
        # Clinical feature encoding
        clinical_features = self.clinical_encoder(clinical)
        
        # Feature fusion
        combined_features = torch.cat([attended_features, clinical_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output
    
    def predict_proba(self, expression, clinical, adjacency):
        """Return prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(expression, clinical, adjacency)
            probabilities = F.softmax(logits, dim=1)
            return probabilities[:, 1]  # Return flare probability

class BaselineModel(nn.Module):
    """Simple baseline model for comparison"""
    
    def __init__(self, n_features, hidden_dim=64):
        super(BaselineModel, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.classifier(x)

def create_model(config):
    """Factory function to create TAGT model"""
    return TAGTModel(
        n_genes=config['n_genes'],
        hidden_dim=config.get('hidden_dim', 128),
        n_heads=config.get('n_heads', 4),
        dropout=config.get('dropout', 0.1)
    )