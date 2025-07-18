import torch
import torch.nn as nn
import scipy.sparse
import numpy as np

class GraphConv(nn.Module):
    """Simple Graph Convolutional Layer."""
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # Ensure adjacency matrix is sparse and coalesced
        if not adj.is_sparse:
            adj = adj.to_sparse().coalesce()
        else:
            adj = adj.coalesce()
        
        # Apply linear transformation first
        transformed = self.linear(x)
        
        # Handle input dimensions for sparse matrix multiplication
        if x.dim() == 3:  # [batch_size, n_genes, features]
            batch_size, n_genes, n_features = x.shape
            
            # Reshape for batch matrix multiplication
            permuted = transformed.permute(0, 2, 1)  # [batch_size, out_features, n_genes]
            
            # Apply adjacency matrix to each sample in the batch
            conv_results = []
            for i in range(batch_size):
                sample_result = torch.sparse.mm(adj, permuted[i].T).T  # [out_features, n_genes]
                conv_results.append(sample_result)
            
            # Stack and reshape back to original format
            output = torch.stack(conv_results, dim=0)  # [batch_size, out_features, n_genes]
            output = output.permute(0, 2, 1)  # [batch_size, n_genes, out_features]
            
        else:
            # Direct sparse matrix multiplication for 2D input
            output = torch.sparse.mm(adj, transformed)
        
        return output

# Test the GraphConv layer
if __name__ == '__main__':
        adj_scipy = scipy.sparse.random(100, 100, density=0.1, format='csr')
    adj_coo = adj_scipy.tocoo()
    indices = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long)
    values = torch.tensor(adj_coo.data, dtype=torch.float32)
    adj_torch = torch.sparse_coo_tensor(indices, values, adj_coo.shape, dtype=torch.float32).coalesce()

        x = torch.randn(32, 100, 1)

        graph_conv = GraphConv(1, 128)

    # Pass the input through the layer
    output = graph_conv(x, adj_torch)

    print('Input shape:', x.shape)
    print('Output shape:', output.shape)
    print('Test passed!')