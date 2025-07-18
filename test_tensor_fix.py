"""
Test script to validate tensor operations and sparse matrix handling.
"""

import torch
import numpy as np
from scipy import sparse
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sparse_tensor_conversion():
    """Test conversion from scipy sparse to PyTorch sparse tensor."""
    print("Testing sparse tensor conversion...")
    
        n_genes = 100
    density = 0.1
    np.random.seed(42)
    
        data = np.random.random(int(n_genes * n_genes * density))
    row = np.random.randint(0, n_genes, size=len(data))
    col = np.random.randint(0, n_genes, size=len(data))
    
    scipy_sparse = sparse.coo_matrix((data, (row, col)), shape=(n_genes, n_genes))
    print(f"Created scipy sparse matrix: {scipy_sparse.shape}, nnz: {scipy_sparse.nnz}")
    
    # Convert to PyTorch sparse tensor (our new method)
    adj_coo = scipy_sparse.tocoo()
    indices = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long)
    values = torch.tensor(adj_coo.data, dtype=torch.float32)
    torch_sparse = torch.sparse_coo_tensor(indices, values, adj_coo.shape, dtype=torch.float32).coalesce()
    
    print(f"Converted to PyTorch sparse tensor: {torch_sparse.shape}, nnz: {torch_sparse._nnz()}")
    print(f"Is sparse: {torch_sparse.is_sparse}")
    
    return torch_sparse

def test_graph_conv_operations():
    """Test graph convolution operations with batched input."""
    print("\nTesting graph convolution operations...")
    
        batch_size = 4
    n_genes = 100
    in_features = 1
    out_features = 64
    
        torch_sparse = test_sparse_tensor_conversion()
    
        x = torch.randn(batch_size, n_genes, in_features)
    print(f"Input tensor shape: {x.shape}")
    
        linear = torch.nn.Linear(in_features, out_features)
    
    try:
        # Apply linear transformation first
        x_transformed = linear(x)  # [batch_size, n_genes, out_features]
        print(f"After linear transformation: {x_transformed.shape}")
        
        # Reshape for batch matrix multiplication
        x_reshaped = x_transformed.permute(0, 2, 1)  # [batch_size, out_features, n_genes]
        print(f"After permute: {x_reshaped.shape}")
        
        # Apply adjacency matrix to each sample in the batch
        x_out = []
        for i in range(batch_size):
            x_sample = torch.sparse.mm(torch_sparse, x_reshaped[i].T).T  # [out_features, n_genes]
            x_out.append(x_sample)
            if i == 0:
                print(f"Sample {i} after sparse mm: {x_sample.shape}")
        
        x_final = torch.stack(x_out, dim=0)  # [batch_size, out_features, n_genes]
        x_final = x_final.permute(0, 2, 1)  # [batch_size, n_genes, out_features]
        
        print(f"Final output shape: {x_final.shape}")
        print("✅ Graph convolution operations successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in graph convolution: {e}")
        return False

def test_model_forward():
    """Test the complete model forward pass."""
    print("\nTesting complete model forward pass...")
    
    try:
        # Import our classes
        from train_real_data_model import GraphConv, ProductionTAGTModel
        
                batch_size = 4
        n_genes = 100
        n_clinical = 5
        hidden_dim = 64
        
                model = ProductionTAGTModel(n_genes, n_clinical, hidden_dim)
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
                gene_expr = torch.randn(batch_size, n_genes)
        clinical = torch.randn(batch_size, n_clinical)
        adj = test_sparse_tensor_conversion()
        
        print(f"Gene expression shape: {gene_expr.shape}")
        print(f"Clinical data shape: {clinical.shape}")
        print(f"Adjacency matrix shape: {adj.shape}")
        
        # Forward pass
        output = model(gene_expr, adj, clinical)
        print(f"Model output shape: {output.shape}")
        print("✅ Complete model forward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing tensor operations and model fixes...")
    print("=" * 60)
    
    success = True
    
    # Test sparse tensor conversion
    try:
        test_sparse_tensor_conversion()
    except Exception as e:
        print(f"❌ Sparse tensor conversion failed: {e}")
        success = False
    
    # Test graph convolution operations
    if not test_graph_conv_operations():
        success = False
    
    # Test complete model
    if not test_model_forward():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The tensor fixes should work.")
    else:
        print("❌ Some tests failed. Please check the errors above.")