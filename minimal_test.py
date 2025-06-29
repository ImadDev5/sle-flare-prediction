import torch
import numpy as np
from scipy import sparse

print("Testing basic tensor operations...")

# Test 1: Basic tensor creation
try:
    x = torch.randn(2, 10, 1)
    print(f"✅ Basic tensor creation: {x.shape}")
except Exception as e:
    print(f"❌ Basic tensor creation failed: {e}")

# Test 2: Sparse matrix conversion
try:
    # Create scipy sparse matrix
    data = np.array([1, 2, 3])
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    scipy_sparse = sparse.coo_matrix((data, (row, col)), shape=(10, 10))
    
    # Convert to PyTorch sparse
    adj_coo = scipy_sparse.tocoo()
    indices = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long)
    values = torch.tensor(adj_coo.data, dtype=torch.float32)
    torch_sparse = torch.sparse_coo_tensor(indices, values, adj_coo.shape, dtype=torch.float32).coalesce()
    
    print(f"✅ Sparse conversion: {torch_sparse.shape}, sparse: {torch_sparse.is_sparse}")
except Exception as e:
    print(f"❌ Sparse conversion failed: {e}")

# Test 3: Sparse matrix multiplication
try:
    x = torch.randn(10, 5)
    result = torch.sparse.mm(torch_sparse, x)
    print(f"✅ Sparse matrix multiplication: {result.shape}")
except Exception as e:
    print(f"❌ Sparse matrix multiplication failed: {e}")

print("Basic tests completed.")