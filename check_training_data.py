"""
Check what data your optimized TAGT model was actually trained on
"""

import pickle
import numpy as np
import scipy.sparse as sp

print("=== CHECKING YOUR OPTIMIZED TAGT TRAINING DATA ===")

# Check sequences
try:
    with open('data/integrated/sequences_real.pkl', 'rb') as f:
        sequences = pickle.load(f)
    
    print(f"Sequences: {len(sequences)} items")
    print(f"First sequence type: {type(sequences[0])}")
    print(f"First sequence shape/length: {sequences[0].shape if hasattr(sequences[0], 'shape') else len(sequences[0])}")
    
    if hasattr(sequences[0], 'shape'):
        print(f"Sequence data type: {sequences[0].dtype}")
        print(f"Sample values from first sequence: {sequences[0][:5]}")
    else:
        print(f"First sequence content: {sequences[0]}")
        
except Exception as e:
    print(f"Error loading sequences: {e}")

# Check labels
try:
    labels = np.load('data/integrated/labels_real.npy')
    print(f"\nLabels shape: {labels.shape}")
    print(f"Labels type: {labels.dtype}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Flare rate: {np.mean(labels):.2%}")
    print(f"Sample labels: {labels[:10]}")
except Exception as e:
    print(f"Error loading labels: {e}")

# Check adjacency matrix
try:
    adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
    print(f"\nAdjacency matrix shape: {adjacency_sparse.shape}")
    print(f"Adjacency matrix type: {type(adjacency_sparse)}")
    print(f"Adjacency matrix density: {adjacency_sparse.nnz / (adjacency_sparse.shape[0] * adjacency_sparse.shape[1]):.4f}")
except Exception as e:
    print(f"Error loading adjacency: {e}")

# Check what I used for validation
try:
    import pandas as pd
    expression = pd.read_csv('data/processed/expression_real.csv', index_col=0)
    print(f"\nExpression data I used: {expression.shape}")
    print(f"Expression columns (genes): {list(expression.columns[:5])}...")
    print(f"Expression index (samples): {list(expression.index[:5])}...")
except Exception as e:
    print(f"Error loading expression data: {e}")

print("\n=== SUMMARY ===")
print("Your TAGT was trained on:")
print("- sequences_real.pkl (temporal sequences)")
print("- labels_real.npy (378 labels)")  
print("- adjacency_real.npz (graph structure)")
print("\nI tested it on:")
print("- expression_real.csv (static gene expression)")
print("- Same labels_real.npy")
print("- No graph structure")
print("\nTHIS IS THE MISMATCH!")