"""
Debug what the model actually outputs
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.models.optimized_tagt import create_optimized_model

def debug_model():
    print("=== DEBUGGING MODEL OUTPUT ===")
    
    # Load config
    if os.path.exists("configs/optimized_tagt_config.json"):
        with open("configs/optimized_tagt_config.json", 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model_architecture': {
                'n_genes': 1000,
                'hidden_dim': 256,
                'num_graph_layers': 3,
                'num_heads': 8,
                'temporal_hidden_dim': 128,
                'clinical_dim': 15,
                'dropout': 0.1
            }
        }
    
        model = create_optimized_model(config)
    
    # Load weights
    model_path = "results/best_optimized_model.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    # Load one sample of data
    with open('data/integrated/sequences_real.pkl', 'rb') as f:
        sequences = pickle.load(f)
    
    labels = np.load('data/integrated/labels_real.npy')
    
    adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
    adjacency = adjacency_sparse.toarray().astype(np.float32)
    
    # Test with first sample
    seq = sequences[0]
    print(f"\nFirst sequence keys: {seq.keys()}")
    print(f"Expression shape: {seq['expression'].shape}")
    print(f"Label: {labels[0]}")
    
    # Prepare input
    gene_expr = torch.FloatTensor(seq['expression']).unsqueeze(0).unsqueeze(1)  # [1, 1, 1000]
    clinical = torch.zeros(1, 15)  # [1, 15]
    adj = torch.FloatTensor(adjacency).unsqueeze(0)  # [1, 1000, 1000]
    
    print(f"\nInput shapes:")
    print(f"  Gene expression: {gene_expr.shape}")
    print(f"  Clinical: {clinical.shape}")
    print(f"  Adjacency: {adj.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(gene_expr, adj, clinical)
    
    print(f"\nModel outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape} | {value}")
    
    # Check what probabilities look like
    if 'probabilities' in outputs:
        prob = outputs['probabilities'].item()
        print(f"\nProbability: {prob:.6f}")
        print(f"Prediction (>0.5): {prob > 0.5}")
    
    if 'logits' in outputs:
        logit = outputs['logits'].item()
        manual_prob = torch.sigmoid(torch.tensor(logit)).item()
        print(f"Logit: {logit:.6f}")
        print(f"Manual sigmoid: {manual_prob:.6f}")

if __name__ == "__main__":
    debug_model()