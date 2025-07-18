"""Main training script for TAGT model"""
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from datetime import datetime

print("=" * 80)
print("CREATING PPI NETWORK FOR SLE MODEL")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load expression data to get gene list
print("\n1. Loading gene expression data...")
expression_df = pd.read_csv(r"D:\SLE_data\processed\expression_normalized.csv", index_col=0)
genes = expression_df.index.tolist()
print(f"   Found {len(genes)} genes")

print("\n2. Creating scale-free PPI network...")
n_genes = len(genes)
n_edges = int(n_genes * 3)  # Average degree ~6 (typical for PPI networks)

G = nx.barabasi_albert_graph(n_genes, 3)

# Add some additional edges to increase connectivity
print("   Adding additional edges for biological realism...")
for _ in range(n_edges // 2):
    # Add edges preferentially to high-degree nodes
    node1 = np.random.choice(n_genes)
    node2 = np.random.choice(n_genes)
    if node1 != node2 and not G.has_edge(node1, node2):
        # Add edge with probability proportional to degree
        if np.random.random() < (G.degree(node1) + G.degree(node2)) / (2 * n_genes):
            G.add_edge(node1, node2)

# Convert to adjacency matrix
print("\n3. Converting to adjacency matrix...")
adj_matrix = nx.adjacency_matrix(G).astype(np.float32)

# Add edge weights (confidence scores)
print("   Adding edge weights...")
edges = list(G.edges())
for i, j in edges:
    # Assign weights between 0.15 and 0.99 (STRING confidence range)
    weight = np.random.beta(2, 2) * 0.84 + 0.15
    adj_matrix[i, j] = weight
    adj_matrix[j, i] = weight

gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

# Save the network
print("\n4. Saving PPI network...")
output_dir = r"D:\SLE_data\processed\ppi"
import os
os.makedirs(output_dir, exist_ok=True)

# Save adjacency matrix
np.save(os.path.join(output_dir, "adjacency_matrix.npy"), adj_matrix.toarray())

# Save gene mapping
with open(os.path.join(output_dir, "gene_mapping.pkl"), 'wb') as f:
    pickle.dump(gene_to_idx, f)

# Save network statistics
print("\n5. Network statistics:")
print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")
print(f"   Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
print(f"   Density: {nx.density(G):.4f}")
print(f"   Clustering coefficient: {nx.average_clustering(G):.4f}")

# Check if network is connected
n_components = nx.number_connected_components(G)
print(f"   Connected components: {n_components}")
if n_components > 1:
    largest_cc = max(nx.connected_components(G), key=len)
    print(f"   Largest component size: {len(largest_cc)} ({len(largest_cc)/G.number_of_nodes()*100:.1f}%)")

print("\n" + "=" * 80)
print("PPI NETWORK CREATION COMPLETE!")
print("=" * 80)
print(f"Saved to: {output_dir}")
print("Files created:")
print("  - adjacency_matrix.npy")
print("  - gene_mapping.pkl")
print("\nNote: This is a synthetic scale-free network that mimics real PPI properties")
print("For production use, consider using actual human PPI data from STRING or BioGRID")
print("=" * 80)