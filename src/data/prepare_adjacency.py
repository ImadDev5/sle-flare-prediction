import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import gc
import gzip
import logging
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adjacency_preparation.log'),
        logging.StreamHandler()
    ]
)

def prepare_adjacency_matrix():
    """
    Prepares the adjacency matrix by mapping expression data probes to the STRING PPI network.
    """
    # --- 0. Setup ---
    ppi_dir = Path("D:/SLE_data/processed/ppi")
    ppi_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load Mappings and Expression Data ---
    logging.info("Loading ID mapping file...")
    mapping_file = Path("D:/SLE_data/processed/mapping/probe_to_string_id_mapping.csv")
    if not mapping_file.exists():
        logging.error(f"Mapping file not found at {mapping_file}. Please run `create_id_mapping.py` first.")
        return
    
    id_mapping_df = pd.read_csv(mapping_file)
    logging.info(f"Loaded {len(id_mapping_df)} probe-to-STRING ID mappings.")

    logging.info("Loading expression data to identify probes in the dataset...")
    expr_df = pd.read_csv("D:/SLE_data/processed/expression_normalized.csv")
    all_probes_in_expr = expr_df['Unnamed: 0'].tolist()
    
    # Filter mapping to only include probes present in the expression data
    id_mapping_df = id_mapping_df[id_mapping_df['ProbeID'].isin(all_probes_in_expr)]
    logging.info(f"Filtered to {len(id_mapping_df)} mappings for probes present in expression data.")

    # --- 2. Create Final Gene List and Mappings ---
    # HANDLE DUPLICATE STRING IDs: A single protein (StringID) can map to multiple probes.
    # This creates a non-unique index, causing the error. For now, we will simplify this
    # by keeping only the first probe associated with each unique protein.
    logging.info(f"Number of mappings before deduplication: {len(id_mapping_df)}")
    id_mapping_df.drop_duplicates(subset=['StringID'], keep='first', inplace=True)
    logging.info(f"Number of mappings after deduplicating by StringID: {len(id_mapping_df)}")

    # The final list of "genes" for our model will be the probes that have a valid STRING ID
    final_probe_list = sorted(id_mapping_df['ProbeID'].unique().tolist()) # Sort for reproducibility
    n_genes = len(final_probe_list)
    
        probe_to_idx = {probe: i for i, probe in enumerate(final_probe_list)}
    
        string_id_to_idx = id_mapping_df.set_index('StringID')['ProbeID'].map(probe_to_idx)
    
    logging.info(f"Final network will have {n_genes} nodes (genes/probes).")
    
    # Save the final probe list. This is CRITICAL for the training script.
    probe_list_path = ppi_dir / "probe_list.csv"
    pd.DataFrame(final_probe_list, columns=['ProbeID']).to_csv(probe_list_path, index=False)
    logging.info(f"Saved final probe list for the network to {probe_list_path}")

    # --- 3. Build Adjacency Matrix ---
    logging.info("Initializing adjacency matrix...")
    adj_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
    
    logging.info(f"Loading PPI network from D:/SLE_data/raw/STRING/protein.links.v12.0.txt.gz")
    chunk_size = 5_000_000 # Larger chunk size for efficiency
    
    valid_interactions = 0
    total_ppi_read = 0
    
    column_names = ['protein1', 'protein2', 'combined_score']
    
    # Use a set for faster lookups
    valid_string_ids = set(string_id_to_idx.index)

    logging.info("Processing PPI network line-by-line for robustness...")
    ppi_file_path = "D:/SLE_data/raw/STRING/protein.links.v12.0.txt.gz"
    
    with gzip.open(ppi_file_path, 'rt', encoding='utf-8') as f:
        header = f.readline() # Skip header
        total_ppi_read = 0
        
        for line in tqdm(f, desc="Processing PPI File"):
            total_ppi_read += 1
            parts = line.strip().split(' ')
            
            # We only need the first two proteins and the last score column
            protein1 = parts[0]
            protein2 = parts[1]
            score = int(parts[-1])
            
            if score < 700:
                continue
            
            # Check if both proteins are in our dataset
            if protein1 in valid_string_ids and protein2 in valid_string_ids:
                idx1 = string_id_to_idx[protein1]
                idx2 = string_id_to_idx[protein2]
                normalized_score = score / 1000.0
                
                # Update adjacency matrix, ensuring we don't overwrite a higher score
                adj_matrix[idx1, idx2] = max(adj_matrix[idx1, idx2], normalized_score)
                adj_matrix[idx2, idx1] = max(adj_matrix[idx2, idx1], normalized_score)
                valid_interactions += 1

    logging.info(f"Finished processing PPI file. Total interactions read: {total_ppi_read}")
    
    # Convert to PyTorch tensor
    adj_matrix = torch.FloatTensor(adj_matrix)

    # --- 4. Calculate and Save Network Statistics ---
    total_possible_interactions = n_genes * (n_genes - 1) / 2
    actual_interactions = (adj_matrix > 0).sum().item() / 2
    network_density = actual_interactions / total_possible_interactions if total_possible_interactions > 0 else 0
    
    stats = {
        'n_genes': n_genes,
        'valid_interactions': valid_interactions,
        'total_ppi_read': total_ppi_read,
        'density': float(network_density),
        'timestamp': datetime.now().isoformat(),
        'total_possible_interactions': int(total_possible_interactions),
        'actual_interactions': int(actual_interactions),
        'mean_interaction_score': float(adj_matrix[adj_matrix > 0].mean()) if actual_interactions > 0 else 0,
        'max_interaction_score': float(adj_matrix.max())
    }
    
    with open(ppi_dir / "network_stats.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    logging.info("PPI network processing completed successfully")
    logging.info(f"Network statistics: {json.dumps(stats, indent=2)}")
    
    # --- 5. Save Final Matrix and Clean Up ---
    output_path = ppi_dir / "adjacency_matrix.pt"
    torch.save(adj_matrix, output_path)
    logging.info(f"Saved adjacency matrix to {output_path}")
    logging.info(f"Matrix size: {adj_matrix.size()}")
    
    del adj_matrix, id_mapping_df, expr_df
    gc.collect()

if __name__ == "__main__":
    prepare_adjacency_matrix()