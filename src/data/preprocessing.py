"""Main training script for TAGT model"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

print("=" * 80)
print("CREATING INTEGRATED DATASET FOR TAGT MODEL")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load all data
print("\n1. Loading processed data...")
expression_df = pd.read_csv("data/processed/expression_normalized.csv", index_col=0)
clinical_df = pd.read_csv("data/processed/clinical_data.csv", index_col=0)

# Load PPI network data
with open("data/processed/ppi_network_data.pkl", 'rb') as f:
    ppi_data = pickle.load(f)
adj_matrix = ppi_data['adjacency_matrix']
gene_list = ppi_data['gene_list']

print(f"   Expression shape: {expression_df.shape}")
print(f"   Clinical samples: {len(clinical_df)}")
print(f"   PPI network: {adj_matrix.shape}")

print("\n2. Creating temporal sequences...")
patients = clinical_df['patient_id'].unique()
sequences = []
labels = []

for patient in patients:
    patient_data = clinical_df[clinical_df['patient_id'] == patient].sort_values('visit')

    if len(patient_data) >= 2:  # Need at least 2 visits
        for i in range(len(patient_data) - 1):
            current_visit = patient_data.iloc[i]
            next_visit = patient_data.iloc[i + 1]

            # Get expression data for current visit
            # The sample ID is the index of the clinical data (e.g., PATIENT_0_V1)
            sample_id = current_visit.name  # This gets the index (sample ID)
            if sample_id in expression_df.index:
                expression_vector = expression_df.loc[sample_id].values

                                sequences.append({
                    'patient_id': patient,
                    'visit_from': current_visit['visit'],
                    'visit_to': next_visit['visit'],
                    'expression': expression_vector,
                    'current_sledai': current_visit['sledai'],
                    'next_sledai': next_visit['sledai'],
                    'current_flare': current_visit['flare'],
                    'next_flare': next_visit['flare']
                })

                # Label is whether there's a flare in next visit
                labels.append(next_visit['flare'])

print(f"   Created {len(sequences)} sequences from {len(patients)} patients")
print(f"   Flare rate: {sum(labels)/len(labels)*100:.1f}%")

# Save integrated dataset
print("\n3. Saving integrated dataset...")
output_dir = "data/integrated"
os.makedirs(output_dir, exist_ok=True)

# Save sequences
sequences_df = pd.DataFrame(sequences)
sequences_df.to_pickle(os.path.join(output_dir, "sequences.pkl"))

# Save labels
np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))

# Save adjacency matrix (copy)
np.save(os.path.join(output_dir, "adjacency_matrix.npy"), adj_matrix)

# Save gene list (copy)
with open(os.path.join(output_dir, "gene_list.pkl"), 'wb') as f:
    pickle.dump(gene_list, f)

print("\n4. Dataset statistics:")
print(f"   Total sequences: {len(sequences)}")
print(f"   Positive samples (flares): {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"   Negative samples (no flare): {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
print(f"   Features per sample: {len(expression_vector)}")
print(f"   Average SLEDAI change: {np.mean([s['next_sledai'] - s['current_sledai'] for s in sequences]):.1f}")
print(f"   Average visit interval: {np.mean([s['visit_to'] - s['visit_from'] for s in sequences]):.1f} visits")

print("\n" + "=" * 80)
print("INTEGRATED DATASET CREATION COMPLETE!")
print("=" * 80)
print(f"Saved to: {output_dir}")
print("Files created:")
print("  - sequences.pkl")
print("  - labels.npy")
print("  - adjacency_matrix.npy")
print("  - gene_list.pkl")
print("\nReady for TAGT model training!")
print("=" * 80)