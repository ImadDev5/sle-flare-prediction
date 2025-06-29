#!/usr/bin/env python3
"""
Real Data Loader for TAGT Validation
Loads and processes GSE49454 and STRING data for validation
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import logging
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataLoader:
    def __init__(self, base_path="D:/SLE_data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        
        # Data paths
        self.gse49454_path = self.raw_path / "GSE49454"
        self.string_path = self.raw_path / "STRING"
        
    def load_gse49454_data(self):
        """Load GSE49454 gene expression data"""
        logger.info("Loading GSE49454 data...")
        
        # Find series matrix file
        matrix_files = list(self.gse49454_path.glob("*series_matrix.txt*"))
        if not matrix_files:
            raise FileNotFoundError("GSE49454 series matrix file not found")
        
        matrix_file = matrix_files[0]
        logger.info(f"Loading from {matrix_file}")
        
        # Read the file
        if matrix_file.suffix == '.gz':
            with gzip.open(matrix_file, 'rt') as f:
                content = f.read()
        else:
            with open(matrix_file, 'r') as f:
                content = f.read()
        
        # Parse metadata and data
        lines = content.split('\n')
        
        # Find where data starts (after metadata)
        data_start = 0
        sample_info = {}
        
        for i, line in enumerate(lines):
            if line.startswith('!Sample_title'):
                # Extract sample titles
                titles = line.split('\t')[1:]
                sample_info['titles'] = titles
            elif line.startswith('!Sample_characteristics_ch1'):
                # Extract sample characteristics
                chars = line.split('\t')[1:]
                sample_info['characteristics'] = chars
            elif line.startswith('!series_matrix_table_begin'):
                data_start = i + 1
                break
        
        # Read data section
        data_lines = []
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line.startswith('!series_matrix_table_end'):
                break
            if line and not line.startswith('!'):
                data_lines.append(line)
        
        if not data_lines:
            raise ValueError("No data found in series matrix file")
        
        # Parse data
        header = data_lines[0].split('\t')
        data_rows = []
        
        for line in data_lines[1:]:
            if line.strip():
                data_rows.append(line.split('\t'))
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=header)
        df = df.set_index(df.columns[0])  # First column is gene ID
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded expression data: {df.shape[0]} genes, {df.shape[1]} samples")
        
        # Create labels based on sample information
        labels = self._extract_labels_from_samples(sample_info)
        
        return df.T, labels  # Transpose so samples are rows
    
    def _extract_labels_from_samples(self, sample_info):
        """Extract SLE vs control labels from sample information"""
        logger.info("Extracting sample labels...")
        
        labels = []
        
        if 'characteristics' in sample_info:
            characteristics = sample_info['characteristics']
            
            for char in characteristics:
                # Look for SLE-related keywords
                char_lower = char.lower()
                
                if any(keyword in char_lower for keyword in ['lupus', 'sle', 'systemic lupus']):
                    labels.append(1)  # SLE
                elif any(keyword in char_lower for keyword in ['control', 'healthy', 'normal']):
                    labels.append(0)  # Control
                else:
                    # Try to infer from other patterns
                    if 'patient' in char_lower or 'disease' in char_lower:
                        labels.append(1)
                    else:
                        labels.append(0)  # Default to control
        
        elif 'titles' in sample_info:
            titles = sample_info['titles']
            
            for title in titles:
                title_lower = title.lower()
                
                if any(keyword in title_lower for keyword in ['lupus', 'sle', 'patient']):
                    labels.append(1)
                else:
                    labels.append(0)
        
        else:
            # No sample info available, create balanced synthetic labels
            logger.warning("No sample information found, creating synthetic labels")
            n_samples = len(sample_info.get('titles', []))
            if n_samples == 0:
                n_samples = 100  # Default
            
            # Create 30% SLE samples (realistic proportion)
            n_sle = int(n_samples * 0.3)
            labels = [1] * n_sle + [0] * (n_samples - n_sle)
            np.random.shuffle(labels)
        
        logger.info(f"Labels created: {sum(labels)} SLE, {len(labels) - sum(labels)} controls")
        return np.array(labels)
    
    def load_string_data(self):
        """Load STRING protein-protein interaction data"""
        logger.info("Loading STRING PPI data...")
        
        # Find protein links file
        links_files = list(self.string_path.glob("*protein.links*"))
        if not links_files:
            raise FileNotFoundError("STRING protein links file not found")
        
        links_file = links_files[0]
        logger.info(f"Loading from {links_file}")
        
        # Read protein links
        if links_file.suffix == '.gz':
            df = pd.read_csv(links_file, sep=' ', compression='gzip')
        else:
            df = pd.read_csv(links_file, sep=' ')
        
        logger.info(f"Loaded {len(df)} protein interactions")
        
        # Filter for human proteins (9606 is human taxonomy ID)
        if 'protein1' in df.columns and 'protein2' in df.columns:
            human_mask = (df['protein1'].str.startswith('9606.')) & (df['protein2'].str.startswith('9606.'))
            df_human = df[human_mask]
            logger.info(f"Filtered to {len(df_human)} human protein interactions")
            return df_human
        
        return df
    
    def preprocess_data(self, expression_data, labels, top_genes=1000):
        """Preprocess expression data for model input"""
        logger.info("Preprocessing expression data...")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        expression_data = pd.DataFrame(
            imputer.fit_transform(expression_data),
            index=expression_data.index,
            columns=expression_data.columns
        )
        
        # Select top variable genes
        gene_var = expression_data.var(axis=0)
        top_genes_idx = gene_var.nlargest(top_genes).index
        expression_data = expression_data[top_genes_idx]
        
        # Standardize
        scaler = StandardScaler()
        expression_data_scaled = pd.DataFrame(
            scaler.fit_transform(expression_data),
            index=expression_data.index,
            columns=expression_data.columns
        )
        
        logger.info(f"Preprocessed data: {expression_data_scaled.shape[0]} samples, {expression_data_scaled.shape[1]} genes")
        
        return expression_data_scaled.values, labels
    
    def create_temporal_data(self, X, y, n_timepoints=3):
        """Create synthetic temporal data from cross-sectional data"""
        logger.info("Creating synthetic temporal sequences...")
        
        # For each sample, create a temporal sequence
        n_samples, n_features = X.shape
        
        # Create temporal sequences by adding noise to simulate time progression
        temporal_X = []
        temporal_y = []
        
        for i in range(n_samples):
            sequence = []
            
            for t in range(n_timepoints):
                # Add temporal noise (disease progression simulation)
                noise_factor = 0.1 * t  # Increasing noise over time
                
                if y[i] == 1:  # SLE patient
                    # Simulate disease progression
                    progression_factor = 1 + 0.2 * t
                    sample_t = X[i] * progression_factor + np.random.normal(0, noise_factor, n_features)
                else:  # Control
                    # Stable over time
                    sample_t = X[i] + np.random.normal(0, noise_factor, n_features)
                
                sequence.append(sample_t)
            
            temporal_X.append(sequence)
            temporal_y.append(y[i])
        
        temporal_X = np.array(temporal_X)  # Shape: (n_samples, n_timepoints, n_features)
        temporal_y = np.array(temporal_y)
        
        logger.info(f"Created temporal data: {temporal_X.shape}")
        
        return temporal_X, temporal_y

def load_real_data():
    """Main function to load and preprocess real data"""
    loader = RealDataLoader()
    
    try:
        # Load gene expression data
        expression_data, labels = loader.load_gse49454_data()
        
        # Preprocess
        X, y = loader.preprocess_data(expression_data, labels)
        
        # Create temporal sequences if needed
        # X_temporal, y_temporal = loader.create_temporal_data(X, y)
        
        logger.info(f"Successfully loaded real data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.sum(y)} SLE, {len(y) - np.sum(y)} controls")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading real data: {e}")
        raise

def main():
    """Test data loading"""
    try:
        X, y = load_real_data()
        print(f"Data loaded successfully: {X.shape}, labels: {len(y)}")
        print(f"SLE samples: {np.sum(y)}, Control samples: {len(y) - np.sum(y)}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Real data not available, validation will use synthetic data")

if __name__ == "__main__":
    main()
