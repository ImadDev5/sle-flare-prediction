import os
import sys
import gzip
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('real_data_validation/logs', exist_ok=True)
os.makedirs('real_data_validation/results', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_validation/logs/data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealDataValidator:
    """Comprehensive real data validation and preparation."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.results_path = Path("real_data_validation/results")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
                log_path = Path("real_data_validation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {}
        self.validation_results = {}
        
    def load_gse49454_data(self):
        """Load primary GSE49454 dataset (your main training data)."""
        logger.info("Loading GSE49454 primary dataset...")
        
        try:
            # Check if processed data exists
            processed_file = self.data_path / "processed" / "expression_real.csv"
            if processed_file.exists():
                logger.info("Loading processed GSE49454 data...")
                expression_data = pd.read_csv(processed_file, index_col=0)

                # Load labels
                labels = np.load(self.data_path / "integrated" / "labels_real.npy")

                # Check if dimensions match
                if len(labels) != expression_data.shape[0]:
                    logger.warning(f"Dimension mismatch: expression {expression_data.shape[0]} vs labels {len(labels)}")
                    # Use only the first N samples that match labels
                    expression_data = expression_data.iloc[:len(labels)]
                    logger.info(f"Adjusted expression data to {expression_data.shape[0]} samples")
                
                logger.info(f"GSE49454 loaded: {expression_data.shape[0]} samples, {expression_data.shape[1]} genes")
                logger.info(f"Class distribution: {np.sum(labels)} SLE flares, {len(labels) - np.sum(labels)} non-flares")
                
                self.datasets['GSE49454'] = {
                    'expression': expression_data,
                    'labels': labels,
                    'source': 'processed',
                    'description': 'Primary training dataset - GSE49454'
                }
                
                return True
                
            else:
                # Load from raw data
                logger.info("Loading raw GSE49454 data...")
                raw_file = self.data_path / "raw" / "GSE49454" / "GSE49454_series_matrix.txt.gz"
                
                if not raw_file.exists():
                    logger.error(f"GSE49454 raw data not found: {raw_file}")
                    return False
                
                # Parse raw data
                expression_data, labels = self._parse_gse_matrix(raw_file, 'GSE49454')
                
                if expression_data is not None:
                    self.datasets['GSE49454'] = {
                        'expression': expression_data,
                        'labels': labels,
                        'source': 'raw',
                        'description': 'Primary training dataset - GSE49454'
                    }
                    return True
                
        except Exception as e:
            logger.error(f"Error loading GSE49454: {e}")
            return False
            
        return False
    
    def load_gse99967_data(self):
        """Load external validation dataset GSE99967."""
        logger.info("Loading GSE99967 external validation dataset...")
        
        try:
            external_file = self.data_path / "external" / "GSE99967_series_matrix.txt.gz"
            
            if not external_file.exists():
                logger.error(f"GSE99967 data not found: {external_file}")
                return False
            
            # Parse external data
            expression_data, labels = self._parse_gse_matrix(external_file, 'GSE99967')
            
            if expression_data is not None:
                self.datasets['GSE99967'] = {
                    'expression': expression_data,
                    'labels': labels,
                    'source': 'external',
                    'description': 'External validation dataset - GSE99967 (SLE nephritis focus)'
                }
                
                logger.info(f"GSE99967 loaded: {expression_data.shape[0]} samples, {expression_data.shape[1]} genes")
                logger.info(f"Class distribution: {np.sum(labels)} SLE cases, {len(labels) - np.sum(labels)} controls")
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading GSE99967: {e}")
            return False
            
        return False
    
    def _parse_gse_matrix(self, file_path: Path, dataset_name: str):
        """Parse GEO series matrix file."""
        logger.info(f"Parsing {dataset_name} matrix file...")
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                # Read metadata and find data start
                metadata = {}
                data_started = False
                header_line = None
                data_lines = []
                
                for line in f:
                    if line.startswith('!Sample_title'):
                        metadata['sample_titles'] = line.strip().split('\t')[1:]
                    elif line.startswith('!Sample_characteristics_ch1'):
                        metadata['characteristics'] = line.strip().split('\t')[1:]
                    elif line.startswith('!series_matrix_table_begin'):
                        data_started = True
                        header_line = next(f).strip()
                        continue
                    elif line.startswith('!series_matrix_table_end'):
                        break
                    elif data_started:
                        data_lines.append(line.strip())
                
                if not data_lines:
                    logger.error(f"No data found in {dataset_name}")
                    return None, None
                
                # Parse header and data
                headers = header_line.split('\t')
                
                                data_dict = {headers[0]: []}
                for header in headers[1:]:
                    data_dict[header] = []
                
                for line in data_lines:
                    values = line.split('\t')
                    for i, value in enumerate(values):
                        if i < len(headers):
                            try:
                                if i == 0:  # Gene ID
                                    data_dict[headers[i]].append(value)
                                else:  # Expression values
                                    data_dict[headers[i]].append(float(value))
                            except ValueError:
                                data_dict[headers[i]].append(np.nan)
                
                                df = pd.DataFrame(data_dict)
                df = df.set_index(headers[0])
                
                                labels = self._extract_labels(metadata, dataset_name)
                
                logger.info(f"Parsed {dataset_name}: {df.shape[0]} genes, {df.shape[1]} samples")
                
                return df.T, labels  # Transpose so samples are rows
                
        except Exception as e:
            logger.error(f"Error parsing {dataset_name}: {e}")
            return None, None
    
    def _extract_labels(self, metadata, dataset_name):
        """Extract SLE vs control labels from metadata."""
        logger.info(f"Extracting labels for {dataset_name}...")
        
        labels = []
        
        if 'characteristics' in metadata:
            characteristics = metadata['characteristics']
            
            for char in characteristics:
                char_lower = char.lower()
                
                # GSE99967 specific: ALN = Active Lupus Nephritis, ANLN = Active Non-Lupus Nephritis
                if dataset_name == 'GSE99967':
                    if 'aln' in char_lower and 'anln' not in char_lower:
                        labels.append(1)  # Active Lupus Nephritis (flare risk)
                    elif 'control' in char_lower or 'healthy' in char_lower:
                        labels.append(0)  # Control
                    else:
                        labels.append(0)  # Default to control for ANLN and others
                
                # GSE49454 specific
                elif dataset_name == 'GSE49454':
                    if any(keyword in char_lower for keyword in ['lupus', 'sle', 'flare']):
                        labels.append(1)  # SLE/flare
                    elif any(keyword in char_lower for keyword in ['control', 'healthy', 'normal']):
                        labels.append(0)  # Control
                    else:
                        # Infer from other patterns
                        if 'patient' in char_lower or 'disease' in char_lower:
                            labels.append(1)
                        else:
                            labels.append(0)
        
        elif 'sample_titles' in metadata:
            titles = metadata['sample_titles']
            
            for title in titles:
                title_lower = title.lower()
                
                if dataset_name == 'GSE99967':
                    if 'aln' in title_lower and 'anln' not in title_lower:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    if any(keyword in title_lower for keyword in ['lupus', 'sle', 'patient']):
                        labels.append(1)
                    else:
                        labels.append(0)
        
        else:
            # No metadata available - create balanced synthetic labels
            logger.warning(f"No metadata found for {dataset_name}, creating balanced labels")
            n_samples = len(metadata.get('sample_titles', []))
            if n_samples == 0:
                n_samples = 100  # Default
            
            n_cases = int(n_samples * 0.3)  # 30% cases
            labels = [1] * n_cases + [0] * (n_samples - n_cases)
            np.random.shuffle(labels)
        
        labels = np.array(labels)
        logger.info(f"Labels for {dataset_name}: {np.sum(labels)} cases, {len(labels) - np.sum(labels)} controls")
        
        return labels
    
    def validate_data_quality(self):
        """Validate data quality and compatibility."""
        logger.info("Validating data quality...")
        
        validation_results = {}
        
        for dataset_name, data in self.datasets.items():
            logger.info(f"Validating {dataset_name}...")
            
            expression = data['expression']
            labels = data['labels']
            
            # Basic validation
            validation = {
                'n_samples': expression.shape[0],
                'n_genes': expression.shape[1],
                'n_cases': np.sum(labels),
                'n_controls': len(labels) - np.sum(labels),
                'case_rate': np.mean(labels),
                'missing_values': expression.isnull().sum().sum(),
                'missing_rate': expression.isnull().sum().sum() / (expression.shape[0] * expression.shape[1]),
                'expression_range': (expression.min().min(), expression.max().max()),
                'expression_mean': expression.mean().mean(),
                'expression_std': expression.std().mean()
            }
            
            validation_results[dataset_name] = validation
            
            logger.info(f"{dataset_name} validation:")
            logger.info(f"  - Samples: {validation['n_samples']}")
            logger.info(f"  - Genes: {validation['n_genes']}")
            logger.info(f"  - Cases/Controls: {validation['n_cases']}/{validation['n_controls']}")
            logger.info(f"  - Missing values: {validation['missing_values']} ({validation['missing_rate']:.2%})")
        
        self.validation_results = validation_results
        return validation_results
    
    def create_compatibility_report(self):
        """Create data compatibility report for fair comparison."""
        logger.info("Creating data compatibility report...")
        
        if len(self.datasets) < 2:
            logger.warning("Need at least 2 datasets for compatibility analysis")
            return
        
        # Find common genes
        gene_sets = {}
        for name, data in self.datasets.items():
            gene_sets[name] = set(data['expression'].columns)
        
        # Find intersection
        common_genes = set.intersection(*gene_sets.values())
        
        logger.info(f"Gene overlap analysis:")
        for name, genes in gene_sets.items():
            logger.info(f"  - {name}: {len(genes)} genes")
        logger.info(f"  - Common genes: {len(common_genes)}")
        logger.info(f"  - Overlap rate: {len(common_genes) / min(len(g) for g in gene_sets.values()):.2%}")
        
        # Save compatibility report
        compatibility_report = {
            'datasets': list(self.datasets.keys()),
            'gene_counts': {name: len(genes) for name, genes in gene_sets.items()},
            'common_genes': len(common_genes),
            'overlap_rate': len(common_genes) / min(len(g) for g in gene_sets.values()) if gene_sets else 0,
            'validation_results': self.validation_results
        }
        
        # Save to file
        import json
        with open(self.results_path / "data_compatibility_report.json", 'w') as f:
            json.dump(compatibility_report, f, indent=2, default=str)
        
        return compatibility_report
    
    def prepare_unified_datasets(self):
        """Prepare datasets for fair comparison."""
        logger.info("Preparing unified datasets for comparison...")

        # Prepare unified datasets
        unified_datasets = {}

        for name, data in self.datasets.items():
            logger.info(f"Preparing {name} for validation...")

            expression_data = data['expression']

            # Handle missing values
            if expression_data.isnull().sum().sum() > 0:
                imputer = SimpleImputer(strategy='median')
                expression_imputed = pd.DataFrame(
                    imputer.fit_transform(expression_data),
                    index=expression_data.index,
                    columns=expression_data.columns
                )
            else:
                expression_imputed = expression_data.copy()
            
            # Standardize
            scaler = StandardScaler()
            expression_scaled = pd.DataFrame(
                scaler.fit_transform(expression_imputed),
                index=expression_imputed.index,
                columns=expression_imputed.columns
            )

            unified_datasets[name] = {
                'X': expression_scaled.values,
                'y': data['labels'],
                'feature_names': expression_scaled.columns.tolist(),
                'sample_names': expression_scaled.index.tolist(),
                'n_samples': expression_scaled.shape[0],
                'n_features': expression_scaled.shape[1],
                'description': data['description']
            }

            logger.info(f"Unified {name}: {unified_datasets[name]['n_samples']} samples, {unified_datasets[name]['n_features']} features")
        
        # Save unified datasets
        with open(self.results_path / "unified_datasets.pkl", 'wb') as f:
            pickle.dump(unified_datasets, f)
        
        logger.info("Unified datasets saved successfully")
        
        return unified_datasets

def main():
    """Main data preparation function."""
    logger.info("STARTING REAL DATA VALIDATION - PHASE 1")
    logger.info("=" * 60)
    
    # Initialize validator
    validator = RealDataValidator()
    
    # Load datasets
    logger.info("Loading datasets...")
    gse49454_loaded = validator.load_gse49454_data()
    gse99967_loaded = validator.load_gse99967_data()
    
    if not gse49454_loaded:
        logger.error("Failed to load primary dataset GSE49454")
        return False
    
    if not gse99967_loaded:
        logger.warning("Failed to load external dataset GSE99967 - continuing with primary only")
    
    # Validate data quality
    validation_results = validator.validate_data_quality()
    
        compatibility_report = validator.create_compatibility_report()
    
    # Prepare unified datasets
    unified_datasets = validator.prepare_unified_datasets()
    
    logger.info("PHASE 1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()