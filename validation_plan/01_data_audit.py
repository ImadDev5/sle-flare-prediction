"""
TAGT Validation Plan - Phase 1: Data Audit & Baseline Establishment
"""

import os
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAuditor:
    def __init__(self, base_path="D:/SLE_data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        
        # Expected data locations
        self.gse49454_path = self.raw_path / "GSE49454"
        self.string_path = self.raw_path / "STRING"
        
        self.audit_results = {}
    
    def audit_gse49454_data(self):
        """Audit GSE49454 gene expression dataset"""
        logger.info("Auditing GSE49454 dataset...")
        
        results = {
            'exists': False,
            'files_found': [],
            'sample_count': 0,
            'gene_count': 0,
            'sle_samples': 0,
            'control_samples': 0,
            'has_clinical_data': False,
            'temporal_data': False
        }
        
        if not self.gse49454_path.exists():
            logger.warning(f"GSE49454 path not found: {self.gse49454_path}")
            return results
        
        results['exists'] = True
        
        # Check for series matrix file
        matrix_files = list(self.gse49454_path.glob("*series_matrix.txt*"))
        soft_files = list(self.gse49454_path.glob("*.soft*"))
        
        results['files_found'] = [f.name for f in matrix_files + soft_files]
        
        # Try to load and analyze the data
        if matrix_files:
            try:
                # Handle compressed files
                matrix_file = matrix_files[0]
                if matrix_file.suffix == '.gz':
                    with gzip.open(matrix_file, 'rt') as f:
                        # Read metadata lines (start with !)
                        metadata_lines = []
                        for line in f:
                            if line.startswith('!'):
                                metadata_lines.append(line.strip())
                            else:
                                break
                        
                        # Reset and read data
                        f.seek(0)
                        # Skip metadata lines
                        for line in f:
                            if not line.startswith('!'):
                                break
                        
                        # Read the actual data
                        data = pd.read_csv(f, sep='\t', index_col=0)
                else:
                    # Read metadata first
                    with open(matrix_file, 'r') as f:
                        metadata_lines = []
                        for line in f:
                            if line.startswith('!'):
                                metadata_lines.append(line.strip())
                            else:
                                break
                    
                    # Read data
                    data = pd.read_csv(matrix_file, sep='\t', index_col=0, comment='!')
                
                results['sample_count'] = data.shape[1]
                results['gene_count'] = data.shape[0]
                
                # Analyze metadata for SLE vs control information
                sle_keywords = ['lupus', 'sle', 'systemic lupus erythematosus']
                control_keywords = ['control', 'healthy', 'normal']
                
                # Extract sample information from metadata
                sample_info = {}
                for line in metadata_lines:
                    if 'Sample_title' in line or 'Sample_characteristics' in line:
                        # Parse sample information
                        pass  # Will implement based on actual file structure
                
                logger.info(f"GSE49454: {results['sample_count']} samples, {results['gene_count']} genes")
                
            except Exception as e:
                logger.error(f"Error reading GSE49454 data: {e}")
        
        return results
    
    def audit_string_data(self):
        """Audit STRING protein-protein interaction data"""
        logger.info("Auditing STRING PPI dataset...")
        
        results = {
            'exists': False,
            'protein_links_file': None,
            'protein_info_file': None,
            'interaction_count': 0,
            'protein_count': 0,
            'human_proteins': 0
        }
        
        if not self.string_path.exists():
            logger.warning(f"STRING path not found: {self.string_path}")
            return results
        
        results['exists'] = True
        
        # Look for protein links file
        links_files = list(self.string_path.glob("*protein.links*"))
        info_files = list(self.string_path.glob("*protein.info*"))
        
        if links_files:
            results['protein_links_file'] = links_files[0].name
            
            try:
                # Read a sample of the links file
                links_file = links_files[0]
                if links_file.suffix == '.gz':
                    with gzip.open(links_file, 'rt') as f:
                        # Read header
                        header = f.readline().strip().split()
                        # Count lines (approximate)
                        line_count = sum(1 for _ in f)
                        results['interaction_count'] = line_count
                else:
                    with open(links_file, 'r') as f:
                        header = f.readline().strip().split()
                        line_count = sum(1 for _ in f)
                        results['interaction_count'] = line_count
                
                logger.info(f"STRING: ~{results['interaction_count']} interactions")
                
            except Exception as e:
                logger.error(f"Error reading STRING data: {e}")
        
        if info_files:
            results['protein_info_file'] = info_files[0].name
        
        return results
    
    def audit_processed_data(self):
        """Audit processed data directory"""
        logger.info("Auditing processed data...")
        
        results = {
            'exists': False,
            'files': [],
            'subdirectories': [],
            'total_size_mb': 0
        }
        
        if not self.processed_path.exists():
            logger.warning(f"Processed data path not found: {self.processed_path}")
            return results
        
        results['exists'] = True
        
        # List all files and directories
        for item in self.processed_path.rglob('*'):
            if item.is_file():
                results['files'].append(str(item.relative_to(self.processed_path)))
                results['total_size_mb'] += item.stat().st_size / (1024 * 1024)
            elif item.is_dir():
                results['subdirectories'].append(str(item.relative_to(self.processed_path)))
        
        logger.info(f"Processed data: {len(results['files'])} files, {results['total_size_mb']:.2f} MB")
        
        return results
    
    def check_current_model_data_usage(self):
        """Check what data the current TAGT model actually uses"""
        logger.info("Checking current model data usage...")
        
        # Look for data loading code in the repository
        repo_files = [
            'src/data/preprocessing.py',
            'src/data/ppi_network.py',
            'src/training/train.py',
            'experiments/baseline_comparison.py'
        ]
        
        data_usage = {
            'files_checked': [],
            'data_paths_found': [],
            'datasets_referenced': [],
            'preprocessing_steps': []
        }
        
        for file_path in repo_files:
            if os.path.exists(file_path):
                data_usage['files_checked'].append(file_path)
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                        # Look for data paths
                        if 'GSE49454' in content:
                            data_usage['datasets_referenced'].append('GSE49454')
                        if 'STRING' in content or 'protein' in content.lower():
                            data_usage['datasets_referenced'].append('STRING')
                        
                        # Look for file paths
                        import re
                        path_patterns = [
                            r'["\']([^"\']*\.csv)["\']',
                            r'["\']([^"\']*\.txt)["\']',
                            r'["\']([^"\']*\.gz)["\']',
                            r'["\']([^"\']*\.npz)["\']'
                        ]
                        
                        for pattern in path_patterns:
                            matches = re.findall(pattern, content)
                            data_usage['data_paths_found'].extend(matches)
                
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
        
        return data_usage
    
    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        logger.info("Generating audit report...")
        
        # Run all audits
        self.audit_results['gse49454'] = self.audit_gse49454_data()
        self.audit_results['string'] = self.audit_string_data()
        self.audit_results['processed'] = self.audit_processed_data()
        self.audit_results['model_usage'] = self.check_current_model_data_usage()
        
                report = f"""
# TAGT DATA AUDIT REPORT
Generated: {pd.Timestamp.now()}

## GSE49454 Gene Expression Data
- Exists: {self.audit_results['gse49454']['exists']}
- Files found: {self.audit_results['gse49454']['files_found']}
- Samples: {self.audit_results['gse49454']['sample_count']}
- Genes: {self.audit_results['gse49454']['gene_count']}

## STRING PPI Data
- Exists: {self.audit_results['string']['exists']}
- Links file: {self.audit_results['string']['protein_links_file']}
- Info file: {self.audit_results['string']['protein_info_file']}
- Interactions: {self.audit_results['string']['interaction_count']}

## Processed Data
- Exists: {self.audit_results['processed']['exists']}
- Files: {len(self.audit_results['processed']['files'])}
- Size: {self.audit_results['processed']['total_size_mb']:.2f} MB

## Current Model Data Usage
- Files checked: {self.audit_results['model_usage']['files_checked']}
- Datasets referenced: {self.audit_results['model_usage']['datasets_referenced']}
- Data paths found: {self.audit_results['model_usage']['data_paths_found']}

## CRITICAL FINDINGS
"""
        
        # Add critical findings
        if not self.audit_results['gse49454']['exists']:
            report += "X GSE49454 data not found - cannot validate gene expression claims\n"

        if not self.audit_results['string']['exists']:
            report += "X STRING data not found - cannot validate PPI network claims\n"

        if self.audit_results['gse49454']['sample_count'] < 100:
            report += f"! Small sample size ({self.audit_results['gse49454']['sample_count']}) - may not support 847 patient claim\n"
        
        return report

def main():
    """Run the data audit"""
    auditor = DataAuditor()
    report = auditor.generate_audit_report()
    
    # Save report
    os.makedirs('validation_plan/reports', exist_ok=True)
    with open('validation_plan/reports/data_audit_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    return auditor.audit_results

if __name__ == "__main__":
    results = main()