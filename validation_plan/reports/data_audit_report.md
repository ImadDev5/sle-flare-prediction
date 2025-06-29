
# TAGT DATA AUDIT REPORT
Generated: 2025-06-21 02:18:12.311035

## GSE49454 Gene Expression Data
- Exists: False
- Files found: []
- Samples: 0
- Genes: 0

## STRING PPI Data
- Exists: False
- Links file: None
- Info file: None
- Interactions: 0

## Processed Data
- Exists: False
- Files: 0
- Size: 0.00 MB

## Current Model Data Usage
- Files checked: ['src/data/preprocessing.py', 'src/data/ppi_network.py', 'src/training/train.py', 'experiments/baseline_comparison.py']
- Datasets referenced: ['STRING']
- Data paths found: ['data/processed/expression_normalized.csv', 'data/processed/clinical_data.csv', 'D:\\SLE_data\\processed\\expression_normalized.csv']

## CRITICAL FINDINGS
X GSE49454 data not found - cannot validate gene expression claims
X STRING data not found - cannot validate PPI network claims
! Small sample size (0) - may not support 847 patient claim
