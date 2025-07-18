import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.optimized_tagt import create_optimized_model

os.makedirs('real_data_validation/logs', exist_ok=True)
os.makedirs('real_data_validation/results', exist_ok=True)
os.makedirs('real_data_validation/figures', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_validation/logs/correct_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CorrectValidator:
    Generate the CORRECT comparison report.
        
        for model_name, results in self.results.items():
            auc = results['auc_roc']['mean']
            auc_std = results['auc_roc']['std']
            acc = results['accuracy']['mean']
            acc_std = results['accuracy']['std']
            prec = results['precision']['mean']
            prec_std = results['precision']['std']
            rec = results['recall']['mean']
            rec_std = results['recall']['std']
            f1 = results['f1_score']['mean']
            f1_std = results['f1_score']['std']
            
            report += f"| **{model_name}** | {auc:.3f}±{auc_std:.3f} | {acc:.3f}±{acc_std:.3f} | {prec:.3f}±{prec_std:.3f} | {rec:.3f}±{rec_std:.3f} | {f1:.3f}±{f1_std:.3f} |\n"
        
        # Find best model
        best_auc_model = max(self.results.items(), key=lambda x: x[1]['auc_roc']['mean'])
        
        report += f"""

## Key Findings

### Best Performing Model
**{best_auc_model[0]}** achieved the highest AUC-ROC of {best_auc_model[1]['auc_roc']['mean']:.4f}

### TAGT Performance
"""
        
        if 'TAGT (Yours)' in self.results:
            tagt_auc = self.results['TAGT (Yours)']['auc_roc']['mean']
            tagt_acc = self.results['TAGT (Yours)']['accuracy']['mean']
            
            report += f"""
- **AUC-ROC**: {tagt_auc:.4f} ({"Excellent" if tagt_auc > 0.9 else "Good" if tagt_auc > 0.8 else "Moderate" if tagt_auc > 0.7 else "Limited"} clinical utility)
- **Accuracy**: {tagt_acc:.4f}

### Comparison with Traditional Models
"""
            
            traditional_aucs = [results['auc_roc']['mean'] for name, results in self.results.items() if name != 'TAGT (Yours)']
            if traditional_aucs:
                best_traditional = max(traditional_aucs)
                improvement = ((tagt_auc - best_traditional) / best_traditional) * 100
                
                report += f"""
- TAGT vs Best Traditional: {improvement:+.1f}% {"improvement" if improvement > 0 else "difference"}
- TAGT {"outperforms" if tagt_auc > best_traditional else "underperforms"} traditional models
"""
        
        report += f"""

## Clinical Assessment
- **Dataset**: Real SLE patient data with temporal sequences
- **Validation**: Proper cross-validation on YOUR training data
- **Architecture**: TAGT tested with correct temporal + graph structure
- **Comparison**: Fair comparison on identical 1000-dimensional dataset

This represents the CORRECT validation of your TAGT model!
"""
        
        # Save report
        with open(self.results_path / "correct_validation_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("CORRECT validation report generated")
        
        return report

def main():
    """Run the CORRECT validation."""
    logger.info("🎯 STARTING CORRECT VALIDATION")
    
    validator = CorrectValidator()
    success = validator.run_correct_validation()
    
    if success:
        logger.info("🎉 CORRECT VALIDATION COMPLETED SUCCESSFULLY!")
        return True
    else:
        logger.error("❌ CORRECT VALIDATION FAILED")
        return False

if __name__ == "__main__":
    success = main()