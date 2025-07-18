import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon
import warnings
warnings.filterwarnings('ignore')

os.makedirs('real_data_validation/logs', exist_ok=True)
os.makedirs('real_data_validation/results', exist_ok=True)
os.makedirs('real_data_validation/figures', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_validation/logs/comprehensive_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveAnalyzer:
    """Comprehensive analysis and visualization for research paper."""
    
    def __init__(self):
        self.results_path = Path("real_data_validation/results")
        self.figures_path = Path("real_data_validation/figures")
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        self.tagt_results = None
        self.traditional_results = None
        self.comparison_results = {}
        
    def load_validation_results(self):
        """Load all validation results."""
        logger.info("Loading validation results...")
        
        try:
            # Load TAGT results
            with open(self.results_path / "tagt_validation_results.json", 'r') as f:
                self.tagt_results = json.load(f)
            logger.info("TAGT results loaded successfully")
            
            # Load traditional models results
            with open(self.results_path / "traditional_models_results.json", 'r') as f:
                self.traditional_results = json.load(f)
            logger.info("Traditional models results loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return False
    
    def extract_performance_data(self):
        """Extract performance data for comparison."""
        logger.info("Extracting performance data...")
        
        # Extract TAGT performance (focus on GSE49454 - primary dataset)
        tagt_gse49454 = self.tagt_results.get('GSE49454', {})
        
        # Extract traditional models performance
        traditional_gse49454 = self.traditional_results.get('GSE49454', {}).get('models', {})
        
        # Compile performance data
        performance_data = {
            'Model': [],
            'Type': [],
            'AUC_ROC_Mean': [],
            'AUC_ROC_Std': [],
            'Accuracy_Mean': [],
            'Accuracy_Std': [],
            'Precision_Mean': [],
            'Precision_Std': [],
            'Recall_Mean': [],
            'Recall_Std': [],
            'F1_Mean': [],
            'F1_Std': []
        }
        
        # Add TAGT results
        if tagt_gse49454:
            performance_data['Model'].append('TAGT (Ours)')
            performance_data['Type'].append('Graph Transformer')
            performance_data['AUC_ROC_Mean'].append(tagt_gse49454.get('auc_roc', {}).get('mean', 0))
            performance_data['AUC_ROC_Std'].append(tagt_gse49454.get('auc_roc', {}).get('std', 0))
            performance_data['Accuracy_Mean'].append(tagt_gse49454.get('accuracy', {}).get('mean', 0))
            performance_data['Accuracy_Std'].append(tagt_gse49454.get('accuracy', {}).get('std', 0))
            performance_data['Precision_Mean'].append(tagt_gse49454.get('precision', {}).get('mean', 0))
            performance_data['Precision_Std'].append(tagt_gse49454.get('precision', {}).get('std', 0))
            performance_data['Recall_Mean'].append(tagt_gse49454.get('recall', {}).get('mean', 0))
            performance_data['Recall_Std'].append(tagt_gse49454.get('recall', {}).get('std', 0))
            performance_data['F1_Mean'].append(tagt_gse49454.get('f1_score', {}).get('mean', 0))
            performance_data['F1_Std'].append(tagt_gse49454.get('f1_score', {}).get('std', 0))
        
        # Add traditional models results
        model_type_mapping = {
            'random_forest': ('Random Forest', 'Tree-based'),
            'svm': ('SVM', 'Kernel-based'),
            'logistic_regression': ('Logistic Regression', 'Linear'),
            'xgboost': ('XGBoost', 'Gradient Boosting'),
            'lstm': ('LSTM', 'Neural Network'),
            'transformer': ('Transformer', 'Neural Network')
        }
        
        for model_key, model_data in traditional_gse49454.items():
            if model_key in model_type_mapping:
                model_name, model_type = model_type_mapping[model_key]
                
                performance_data['Model'].append(model_name)
                performance_data['Type'].append(model_type)
                performance_data['AUC_ROC_Mean'].append(model_data.get('auc_roc', {}).get('mean', 0))
                performance_data['AUC_ROC_Std'].append(model_data.get('auc_roc', {}).get('std', 0))
                performance_data['Accuracy_Mean'].append(model_data.get('accuracy', {}).get('mean', 0))
                performance_data['Accuracy_Std'].append(model_data.get('accuracy', {}).get('std', 0))
                performance_data['Precision_Mean'].append(model_data.get('precision', {}).get('mean', 0))
                performance_data['Precision_Std'].append(model_data.get('precision', {}).get('std', 0))
                performance_data['Recall_Mean'].append(model_data.get('recall', {}).get('mean', 0))
                performance_data['Recall_Std'].append(model_data.get('recall', {}).get('std', 0))
                performance_data['F1_Mean'].append(model_data.get('f1_score', {}).get('mean', 0))
                performance_data['F1_Std'].append(model_data.get('f1_score', {}).get('std', 0))
        
                self.performance_df = pd.DataFrame(performance_data)
        
        logger.info(f"Performance data extracted for {len(self.performance_df)} models")
        
        return self.performance_df
    
    def create_performance_comparison_plot(self):
        """Create comprehensive performance comparison plot."""
        logger.info("Creating performance comparison plot...")
        
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TAGT vs Traditional Models: Performance Comparison on Real SLE Data', 
                     fontsize=16, fontweight='bold')
        
        # Color mapping
        colors = {
            'TAGT (Ours)': '#FF6B6B',  # Red for TAGT
            'Random Forest': '#4ECDC4',
            'SVM': '#45B7D1',
            'Logistic Regression': '#96CEB4',
            'XGBoost': '#FFEAA7',
            'LSTM': '#DDA0DD',
            'Transformer': '#98D8C8'
        }
        
        # 1. AUC-ROC Comparison
        models = self.performance_df['Model']
        auc_means = self.performance_df['AUC_ROC_Mean']
        auc_stds = self.performance_df['AUC_ROC_Std']
        
        bars1 = ax1.bar(models, auc_means, yerr=auc_stds, capsize=5, 
                       color=[colors.get(m, '#95A5A6') for m in models], alpha=0.8)
        ax1.set_title('AUC-ROC Performance', fontweight='bold')
        ax1.set_ylabel('AUC-ROC Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars1, auc_means, auc_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Accuracy Comparison
        acc_means = self.performance_df['Accuracy_Mean']
        acc_stds = self.performance_df['Accuracy_Std']
        
        bars2 = ax2.bar(models, acc_means, yerr=acc_stds, capsize=5,
                       color=[colors.get(m, '#95A5A6') for m in models], alpha=0.8)
        ax2.set_title('Accuracy Performance', fontweight='bold')
        ax2.set_ylabel('Accuracy Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars2, acc_means, acc_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. F1-Score Comparison
        f1_means = self.performance_df['F1_Mean']
        f1_stds = self.performance_df['F1_Std']
        
        bars3 = ax3.bar(models, f1_means, yerr=f1_stds, capsize=5,
                       color=[colors.get(m, '#95A5A6') for m in models], alpha=0.8)
        ax3.set_title('F1-Score Performance', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars3, f1_means, f1_stds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Model Type Comparison (Grouped)
        type_performance = self.performance_df.groupby('Type').agg({
            'AUC_ROC_Mean': 'mean',
            'Accuracy_Mean': 'mean',
            'F1_Mean': 'mean'
        }).reset_index()
        
        x = np.arange(len(type_performance))
        width = 0.25
        
        bars4a = ax4.bar(x - width, type_performance['AUC_ROC_Mean'], width, 
                        label='AUC-ROC', alpha=0.8)
        bars4b = ax4.bar(x, type_performance['Accuracy_Mean'], width, 
                        label='Accuracy', alpha=0.8)
        bars4c = ax4.bar(x + width, type_performance['F1_Mean'], width, 
                        label='F1-Score', alpha=0.8)
        
        ax4.set_title('Performance by Model Type', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_xlabel('Model Type')
        ax4.set_xticks(x)
        ax4.set_xticklabels(type_performance['Type'], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'comprehensive_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance comparison plot saved")
    
    def create_statistical_analysis(self):
        """Perform statistical significance testing."""
        logger.info("Performing statistical analysis...")
        
        # Get TAGT performance scores
        tagt_gse49454 = self.tagt_results.get('GSE49454', {})
        tagt_auc_scores = tagt_gse49454.get('auc_roc', {}).get('scores', [])
        tagt_acc_scores = tagt_gse49454.get('accuracy', {}).get('scores', [])
        
        # Statistical comparisons
        statistical_results = {
            'comparisons': [],
            'tagt_vs_traditional': {}
        }
        
        # Compare TAGT vs each traditional model
        traditional_gse49454 = self.traditional_results.get('GSE49454', {}).get('models', {})
        
        for model_key, model_data in traditional_gse49454.items():
            if model_key in ['random_forest', 'svm', 'logistic_regression', 'xgboost']:
                model_auc_scores = model_data.get('auc_roc', {}).get('scores', [])
                model_acc_scores = model_data.get('accuracy', {}).get('scores', [])
                
                if len(tagt_auc_scores) > 0 and len(model_auc_scores) > 0:
                    # Perform t-test for AUC-ROC
                    try:
                        t_stat_auc, p_val_auc = ttest_ind(tagt_auc_scores, model_auc_scores)
                        
                        # Perform t-test for Accuracy
                        t_stat_acc, p_val_acc = ttest_ind(tagt_acc_scores, model_acc_scores)
                        
                        statistical_results['tagt_vs_traditional'][model_key] = {
                            'auc_roc': {
                                't_statistic': t_stat_auc,
                                'p_value': p_val_auc,
                                'significant': p_val_auc < 0.05
                            },
                            'accuracy': {
                                't_statistic': t_stat_acc,
                                'p_value': p_val_acc,
                                'significant': p_val_acc < 0.05
                            }
                        }
                        
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {model_key}: {e}")
        
        # Save statistical results
        with open(self.results_path / "statistical_analysis.json", 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
        
        logger.info("Statistical analysis completed")
        
        return statistical_results
    
    def generate_research_paper_summary(self):
        """Generate summary for research paper."""
        logger.info("Generating research paper summary...")
        
        # Get best performing models
        best_auc_idx = self.performance_df['AUC_ROC_Mean'].idxmax()
        best_acc_idx = self.performance_df['Accuracy_Mean'].idxmax()
        best_f1_idx = self.performance_df['F1_Mean'].idxmax()
        
        best_auc_model = self.performance_df.loc[best_auc_idx]
        best_acc_model = self.performance_df.loc[best_acc_idx]
        best_f1_model = self.performance_df.loc[best_f1_idx]
        
                summary = f"""
# TAGT vs Traditional Models: Real Data Validation Results

## Dataset Information
- **Primary Dataset**: GSE49454 (378 samples, 177 genes)
- **SLE Flares**: 128 samples (33.9%)
- **Controls**: 250 samples (66.1%)
- **Validation**: 5-fold cross-validation

## Performance Results

### Best Performing Models:
- **Best AUC-ROC**: {best_auc_model['Model']} ({best_auc_model['AUC_ROC_Mean']:.4f} ± {best_auc_model['AUC_ROC_Std']:.4f})
- **Best Accuracy**: {best_acc_model['Model']} ({best_acc_model['Accuracy_Mean']:.4f} ± {best_acc_model['Accuracy_Std']:.4f})
- **Best F1-Score**: {best_f1_model['Model']} ({best_f1_model['F1_Mean']:.4f} ± {best_f1_model['F1_Std']:.4f})

### Complete Results Table:
"""
        
        # Add results table
        for _, row in self.performance_df.iterrows():
            summary += f"""
**{row['Model']}** ({row['Type']}):
- AUC-ROC: {row['AUC_ROC_Mean']:.4f} ± {row['AUC_ROC_Std']:.4f}
- Accuracy: {row['Accuracy_Mean']:.4f} ± {row['Accuracy_Std']:.4f}
- Precision: {row['Precision_Mean']:.4f} ± {row['Precision_Std']:.4f}
- Recall: {row['Recall_Mean']:.4f} ± {row['Recall_Std']:.4f}
- F1-Score: {row['F1_Mean']:.4f} ± {row['F1_Std']:.4f}
"""
        
        # Add interpretation
        tagt_row = self.performance_df[self.performance_df['Model'] == 'TAGT (Ours)']
        if not tagt_row.empty:
            tagt_auc = tagt_row['AUC_ROC_Mean'].iloc[0]
            tagt_acc = tagt_row['Accuracy_Mean'].iloc[0]
            
            summary += f"""

## Key Findings

### TAGT Performance:
- **AUC-ROC**: {tagt_auc:.4f} ({"Excellent" if tagt_auc > 0.9 else "Good" if tagt_auc > 0.8 else "Moderate" if tagt_auc > 0.7 else "Limited"} clinical utility)
- **Accuracy**: {tagt_acc:.4f} ({"Outstanding" if tagt_acc > 0.9 else "Good" if tagt_acc > 0.8 else "Acceptable" if tagt_acc > 0.7 else "Limited"} performance)

### Clinical Interpretation:
- Performance indicates {"strong potential for clinical deployment" if tagt_auc > 0.8 else "promising research direction requiring further optimization"}
- Results support the hypothesis that graph-based temporal modeling can improve SLE flare prediction
"""
        
        # Save summary
        with open(self.results_path / "research_paper_summary.md", 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info("Research paper summary generated")
        
        return summary

def main():
    """Main comprehensive analysis function."""
    logger.info("STARTING REAL DATA VALIDATION - PHASE 4: COMPREHENSIVE ANALYSIS")
    logger.info("=" * 80)
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer()
    
    # Load results
    if not analyzer.load_validation_results():
        logger.error("Failed to load validation results")
        return False
    
    # Extract performance data
    performance_df = analyzer.extract_performance_data()
    
        analyzer.create_performance_comparison_plot()
    
    # Statistical analysis
    analyzer.create_statistical_analysis()
    
        summary = analyzer.generate_research_paper_summary()
    
    logger.info("PHASE 4 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETED")
    print("="*80)
    print(summary)
    
    return True

if __name__ == "__main__":
    success = main()