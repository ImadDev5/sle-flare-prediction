import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

class ModelEvaluator:
    def __init__(self, model_path, data_paths):
        """Initialize evaluator with model and data paths"""
        self.model_path = model_path
        self.data_paths = data_paths
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = {}
        
        # Create output directory
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Output directory: {self.output_dir}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            model = TAGTModel(
                n_genes=10000,  # Match with training configuration
                hidden_dim=256,
                n_heads=8,
                dropout=0.2
            ).to(self.device)
            
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            
            logging.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def load_data(self):
        """Load evaluation data"""
        try:
            # Load expression data
            expr_df = pd.read_csv(self.data_paths['expression'])
            logging.info(f"Loaded expression data: {expr_df.shape}")
            
            # Load clinical data
            clinical_df = pd.read_csv(self.data_paths['clinical'])
            logging.info(f"Loaded clinical data: {clinical_df.shape}")
            
            # Load adjacency matrix
            adjacency_matrix = torch.load(self.data_paths['adjacency'])
            logging.info(f"Loaded adjacency matrix: {adjacency_matrix.shape}")
            
            return expr_df, clinical_df, adjacency_matrix
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        try:
            # Load model and data
            model = self.load_model()
            expr_df, clinical_df, adjacency_matrix = self.load_data()
            
            # Prepare data
            X = torch.FloatTensor(expr_df.values[:, 1:])  # Skip first column (index)
            y = torch.LongTensor(clinical_df['flare'].values)
            sledai = torch.FloatTensor(clinical_df['current_sledai'].values)
            
            # Move to device
            X = X.to(self.device)
            y = y.to(self.device)
            sledai = sledai.to(self.device)
            adjacency_matrix = adjacency_matrix.to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(X, sledai, adjacency_matrix)
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y.cpu(), predictions.cpu()),
                'precision': precision_score(y.cpu(), predictions.cpu()),
                'recall': recall_score(y.cpu(), predictions.cpu()),
                'f1': f1_score(y.cpu(), predictions.cpu()),
                'roc_auc': roc_auc_score(y.cpu(), probabilities.cpu()),
                'mcc': matthews_corrcoef(y.cpu(), predictions.cpu()),
                'confusion_matrix': confusion_matrix(y.cpu(), predictions.cpu()).tolist(),
                'classification_report': classification_report(
                    y.cpu(),
                    predictions.cpu(),
                    output_dict=True
                )
            }
            
            # Save metrics
            self.metrics = metrics
            self._save_metrics()
            
            # Generate plots
            self._generate_plots(y.cpu(), predictions.cpu(), probabilities.cpu())
            
            # Log results
            logging.info("\nEvaluation Results:")
            for metric, value in metrics.items():
                if metric == 'confusion_matrix':
                    logging.info(f"\nConfusion Matrix:\n{np.array(value)}")
                elif metric == 'classification_report':
                    logging.info(f"\nClassification Report:\n{json.dumps(value, indent=2)}")
                else:
                    logging.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise
    
    def _save_metrics(self):
        """Save evaluation metrics to JSON"""
        metrics_file = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logging.info(f"Saved metrics to {metrics_file}")
    
    def _generate_plots(self, y_true, y_pred, probabilities):
        """Generate evaluation plots"""
        try:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, probabilities)
            roc_auc = roc_auc_score(y_true, probabilities)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(self.output_dir / 'roc_curve.png')
            plt.close()
            
            # Confusion matrix heatmap
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Flare', 'Flare'],
                        yticklabels=['No Flare', 'Flare'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(self.output_dir / 'confusion_matrix.png')
            plt.close()
            
            logging.info("Generated evaluation plots")
            
        except Exception as e:
            logging.error(f"Error generating plots: {str(e)}")
            raise

def main():
    """Main evaluation function"""
    # Data paths
    data_paths = {
        'expression': 'D:/SLE_data/processed/expression_normalized.csv',
        'clinical': 'D:/SLE_data/processed/clinical_data.csv',
        'adjacency': 'D:/SLE_data/processed/ppi/adjacency_matrix.pt'
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='models/best_model.pth',
        data_paths=data_paths
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print final results
    print("\nFinal Evaluation Results:")
    for metric, value in metrics.items():
        if metric == 'confusion_matrix':
            print(f"\nConfusion Matrix:\n{np.array(value)}")
        elif metric == 'classification_report':
            print(f"\nClassification Report:\n{json.dumps(value, indent=2)}")
        else:
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
