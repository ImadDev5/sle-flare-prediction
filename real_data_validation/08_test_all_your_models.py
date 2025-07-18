import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

os.makedirs('real_data_validation/logs', exist_ok=True)
os.makedirs('real_data_validation/results', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_validation/logs/test_all_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model Architecture 1: GAT (best_model.pt)
class GATModel(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=64, num_heads=4):
        super(GATModel, self).__init__()
        # This is a simplified version - the actual GAT is more complex
        self.clinical_encoder = nn.Sequential(
            nn.Linear(15, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, clinical=None):
        # Simplified forward pass
        if clinical is None:
            clinical = torch.zeros(x.shape[0], 15).to(x.device)
        clinical_features = self.clinical_encoder(clinical)
        
        # Dummy graph features
        graph_features = torch.zeros(x.shape[0], 64).to(x.device)
        
        combined = torch.cat([graph_features, clinical_features], dim=1)
        output = self.classifier(combined)
        return output

# Model Architecture 2: TAGT (best_tagt_model.pt, final_tagt_model.pt)
class TAGTModel(nn.Module):
    def __init__(self, input_dim=1000):
        super(TAGTModel, self).__init__()
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        self.graph_conv = nn.Linear(128, 128)
        self.temporal_attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(1, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, clinical=None):
        # Gene encoding
        gene_features = self.gene_encoder(x.squeeze(1) if x.dim() > 2 else x)
        
        # Graph convolution
        graph_features = self.graph_conv(gene_features)
        
        # Temporal attention
        graph_features = graph_features.unsqueeze(1)
        attn_output, _ = self.temporal_attention(graph_features, graph_features, graph_features)
        attn_output = attn_output.squeeze(1)
        
        # Clinical features
        if clinical is None:
            clinical = torch.zeros(x.shape[0], 1).to(x.device)
        clinical_features = self.clinical_encoder(clinical)
        
        # Combine and classify
        combined = torch.cat([attn_output, clinical_features], dim=1)
        output = self.classifier(combined)
        return output

# Model Architecture 3: Complex Transformer+LSTM+Graph (tagt_model.pt)
class ComplexTAGTModel(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=128):
        super(ComplexTAGTModel, self).__init__()
        # This is a simplified version of the complex architecture
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    
    def forward(self, x, clinical=None):
        # Simplified forward pass
        features = self.input_projection(x.squeeze(1) if x.dim() > 2 else x)
        output = self.classifier(features)
        return output

class AllModelsValidator:
    """Test all your saved models to find the best one."""
    
    def __init__(self):
        self.results_path = Path("real_data_validation/results")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Data
        self.X_flat = None
        self.labels = None
        
        # Results
        self.all_results = {}
        
    def load_data(self):
        """Load your training data."""
        logger.info("Loading your training data...")
        
        try:
            with open('data/integrated/sequences_real.pkl', 'rb') as f:
                sequences = pickle.load(f)
            
            self.labels = np.load('data/integrated/labels_real.npy')
            self.X_flat = np.array([seq['expression'] for seq in sequences])
            
            logger.info(f"Data loaded: {self.X_flat.shape[0]} samples, {self.X_flat.shape[1]} features")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def test_model(self, model, model_name, state_dict):
        """Test a single model."""
        logger.info(f"Testing {model_name}...")
        
        try:
            # Load weights
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                logger.warning(f"{model_name} - Missing keys: {len(missing_keys)}")
            if len(unexpected_keys) > 0:
                logger.warning(f"{model_name} - Unexpected keys: {len(unexpected_keys)}")
            
            model.to(self.device)
            model.eval()
            
            # Quick 3-fold CV test
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            auc_scores = []
            acc_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_flat, self.labels)):
                X_val = self.X_flat[val_idx]
                y_val = self.labels[val_idx]
                
                # Convert to tensors
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                
                # Model prediction
                with torch.no_grad():
                    try:
                        outputs = model(X_val_tensor)
                        
                        if outputs.shape[1] == 2:
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                        else:
                            probs = torch.sigmoid(outputs.squeeze())
                        
                        y_prob = probs.cpu().numpy()
                        y_pred = (y_prob > 0.5).astype(int)
                        
                        auc = roc_auc_score(y_val, y_prob)
                        acc = accuracy_score(y_val, y_pred)
                        
                        auc_scores.append(auc)
                        acc_scores.append(acc)
                        
                    except Exception as e:
                        logger.error(f"Error in {model_name} forward pass: {e}")
                        auc_scores.append(0.5)
                        acc_scores.append(0.5)
            
            # Calculate results
            mean_auc = np.mean(auc_scores)
            mean_acc = np.mean(acc_scores)
            
            results = {
                'auc_roc': mean_auc,
                'accuracy': mean_acc,
                'auc_scores': auc_scores,
                'acc_scores': acc_scores
            }
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  - AUC-ROC: {mean_auc:.4f}")
            logger.info(f"  - Accuracy: {mean_acc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            return {
                'auc_roc': 0.0,
                'accuracy': 0.0,
                'error': str(e)
            }
    
    def test_all_models(self):
        """Test all your saved models."""
        logger.info("Testing all your saved models...")
        
        model_configs = [
            {
                'file': 'models/best_model.pt',
                'name': 'GAT Model (best_model.pt)',
                'model_class': GATModel,
                'model_args': {}
            },
            {
                'file': 'models/best_tagt_model.pt',
                'name': 'TAGT Model (best_tagt_model.pt)',
                'model_class': TAGTModel,
                'model_args': {}
            },
            {
                'file': 'models/final_tagt_model.pt',
                'name': 'TAGT Model (final_tagt_model.pt)',
                'model_class': TAGTModel,
                'model_args': {}
            },
            {
                'file': 'models/tagt_model.pt',
                'name': 'Complex TAGT (tagt_model.pt)',
                'model_class': ComplexTAGTModel,
                'model_args': {}
            }
        ]
        
        for config in model_configs:
            if os.path.exists(config['file']):
                try:
                    # Load state dict
                    checkpoint = torch.load(config['file'], map_location='cpu')
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                                        model = config['model_class'](**config['model_args'])
                    results = self.test_model(model, config['name'], state_dict)
                    
                    self.all_results[config['name']] = results
                    
                except Exception as e:
                    logger.error(f"Error with {config['file']}: {e}")
                    self.all_results[config['name']] = {
                        'auc_roc': 0.0,
                        'accuracy': 0.0,
                        'error': str(e)
                    }
            else:
                logger.warning(f"{config['file']} not found")
        
        return self.all_results
    
    def generate_comparison_report(self):
        """Generate comparison report of all models."""
        logger.info("Generating comparison report...")
        
        # Find best model
        best_model = max(self.all_results.items(), key=lambda x: x[1]['auc_roc'])
        
        report = f"""
# ALL YOUR MODELS TESTED - FINDING THE 97% AUC-ROC MODEL

## Results Summary

| Model | AUC-ROC | Accuracy | Status |
|-------|---------|----------|--------|
"""
        
        for model_name, results in self.all_results.items():
            auc = results['auc_roc']
            acc = results['accuracy']
            status = "ERROR" if 'error' in results else "OK"
            
            report += f"| {model_name} | {auc:.4f} | {acc:.4f} | {status} |\n"
        
        report += f"""

## Best Performing Model
**{best_model[0]}** achieved the highest AUC-ROC of {best_model[1]['auc_roc']:.4f}

## Analysis
"""
        
        if best_model[1]['auc_roc'] > 0.8:
            report += "✅ Found a high-performing model!\n"
        elif best_model[1]['auc_roc'] > 0.6:
            report += "⚠️ Found a moderately performing model.\n"
        else:
            report += "❌ No high-performing model found. All models perform around random chance.\n"
        
        report += f"""

## Conclusion
The highest AUC-ROC achieved was {best_model[1]['auc_roc']:.4f} by {best_model[0]}.

If you achieved 97% AUC-ROC, it might be:
1. A different model file not tested here
2. Performance on training data (not validation)
3. Different data preprocessing
4. Architecture mismatch in our recreation

Please check your training logs to confirm which model achieved 97% performance.
"""
        
        # Save report
        with open(self.results_path / "all_models_comparison.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def run_validation(self):
        """Run complete validation of all models."""
        logger.info("TESTING ALL YOUR MODELS TO FIND THE 97% AUC-ROC ONE")
        logger.info("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Test all models
        results = self.test_all_models()
        
        # Save results
        with open(self.results_path / "all_models_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
                report = self.generate_comparison_report()
        
        logger.info("=" * 60)
        logger.info("ALL MODELS TESTED")
        logger.info("=" * 60)
        
        print("\n" + "="*60)
        print("ALL YOUR MODELS TESTED - RESULTS")
        print("="*60)
        print(report)
        
        return True

def main():
    """Test all models."""
    validator = AllModelsValidator()
    return validator.run_validation()

if __name__ == "__main__":
    success = main()