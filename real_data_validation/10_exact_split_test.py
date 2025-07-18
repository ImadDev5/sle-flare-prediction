import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.optimized_tagt import create_optimized_model

os.makedirs('real_data_validation/logs', exist_ok=True)
os.makedirs('real_data_validation/results', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_validation/logs/exact_split_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExactSplitValidator:
    """Test with YOUR exact train/test split."""
    
    def __init__(self):
        self.results_path = Path("real_data_validation/results")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Data storage
        self.sequences = None
        self.labels = None
        self.adjacency = None
        self.model = None
        self.config = None
        
    def load_data_and_split_exactly_like_training(self):
        """Load data and split EXACTLY like your training."""
        logger.info("Loading data and creating EXACT same split as your training...")
        
        try:
            # Load sequences (temporal data)
            with open('data/integrated/sequences_real.pkl', 'rb') as f:
                self.sequences = pickle.load(f)
            
            # Load labels
            self.labels = np.load('data/integrated/labels_real.npy')
            
            # Load adjacency matrix
            adjacency_sparse = sp.load_npz('data/processed/adjacency_real.npz')
            self.adjacency = adjacency_sparse.toarray().astype(np.float32)
            
            logger.info(f"Data loaded: {len(self.sequences)} sequences, {self.labels.shape[0]} labels")
            logger.info(f"Flare rate: {np.mean(self.labels):.2%}")
            
                        indices = np.arange(len(self.sequences))
            
            self.train_idx, self.test_idx = train_test_split(
                indices, 
                test_size=0.2, 
                random_state=42,  # EXACT same random state
                stratify=self.labels
            )
            
            logger.info(f"EXACT split created:")
            logger.info(f"  - Train: {len(self.train_idx)} samples")
            logger.info(f"  - Test: {len(self.test_idx)} samples")
            logger.info(f"  - Train flare rate: {np.mean(self.labels[self.train_idx]):.2%}")
            logger.info(f"  - Test flare rate: {np.mean(self.labels[self.test_idx]):.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def load_model(self):
        """Load your trained model."""
        logger.info("Loading your trained model...")
        
        try:
            # Load config
            if os.path.exists("configs/optimized_tagt_config.json"):
                with open("configs/optimized_tagt_config.json", 'r') as f:
                    self.config = json.load(f)
                logger.info("Loaded config from configs/optimized_tagt_config.json")
            else:
                # Default config
                self.config = {
                    'model_architecture': {
                        'n_genes': 1000,
                        'hidden_dim': 256,
                        'num_graph_layers': 3,
                        'num_heads': 8,
                        'temporal_hidden_dim': 128,
                        'clinical_dim': 15,
                        'dropout': 0.1
                    }
                }
                logger.info("Using default config")
            
                        self.model = create_optimized_model(self.config)
            
            # Load weights
            model_path = "results/best_optimized_model.pth"
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model loaded from {model_path}")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def test_on_exact_test_set(self):
        """Test on the EXACT same test set used in training."""
        logger.info("Testing on EXACT same test set as training...")
        
        try:
            # Get test data
            test_sequences = [self.sequences[i] for i in self.test_idx]
            y_test = self.labels[self.test_idx]
            
            logger.info(f"Test set: {len(test_sequences)} samples")
            
            # Prepare batch data
            batch_expressions = []
            batch_clinical = []
            
            for seq in test_sequences:
                # Gene expression
                batch_expressions.append(seq['expression'])
                
                # Clinical features
                clinical_features = np.zeros(15)
                clinical_features[0] = seq.get('current_sledai', 0)
                clinical_features[1] = seq.get('next_sledai', 0)
                clinical_features[2] = seq.get('current_flare', 0)
                clinical_features[3] = seq.get('next_flare', 0)
                batch_clinical.append(clinical_features)
            
            # Convert to tensors
            batch_size = len(test_sequences)
            X_expr = torch.FloatTensor(np.array(batch_expressions)).unsqueeze(1).to(self.device)
            X_clinical = torch.FloatTensor(np.array(batch_clinical)).to(self.device)
            adjacency_batch = torch.FloatTensor(self.adjacency).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            
            # Model prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_expr, adjacency_batch, X_clinical)
                
                # Extract predictions
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'probabilities' in outputs:
                        probs = outputs['probabilities']
                        logits = torch.log(probs / (1 - probs + 1e-8))
                    else:
                        logits = outputs.get('output', list(outputs.values())[0])
                else:
                    logits = outputs
                
                # Get probabilities
                if logits.dim() > 1 and logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                else:
                    probs = torch.sigmoid(logits.squeeze())
                
                y_prob = probs.cpu().numpy()
                y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            auc_roc = roc_auc_score(y_test, y_prob)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results = {
                'auc_roc': auc_roc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'test_size': len(y_test),
                'predictions': {
                    'y_true': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_prob': y_prob.tolist()
                }
            }
            
            logger.info("RESULTS ON EXACT SAME TEST SET AS TRAINING:")
            logger.info(f"  - AUC-ROC: {auc_roc:.4f}")
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1-Score: {f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in testing: {e}")
            return None
    
    def run_exact_validation(self):
        """Run validation with exact same setup as training."""
        logger.info("TESTING WITH EXACT SAME SETUP AS YOUR 97% TRAINING")
        logger.info("=" * 60)
        
        # Load data and create exact split
        if not self.load_data_and_split_exactly_like_training():
            return False
        
        # Load model
        if not self.load_model():
            return False
        
        # Test on exact test set
        results = self.test_on_exact_test_set()
        
        if results is None:
            return False
        
        # Save results
        with open(self.results_path / "exact_split_test_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Compare with training logs
        training_auc = 0.9715  # From your training logs
        training_acc = 0.8816  # From your training logs
        
        logger.info("=" * 60)
        logger.info("COMPARISON WITH YOUR TRAINING RESULTS:")
        logger.info(f"Training AUC-ROC: {training_auc:.4f}")
        logger.info(f"Current AUC-ROC:  {results['auc_roc']:.4f}")
        logger.info(f"Difference:       {results['auc_roc'] - training_auc:+.4f}")
        logger.info("")
        logger.info(f"Training Accuracy: {training_acc:.4f}")
        logger.info(f"Current Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"Difference:        {results['accuracy'] - training_acc:+.4f}")
        
        if abs(results['auc_roc'] - training_auc) < 0.01:
            logger.info("✅ MATCH! Results are very close to training!")
        elif abs(results['auc_roc'] - training_auc) < 0.05:
            logger.info("⚠️ CLOSE! Results are reasonably close to training.")
        else:
            logger.info("❌ MISMATCH! Results differ significantly from training.")
        
        logger.info("=" * 60)
        
        return True

def main():
    """Run exact split validation."""
    validator = ExactSplitValidator()
    return validator.run_exact_validation()

if __name__ == "__main__":
    success = main()