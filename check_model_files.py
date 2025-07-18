"""
Check all model files to find the one with best performance
"""

import torch
import os

model_files = [
    'models/best_model.pt',
    'models/best_tagt_model.pt', 
    'models/final_tagt_model.pt',
    'models/tagt_model.pt'
]

print("=== CHECKING ALL MODEL FILES ===")

for model_file in model_files:
    if os.path.exists(model_file):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            print(f"\n{model_file}:")
            
            if isinstance(checkpoint, dict):
                print(f"  Keys: {list(checkpoint.keys())}")
                
                # Look for performance metrics
                if 'best_val_auc' in checkpoint:
                    print(f"  Best Val AUC: {checkpoint['best_val_auc']}")
                if 'val_auc' in checkpoint:
                    print(f"  Val AUC: {checkpoint['val_auc']}")
                if 'epoch' in checkpoint:
                    print(f"  Epoch: {checkpoint['epoch']}")
                if 'best_epoch' in checkpoint:
                    print(f"  Best Epoch: {checkpoint['best_epoch']}")
                if 'train_loss' in checkpoint:
                    print(f"  Train Loss: {checkpoint['train_loss']}")
                if 'val_loss' in checkpoint:
                    print(f"  Val Loss: {checkpoint['val_loss']}")
                    
                # Check model state dict size
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  Model parameters: {len(state_dict)} layers")
                    
                    # Check key layer dimensions
                    if 'gene_encoder.0.weight' in state_dict:
                        print(f"  Gene encoder input: {state_dict['gene_encoder.0.weight'].shape}")
                    if 'classifier.0.weight' in state_dict:
                        print(f"  Classifier input: {state_dict['classifier.0.weight'].shape}")
                        
            else:
                print(f"  Direct state dict with {len(checkpoint)} parameters")
                
        except Exception as e:
            print(f"  Error loading {model_file}: {e}")
    else:
        print(f"{model_file}: Not found")

print("\n=== SUMMARY ===")
print("Looking for the model with highest AUC-ROC performance...")