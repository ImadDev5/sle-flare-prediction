import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import pickle
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)

class ModelDeployer:
    def __init__(self, model_path, config_path):
        """Initialize model deployer"""
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create deployment directory
        self.deployment_dir = Path("deployment")
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Deployment directory: {self.deployment_dir}")
    
    def load_model(self):
        """Load and optimize model for deployment"""
        try:
            # Load model
            model = TAGTModel(
                n_genes=10000,
                hidden_dim=256,
                n_heads=8,
                dropout=0.2
            ).to(self.device)
            
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            
            # Convert to TorchScript
            logging.info("Converting model to TorchScript...")
            example_input = (
                torch.randn(1, 10000).to(self.device),  # Expression
                torch.randn(1, 1).to(self.device),      # SLEDAI
                torch.randn(10000, 10000).to(self.device)  # Adjacency
            )
            
            scripted_model = torch.jit.trace(model, example_input)
            
            # Save TorchScript model
            torchscript_path = self.deployment_dir / 'model.pt'
            torch.jit.save(scripted_model, torchscript_path)
            logging.info(f"Saved TorchScript model to {torchscript_path}")
            
            return scripted_model
            
        except Exception as e:
            logging.error(f"Error loading/optimizing model: {str(e)}")
            raise
    
    def create_config(self):
        """Create deployment configuration"""
        try:
            # Load existing config if available
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Add deployment metadata
            config['deployment'] = {
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0.0',
                'device': str(self.device),
                'input_shape': {
                    'expression': [1, 10000],
                    'sledai': [1, 1],
                    'adjacency': [10000, 10000]
                },
                'output_shape': [1, 2],
                'deployment_dir': str(self.deployment_dir)
            }
            
            # Save config
            config_path = self.deployment_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info(f"Created deployment config at {config_path}")
            return config
            
        except Exception as e:
            logging.error(f"Error creating config: {str(e)}")
            raise
    
    def create_api_spec(self):
        """Create API specification"""
        try:
            api_spec = {
                'openapi': '3.0.0',
                'info': {
                    'title': 'SLE Flare Prediction API',
                    'version': '1.0.0',
                    'description': 'API for predicting SLE flares using genomic and clinical data'
                },
                'paths': {
                    '/predict': {
                        'post': {
                            'summary': 'Predict SLE flare',
                            'requestBody': {
                                'required': True,
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'object',
                                            'properties': {
                                                'expression': {
                                                    'type': 'array',
                                                    'items': {'type': 'number'}
                                                },
                                                'sledai': {'type': 'number'},
                                                'adjacency_matrix': {
                                                    'type': 'array',
                                                    'items': {
                                                        'type': 'array',
                                                        'items': {'type': 'number'}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            'responses': {
                                '200': {
                                    'description': 'Successful prediction',
                                    'content': {
                                        'application/json': {
                                            'schema': {
                                                'type': 'object',
                                                'properties': {
                                                    'prediction': {'type': 'integer'},
                                                    'probability': {'type': 'number'},
                                                    'confidence': {'type': 'number'}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            # Save API spec
            api_spec_path = self.deployment_dir / 'api_spec.yaml'
            with open(api_spec_path, 'w') as f:
                import yaml
                yaml.dump(api_spec, f)
            
            logging.info(f"Created API specification at {api_spec_path}")
            return api_spec
            
        except Exception as e:
            logging.error(f"Error creating API spec: {str(e)}")
            raise
    
    def deploy(self):
        """Deploy the model"""
        try:
            # Load and optimize model
            model = self.load_model()
            
            # Create configuration
            config = self.create_config()
            
            # Create API specification
            api_spec = self.create_api_spec()
            
            # Create deployment package
            package = {
                'model': str(self.deployment_dir / 'model.pt'),
                'config': str(self.deployment_dir / 'config.json'),
                'api_spec': str(self.deployment_dir / 'api_spec.yaml'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save deployment package
            package_path = self.deployment_dir / 'deployment_package.json'
            with open(package_path, 'w') as f:
                json.dump(package, f, indent=4)
            
            logging.info("Deployment completed successfully!")
            logging.info(f"Deployment package created at {package_path}")
            
            return package
            
        except Exception as e:
            logging.error(f"Error during deployment: {str(e)}")
            raise

def main():
    """Main deployment function"""
    deployer = ModelDeployer(
        model_path='models/best_model.pth',
        config_path='config.json'
    )
    
    try:
        deployment_package = deployer.deploy()
        print("\nDeployment Package:")
        print(json.dumps(deployment_package, indent=2))
        
    except Exception as e:
        logging.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
