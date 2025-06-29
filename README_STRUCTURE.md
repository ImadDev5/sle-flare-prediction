# SLE Flare Prediction Repository Structure

## Directory Organization

```
├── src/
│   ├── models/           # TAGT model implementation
│   ├── data/            # Data processing utilities
│   ├── training/        # Training scripts
│   │   ├── train_optimized_real_data.py      # Optimized training script
│   │   ├── train_ultimate_real_data_model.py # Ultimate training script
│   │   └── ...
│   └── utils/           # Helper functions
├── experiments/         # Experimental analysis
│   ├── cross_validate_optimized.py    # Cross-validation experiments
│   └── ...
├── configs/            # Model configurations
├── docs/               # Documentation and paper
├── data/               # Dataset directory
└── results/            # Output figures and metrics
    ├── cross_validation.log           # Cross-validation logs
    ├── optimized_results.json         # Optimized model results
    ├── best_optimized_model.pth       # Best trained model
    ├── optimized_training.log         # Training logs
    ├── ultimate_training.log          # Ultimate training logs
    ├── cross_validation_results.json  # Cross-validation results
    └── ...
```

## Key Results Files

### Training Scripts
- `src/training/train_optimized_real_data.py` - Optimized training implementation
- `src/training/train_ultimate_real_data_model.py` - Ultimate training implementation

### Experimental Analysis
- `experiments/cross_validate_optimized.py` - Cross-validation analysis

### Results & Models
- `results/best_optimized_model.pth` - Best performing model weights
- `results/optimized_results.json` - Model performance metrics
- `results/cross_validation_results.json` - Cross-validation performance
- `results/optimized_training.log` - Training process logs
- `results/ultimate_training.log` - Ultimate training logs
- `results/cross_validation.log` - Cross-validation logs

## Usage

The repository is now organized following best practices for ML projects with clear separation of:
- Source code (`src/`)
- Experiments (`experiments/`)
- Configuration files (`configs/`)
- Documentation (`docs/`)
- Results and outputs (`results/`)
