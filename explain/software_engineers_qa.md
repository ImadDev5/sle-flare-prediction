# Software Engineers Q&A

## What is the overall architecture and technology stack?

**Framework & Languages:**
- **Primary**: Python 3.8+ with PyTorch 1.9+
- **Deep Learning**: PyTorch Geometric for graph neural networks
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Bioinformatics**: MyGene for gene mapping

**Project Structure:**
```
├── src/
│   ├── models/           # TAGT model implementations
│   ├── data/            # Data processing utilities
│   ├── training/        # Training scripts and losses
│   └── utils/           # Helper functions
├── configs/             # Model configurations (JSON)
├── data/               # Dataset storage
├── validation_plan/    # Validation and testing
└── explain/            # Documentation Q&A files
```

## What are the key model architectures implemented?

**Primary Models** (in `src/models/`):
1. **breakthrough_tagt.py**: Advanced multi-scale graph attention
2. **ultimate_tagt.py**: Production-ready model with real data
3. **optimized_tagt.py**: Memory-optimized version
4. **tagt_model.py**: Base implementation

**Key Components:**
- **FixedGraphAttentionLayer**: Handles single-feature nodes properly
- **MultiScaleGraphAttention**: Local and global attention mechanisms  
- **PathwayAttention**: Biological interpretability through pathway modeling
- **HierarchicalTemporalEncoder**: Multi-scale LSTM for temporal patterns

## What are the current development practices and CI/CD setup?

**Code Quality Automation** (`.github/workflows/quality_assurance.yml`):
- **Formatters**: Black (line length 88), isort
- **Linters**: Flake8, MyPy for type checking
- **Security**: Bandit, Safety for vulnerability scanning
- **Testing**: Pytest with coverage reporting
- **Pre-commit hooks**: Automatic quality checks

**Quality Configuration** (`quality_config.json`):
- Parallel processing with 4 workers
- GPU optimization checks
- Performance tracking
- Medical AI specific validations

## What are the deployment and GPU optimization strategies?

**Hardware Optimization:**
- **Target GPU**: RTX 3050 (8GB VRAM)
- **Batch Size**: 4-8 (with gradient accumulation)
- **Mixed Precision**: Enabled via autocast/GradScaler
- **Memory Management**: Gradient accumulation steps = 4

**Training Configuration** (`configs/ultimate_tagt_config.json`):
```json
{
  "batch_size": 8,
  "learning_rate": 1e-4,
  "use_amp": true,
  "accumulation_steps": 4,
  "num_workers": 0  # Windows compatibility
}
```

## What are the current performance and scalability considerations?

**Training Performance:**
- **Model Parameters**: ~198,658 parameters (production model)
- **Training Time**: 5 epochs average on RTX 3050
- **Dataset Size**: 378 samples, 1000 genes, 1000 edges
- **Memory Usage**: Optimized for 8GB GPU memory

**Scalability Limitations:**
1. **Single GPU**: Currently designed for single RTX 3050
2. **Memory Constraints**: Batch size limited by GPU memory
3. **Data Loading**: num_workers=0 for Windows compatibility
4. **Model Size**: Fixed at 1000 genes for current PPI network

## What are the key technical challenges and solutions?

**Graph Attention Challenges:**
- **Problem**: Handling single-feature nodes in PPI networks
- **Solution**: `FixedGraphAttentionLayer` with proper dimension handling

**Class Imbalance:**
- **Problem**: 33.9% flare rate in dataset
- **Solution**: Focal Loss with adaptive gamma and alpha parameters

**Memory Optimization:**
- **Problem**: Large graphs don't fit in GPU memory
- **Solution**: Gradient accumulation, mixed precision, batch processing

**Reproducibility:**
- **Problem**: Non-deterministic GPU operations
- **Solution**: Seed setting, deterministic algorithms in configs

## What testing and validation frameworks are in place?

**Validation Pipeline** (`validation_plan/`):
- **Data Audit**: Quality checks and statistics
- **Baseline Models**: RF, SVM, Logistic Regression, LSTM comparisons
- **Cross-validation**: 5-fold stratified validation
- **Metrics Tracking**: Comprehensive performance evaluation

**Current Test Results:**
- **TAGT AUC**: 87.3% (vs baseline ~49-50%)
- **Validation Issues**: Some perfect scores suggesting overfitting
- **Baseline Performance**: Traditional ML models perform poorly

## What are the deployment and production considerations?

**Current Production Setup:**
- **Model Format**: PyTorch state dict
- **Input Requirements**: Gene expression + clinical data
- **Output Format**: Probability scores for flare prediction
- **Device Compatibility**: CUDA-enabled GPU preferred

**Deployment Challenges:**
1. **Model Size**: Requires GPU for reasonable inference time
2. **Data Dependencies**: Needs preprocessed gene expression data
3. **Clinical Integration**: Requires SLEDAI scores and demographics
4. **Real-time Constraints**: Current model not optimized for latency
