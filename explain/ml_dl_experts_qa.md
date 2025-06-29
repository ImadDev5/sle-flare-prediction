# ML/DL Experts Q&A

## What are the core deep learning architectures used in this project?

**Primary Architecture: Breakthrough TAGT Model** (`src/models/breakthrough_tagt.py`)

**Core Components:**
1. **MultiScaleGraphAttention**: Multi-head attention with edge features and multi-scale convolutions (1x1, 3x3, 5x5, 7x7)
2. **PathwayAttention**: Learnable pathway embeddings (50 pathways) with cross-attention for biological interpretability
3. **HierarchicalTemporalEncoder**: Multi-scale LSTM encoders (short, medium, long-term) with temporal fusion
4. **CrossModalFusion**: Multi-head attention fusion between genomic and clinical modalities

**Architecture Details:**
- **Parameters**: ~198,658 parameters (production model)
- **Hidden Dimension**: 256 (breakthrough) / 96 (production)
- **Attention Heads**: 8 (breakthrough) / 4 (production)
- **Graph Layers**: 4 layers with residual connections
- **Input**: Single gene expression values per node + clinical features

## What loss functions and training techniques are employed?

**Advanced Loss Functions** (`src/training/losses.py`):

1. **FocalLoss**: Handles class imbalance with adaptive alpha (0.25-0.9) and gamma (1.0-5.0)
2. **DynamicFocalLoss**: Adapts parameters based on validation metrics
3. **TopoLossV2**: Preserves PPI network topology in embedding space
4. **CommunityPreservationLoss**: Maintains protein complex structures
5. **MultiTaskLoss**: Joint optimization with uncertainty-based weighting

**Training Innovations:**
- **Mixed Precision**: Autocast + GradScaler for RTX 3050 optimization
- **Gradient Accumulation**: 4 steps (effective batch size 16-32)
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Advanced Regularization**: DropEdge, spectral normalization, label smoothing

## How does the model handle graph neural network challenges?

**Graph Attention Innovations:**

1. **FixedGraphAttentionLayer** (`src/training/fixed_graph_attention.py`):
   - Handles single-feature nodes properly (gene expression values)
   - Dimension-aware skip connections
   - Multi-head attention with learnable temperature

2. **Multi-Scale Processing**:
   - Local attention (real adjacency matrix)
   - Global attention (full connectivity)
   - Hierarchical pooling for different interaction scales

3. **Edge Feature Integration**:
   - STRING confidence scores as edge attributes
   - Learnable edge encoders for attention weighting

## What are the key technical innovations for temporal modeling?

**Hierarchical Temporal Architecture:**

1. **Multi-Resolution Encoding**:
   - Short-term: Direct LSTM on full sequence
   - Medium-term: 2x subsampled with interpolation
   - Long-term: 4x subsampled with interpolation

2. **Temporal Fusion**:
   - Multi-head attention across temporal scales
   - Learnable combination of short/medium/long-term patterns

3. **Sequence Processing**:
   - Bidirectional LSTM layers
   - Attention over temporal dimension
   - Variable sequence length support

## How is class imbalance addressed at the model level?

**Class Imbalance Solutions:**

1. **Focal Loss Implementation**:
   ```python
   focal_loss = alpha * (1 - pt)^gamma * cross_entropy
   # alpha=0.75, gamma=2.0 for current dataset (33.9% flare rate)
   ```

2. **Dynamic Parameter Adjustment**:
   - Alpha adaptation based on validation recall
   - Gamma scheduling over training epochs
   - Uncertainty-based weighting in multi-task scenarios

3. **Sampling Strategies**:
   - Gradient accumulation for effective larger batches
   - Stratified cross-validation (5-fold)

## What optimization and hyperparameter tuning approaches are used?

**Optuna-Based Optimization** (`src/training/train_optuna.py`):
- **Multi-objective**: AUC-ROC + F1-Score optimization
- **Pruning**: Early stopping for unpromising trials
- **Search Space**: Hidden dims, attention heads, dropout rates, learning rates
- **Cross-validation Integration**: 5-fold validation within each trial

**Hardware Optimization:**
- **RTX 3050 Specific**: Batch size 4-8, gradient accumulation
- **Memory Management**: Mixed precision, efficient DataLoader
- **Windows Compatibility**: num_workers=0 for stability

## What are the current performance bottlenecks and solutions?

**Identified Bottlenecks:**

1. **Memory Constraints**:
   - **Problem**: Large graphs exceed 8GB VRAM
   - **Solution**: Gradient accumulation, mixed precision, subgraph sampling

2. **Training Stability**:
   - **Problem**: Some CV folds show perfect performance (overfitting)
   - **Solution**: Enhanced regularization, validation monitoring

3. **Scalability**:
   - **Problem**: Fixed 1000-gene limit
   - **Solution**: Hierarchical attention, graph coarsening strategies

**Performance Metrics:**
- **Training Time**: ~5 epochs on RTX 3050
- **Inference Speed**: Single patient prediction in milliseconds
- **Memory Usage**: ~6GB GPU memory during training

## How does the model ensure biological interpretability?

**Interpretability Mechanisms:**

1. **Pathway-Aware Attention**:
   - 50 learnable pathway embeddings
   - Cross-attention between genes and pathways
   - Attention weights provide pathway importance scores

2. **Graph Attention Visualization**:
   - Edge attention weights show protein interaction importance
   - Node attention highlights critical genes
   - Multi-scale attention captures different biological levels

3. **Clinical-Genomic Fusion**:
   - Cross-modal attention weights show feature interactions
   - Temporal attention reveals disease progression patterns
   - SLEDAI score integration with genomic features

## What validation strategies ensure model robustness?

**Comprehensive Validation** (`validation_plan/`):

1. **Cross-Validation Results**:
   - **TAGT AUC**: 87.3% ± 25.4% (high variance indicates overfitting)
   - **Baseline Comparison**: RF (49.5%), SVM (47.5%), LSTM (49.5%)

2. **Ablation Studies**:
   - Component-wise performance analysis
   - Architecture sensitivity testing
   - Feature importance evaluation

3. **Robustness Testing**:
   - Missing data handling
   - Batch size sensitivity
   - Initialization stability

**Current Limitations:**
- High cross-validation variance suggests overfitting
- Perfect performance (1.0) on some folds indicates data leakage or overfitting
- Limited external validation on independent datasets
