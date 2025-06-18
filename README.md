# TAGT: Temporal Attention Graph Transformer for SLE Flare Prediction

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-yellow.svg)]()

A novel deep learning architecture for early prediction of Systemic Lupus Erythematosus (SLE) flares using multi-modal genomic and clinical data.

## 🎯 Overview

TAGT (Temporal Attention Graph Transformer) achieves **96.3% AUC-ROC** in predicting SLE flares by integrating:

- **Gene expression profiles** (1,000 features)
- **Protein-protein interaction networks** (graph structure)
- **Clinical parameters** (SLEDAI scores)
- **Temporal disease progression** (attention mechanisms)

## 🏆 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **TAGT (Ours)** | **0.833** | **0.667** | **0.667** | **0.667** | **0.963** |
| Random Forest | 0.750 | 0.000 | 0.000 | 0.000 | 0.648 |
| SVM | 0.500 | 0.200 | 0.333 | 0.250 | 0.519 |
| LSTM | 0.500 | 0.000 | 0.000 | 0.000 | 0.407 |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ImadDev5/sle-flare-prediction.git
cd sle-flare-prediction
pip install -r requirements.txt
```

### Training

```bash
python src/training/train.py --config configs/tagt_config.json
```

### Experiments

```bash
# Run baseline comparison
python experiments/baseline_comparison.py

# Run ablation study
python experiments/ablation_study.py

# Generate analysis
python experiments/analysis.py
```

## 📁 Repository Structure

```
├── src/
│   ├── models/           # TAGT model implementation
│   ├── data/            # Data processing utilities
│   ├── training/        # Training scripts
│   └── utils/           # Helper functions
├── experiments/         # Experimental analysis
├── configs/            # Model configurations
├── docs/               # Documentation and paper
├── data/               # Dataset directory
└── results/            # Output figures and metrics
```

## 🔬 Architecture

TAGT combines three key components:

1. **Graph Neural Networks** - Model protein-protein interactions
2. **Temporal Attention** - Capture disease progression patterns
3. **Multi-modal Fusion** - Integrate genomic and clinical data

## 🎧 Multimedia Resources

### **📊 Project Overview**
- **[Mind Map](docs/multimedia/project_mindmap.png)** - Visual overview of the TAGT architecture and key concepts
- **[Audio Guide](docs/multimedia/project_audio_guide.mp3)** - Complete project walkthrough (podcast-style explanation)

*Perfect for understanding the project at different levels - visual learners can explore the mind map, while the audio guide provides in-depth technical discussion.*

## 📚 Comprehensive Documentation

### **Detailed Q&A for Different Audiences**
- **[ML Experts](docs/FAQ_ML_EXPERTS.md)** - Deep technical dive into architecture, implementation, and performance
- **[Software Developers](docs/FAQ_SOFTWARE_DEVELOPERS.md)** - Code structure, deployment, and engineering best practices
- **[Medical Professionals](docs/FAQ_MEDICAL_PROFESSIONALS.md)** - Clinical validation, implementation, and patient care integration
- **[Tech-Savvy Non-Specialists](docs/FAQ_TECH_SAVVY_NON_SPECIALISTS.md)** - Accessible technical overview with practical context
- **[Patent Judges](docs/FAQ_PATENT_JUDGES.md)** - Intellectual property, novelty assessment, and prior art analysis
- **[Journal Reviewers](docs/FAQ_JOURNAL_REVIEWERS.md)** - Academic rigor, methodology, and research contribution

### **Research Roadmap**
- **[Future Work & Research Directions](docs/FUTURE_WORK_ROADMAP.md)** - Comprehensive roadmap for TAGT evolution and impact

## 📊 Clinical Impact

- **Early Intervention**: Predict flares before onset
- **Personalized Medicine**: Individual risk assessment
- **Improved Outcomes**: Prevent severe flares and organ damage
- **Healthcare Optimization**: Efficient resource allocation

## 📖 Citation

```bibtex
@misc{tagt2025,
  title = {TAGT: Temporal Attention Graph Transformer for Early Prediction of Systemic Lupus Erythematosus Flares Using Multi-Modal Genomic and Clinical Data},
  author = {ImadDev5},
  year = {2025},
  note = {Preprint available at \url{https://github.com/ImadDev5/sle-flare-prediction}},
  url = {https://github.com/ImadDev5/sle-flare-prediction}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations: imaduddin.dev@gmail.com

---

**Note**: This research demonstrates the potential for AI-driven precision medicine in autoimmune diseases. The model shows promising results for clinical translation with proper validation.
