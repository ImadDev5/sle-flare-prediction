# Real Data Training Instructions for SLE Flare Prediction Model

## 🎯 Objective
Train a production-level SLE flare prediction model using **real genomic data** from 100+ patients with state-of-the-art performance.

## 📊 Data Sources

### 1. GSE49454 - Real SLE Patient Data
- **Source**: NCBI Gene Expression Omnibus (GEO)
- **Description**: Gene expression profiles from SLE patients
- **URL**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49454
- **Patients**: 100+ SLE patients with clinical data
- **Platform**: Affymetrix Human Genome U133 Plus 2.0 Array

### 2. STRING Protein-Protein Interaction Network
- **Source**: STRING Database
- **Description**: High-confidence protein interactions
- **URL**: https://string-db.org/
- **Species**: Homo sapiens (Human)

## 🚀 Quick Start (Automated)

### Option 1: Run the Complete Training Script
```bash
# Double-click this file or run in command prompt:
run_production_training.bat
```

### Option 2: Manual Python Execution
```bash
# Navigate to project directory
cd "c:\Users\ADMIN\OneDrive\Desktop\SLE"

# Activate virtual environment
venv_gpu\Scripts\activate

# Run the production training
python train_real_data_model.py
```

## 📥 Manual Data Download (If Needed)

If the automated download fails, follow these steps:

### Step 1: Download GSE49454 Data
1. Go to: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE49nnn/GSE49454/matrix/
2. Download: `GSE49454_series_matrix.txt.gz`
3. Create folder: `data/raw/GSE49454/`
4. Place file in: `data/raw/GSE49454/GSE49454_series_matrix.txt.gz`

### Step 2: Download STRING Data (Optional)
1. Go to: https://stringdb-static.org/download/protein.info.v12.0/
2. Download: `9606.protein.info.v12.0.txt.gz`
3. Go to: https://stringdb-static.org/download/protein.links.v12.0/
4. Download: `9606.protein.links.v12.0.txt.gz`
5. Create folder: `data/raw/STRING/`
6. Place files in the STRING folder

### Step 3: Run Training
```bash
python train_real_data_model.py
```

## 🏗️ Model Architecture

### Production TAGT (Temporal Attention Graph Transformer)
- **Graph Attention**: Multi-head attention on gene interaction networks
- **Temporal Modeling**: LSTM/Transformer for disease progression
- **Clinical Integration**: Fusion of genomic and clinical features
- **Advanced Loss**: Focal Loss for handling class imbalance
- **Regularization**: Dropout, weight decay, gradient clipping

### Key Features
- **256-dimensional hidden representations**
- **8 attention heads** for multi-scale feature learning
- **4 graph attention layers** for deep gene interaction modeling
- **Temporal sequences** of length 6 for progression modeling
- **12 clinical features** including SLEDAI, complement levels, etc.

## 📈 Expected Performance

### Production Targets
- **AUC-ROC**: > 0.85 (Excellent clinical utility)
- **F1 Score**: > 0.75 (Balanced precision/recall)
- **Accuracy**: > 0.80 (High overall correctness)
- **Precision**: > 0.75 (Low false positive rate)
- **Recall**: > 0.75 (High sensitivity for flare detection)

### Training Details
- **Patients**: 100+ real SLE patients
- **Genes**: 2000+ gene expression features
- **Training Time**: ~30-60 minutes on GPU
- **Early Stopping**: Prevents overfitting
- **Cross-Validation**: Robust performance estimation

## 📁 Output Files

After training, you'll get:

1. **`best_production_model.pth`** - Trained model weights
2. **`production_model_results.json`** - Detailed performance metrics
3. **`data_download.log`** - Download and processing logs

### Sample Results JSON
```json
{
  "model_type": "Production TAGT with Real Data",
  "dataset_info": {
    "n_samples": 150,
    "n_genes": 2000,
    "data_source": "GSE49454 Real Data"
  },
  "final_metrics": {
    "test_auc": 0.8756,
    "test_f1": 0.7834,
    "test_accuracy": 0.8234,
    "test_precision": 0.7923,
    "test_recall": 0.7756
  },
  "production_ready": true
}
```

## 🔧 Troubleshooting

### Issue: Download Fails
**Solution**: Use manual download steps above

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size in the script (line ~400):
```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduce from 16 to 8
```

### Issue: Low Performance
**Solutions**:
1. Increase training epochs (line ~450)
2. Adjust learning rate (line ~430)
3. Add more regularization

### Issue: Missing Dependencies
**Solution**: Install requirements:
```bash
pip install torch torchvision pandas numpy scikit-learn requests
```

## 🎯 Production Deployment

Once trained, the model can be deployed for:

1. **Clinical Decision Support**: Real-time flare risk assessment
2. **Patient Monitoring**: Continuous risk tracking
3. **Treatment Optimization**: Personalized therapy selection
4. **Research**: Biomarker discovery and validation

## 📚 References

1. **GSE49454**: Chaussabel et al. "A modular analysis framework for blood genomics studies"
2. **STRING Database**: Szklarczyk et al. "STRING v11: protein-protein association networks"
3. **TAGT Architecture**: Custom implementation based on Graph Attention Networks
4. **SLE Clinical Features**: Based on SLEDAI and clinical guidelines

## 🆘 Support

If you encounter any issues:

1. Check the log files for detailed error messages
2. Ensure all dependencies are installed
3. Verify data files are downloaded correctly
4. Try reducing batch size if memory issues occur

---

**Ready to train your production-level SLE flare prediction model with real data!** 🚀