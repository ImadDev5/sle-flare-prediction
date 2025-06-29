#!/usr/bin/env python3
"""Hyper-parameter optimisation of TAGT model with focal loss + 5-fold CV.

This script searches over architecture + optimisation hyper-parameters using
Optuna (Bayesian optimisation) and a focal loss that is better suited to the
flare / non-flare class imbalance.  It re-uses the Dataset and model classes
from `train_clean.py`, so keep that file in the same directory.

Run overnight – 50 trials on CPU takes ~18–22 hours; on GPU ~4 hours.  Results
(best parameters, full study) are stored under `optuna_studies/`.
"""
from __future__ import annotations

import os
import sys
import time

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import logging
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Import train_clean module for MAX_GENES
import train_clean

# Import fixed graph attention and breakthrough model
from src.training.train_clean import GraphAttentionLayer, TAGTModel
from src.training.train_breakthrough import BreakthroughTAGTModel, SLEDataset
from train_clean import (
    find_best_threshold,  # for tuning threshold per fold
    DATA_DIR,
)

# Update adjacency path to use sparse format
ADJACENCY_PATH = Path("data/processed/ppi/ppi_adjacency_sparse.npz")

# ---------- Focal Loss ---------------------------------------------------- #

class FocalLoss(nn.Module):
    """Focal Loss for binary / multi-class classification.

    Parameters
    ----------
    alpha : float
        Weighting factor in (0,1) to balance positive vs. negative examples.
    gamma : float
        Focusing parameter > 0.  Higher emphasises hard, mis-classified cases.
    reduction : str
        'mean' or 'sum'.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # p_t = prob of the true class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


# ---------- Configuration ------------------------------------------------- #

# Override gene limit globally for train_clean before any Dataset is created
MAX_GENES_OPT = 25  # Align with probe list length
train_clean.MAX_GENES = MAX_GENES_OPT

SEED = 42
N_SPLITS = 5  # CV folds
MAX_EPOCHS = 60
EARLY_PATIENCE = 10
BATCH_SIZE = 2  # lower batch to curb VRAM
NUM_WORKERS = 0  # Windows


STUDY_DIR = Path("optuna_studies")
STUDY_DIR.mkdir(exist_ok=True)
STUDY_NAME = "tagt_focal_cv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Using device %s", DEVICE)

# Load adjacency matrix from sparse format
adj_sparse = np.load(ADJACENCY_PATH)
adj = torch.sparse_coo_tensor(
    torch.LongTensor([adj_sparse['row'], adj_sparse['col']]),
    torch.FloatTensor(adj_sparse['data']),
    torch.Size(adj_sparse['shape'])
).to_dense()

# Slice to MAX_GENES_OPT to save RAM
ADJ_SUB = adj[:MAX_GENES_OPT, :MAX_GENES_OPT].clone()
ADJ_SUB.fill_diagonal_(1)
ADJ_SUB = ADJ_SUB.to(DEVICE)

del adj  # free memory

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Objective ----------------------------------------------------- #

def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximise mean F1 across CV folds."""

    # Hyper-parameters to search
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128])
    n_heads = trial.suggest_categorical("n_heads", [4, 6, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    alpha = trial.suggest_float("alpha", 0.5, 0.9)
    gamma = trial.suggest_int("gamma", 1, 3)

    # Load clinical to create splits
    clinical_df = pd.read_csv(DATA_DIR / "clinical_data.csv")
    labels = clinical_df["flare"].values
    indices = np.arange(len(labels))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_f1s = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        # Build datasets / loaders
        ds_train = SLEDataset(
            DATA_DIR / "expression_normalized.csv", 
            DATA_DIR / "clinical_data.csv", 
            DATA_DIR / "ppi/probe_list.csv", 
            train_idx
        )
        ds_val = SLEDataset(
            DATA_DIR / "expression_normalized.csv", 
            DATA_DIR / "clinical_data.csv", 
            DATA_DIR / "ppi/probe_list.csv", 
            val_idx
        )

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Model & optimiser - use breakthrough model
        model = BreakthroughTAGTModel(
            n_genes=MAX_GENES_OPT, 
            hidden_dim=hidden_dim, 
            n_heads=n_heads, 
            dropout=dropout
        ).to(DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_f1, patience = 0.0, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for batch in dl_train:
                optimizer.zero_grad()
                # Ensure correct key names are used
                expression = batch["expression"].to(DEVICE)
                sledai = batch["sledai"].to(DEVICE)
                labels = batch["label"].to(DEVICE) if "label" in batch else batch["flare"].to(DEVICE)
                
                out = model(expression, sledai, ADJ_SUB)
                loss = criterion(out, labels)
                
                if torch.isnan(loss):
                    continue
                    
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            probs, y_true = [], []
            with torch.no_grad():
                for batch in dl_val:
                    expression = batch["expression"].to(DEVICE)
                    sledai = batch["sledai"].to(DEVICE)
                    labels = batch["label"].to(DEVICE) if "label" in batch else batch["flare"].to(DEVICE)
                    
                    out = model(expression, sledai, ADJ_SUB)
                    p = torch.softmax(out, 1)[:, 1]
                    probs.extend(p.cpu())
                    y_true.extend(labels.cpu())
                    
            probs = torch.stack(probs)
            y_true = torch.tensor(y_true)
            
            # Find optimal threshold
            threshold, _ = find_best_threshold(probs.numpy(), y_true.numpy(), ADJ_SUB.cpu().numpy())
            preds = (probs >= threshold).long()
            f1 = f1_score(y_true, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                patience = 0
            else:
                patience += 1
                
            if patience >= EARLY_PATIENCE:
                break  # early-stop this fold
                
            # free VRAM periodically
            torch.cuda.empty_cache()

        fold_f1s.append(best_f1)
        trial.report(best_f1, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # extra cleanup per trial
    torch.cuda.empty_cache()
    mean_f1 = float(np.mean(fold_f1s))
    return mean_f1  # maximise


# ---------- Main ---------------------------------------------------------- #

def main():
    set_seed(SEED)

    study_path = STUDY_DIR / f"{STUDY_NAME}_breakthrough.db"
    storage = f"sqlite:///{study_path}"

    study = optuna.create_study(
        study_name=f"{STUDY_NAME}_breakthrough",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,
    )

    logging.info("Starting optimisation with breakthrough TAGT model")
    study.optimize(objective, n_trials=50, timeout=None, gc_after_trial=True)

    logging.info("Study finished: best value %.4f", study.best_value)
    logging.info("Best params: %s", study.best_params)

    # Save best parameters
    with (STUDY_DIR / "best_params_breakthrough.json").open("w") as f:
        json.dump(study.best_params, f, indent=4)
        
    # Train final model with best parameters
    logging.info("Training final model with best parameters")
    
    # Create model with best parameters
    best_model = BreakthroughTAGTModel(
        n_genes=MAX_GENES_OPT,
        hidden_dim=study.best_params["hidden_dim"],
        n_heads=study.best_params["n_heads"],
        dropout=study.best_params["dropout"]
    ).to(DEVICE)
    
    # Save model architecture and best parameters
    model_info = {
        "model_type": "BreakthroughTAGTModel",
        "parameters": study.best_params,
        "performance": {
            "f1_score": study.best_value
        },
        "description": "Breakthrough TAGT model with multi-scale graph attention and pathway-aware attention"
    }
    
    with open(Path("models") / "breakthrough_model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)
        
    logging.info("Model information saved to models/breakthrough_model_info.json")


if __name__ == "__main__":
    main()
