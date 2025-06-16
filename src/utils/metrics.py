"""
Evaluation metrics for SLE flare prediction
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.5
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted way"""
    print(f"\n{model_name} Performance:")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    if 'auc' in metrics:
        print(f"AUC-ROC:   {metrics['auc']:.3f}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Flare', 'Flare'],
                yticklabels=['No Flare', 'Flare'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for a metric"""
    
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return lower, upper


class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_best_epoch(self, metric='val_f1'):
        if metric in self.metrics and self.metrics[metric]:
            return np.argmax(self.metrics[metric])
        return 0
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.metrics['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # Accuracy curves
        axes[0, 1].plot(self.metrics['train_acc'], label='Train')
        axes[0, 1].plot(self.metrics['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        
        # F1 curves
        axes[1, 0].plot(self.metrics['train_f1'], label='Train')
        axes[1, 0].plot(self.metrics['val_f1'], label='Validation')
        axes[1, 0].set_title('F1-Score')
        axes[1, 0].legend()
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
