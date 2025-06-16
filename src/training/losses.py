"""
Loss Functions for TAGT v2.0 Model
- Dynamic Focal Loss for class imbalance
- TopoLoss v2 for preserving protein-protein interaction network topology
- Community Preservation Loss for maintaining protein complex structures
- Multi-task loss for joint optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import pairwise_distances


class FocalLoss(nn.Module):
    """
    Dynamic Focal Loss for imbalanced classification
    FL(p_t) = -α_t(1-p_t)^γ log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize focal loss
        
        Parameters:
        -----------
        alpha : float or torch.Tensor
            Class weight factor, can be a single value or per-class weights
        gamma : float
            Focusing parameter, reduces loss for well-classified examples
        reduction : str
            Reduction method: 'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions, shape (batch_size, 1) or (batch_size,)
        targets : torch.Tensor
            Ground truth labels, shape (batch_size, 1) or (batch_size,)
            
        Returns:
        --------
        torch.Tensor
            Focal loss
        """
        # Ensure inputs and targets have the same shape
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary case
        p = torch.sigmoid(inputs) if inputs.max() > 1 or inputs.min() < 0 else inputs
        p = torch.clamp(p, self.eps, 1.0 - self.eps)
        
        # Calculate focal weight
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = self.alpha[targets.long()]
            
        # Calculate loss
        loss = -alpha_t * focal_weight * torch.log(pt)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DynamicFocalLoss(FocalLoss):
    """
    Dynamic Focal Loss with adaptive alpha and gamma
    - Alpha is adjusted based on validation recall
    - Gamma can be scheduled over training
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', 
                 alpha_min=0.1, alpha_max=0.9, gamma_min=1.0, gamma_max=5.0):
        """
        Initialize dynamic focal loss
        
        Parameters:
        -----------
        alpha : float or torch.Tensor
            Initial class weight factor
        gamma : float
            Initial focusing parameter
        reduction : str
            Reduction method: 'none', 'mean', 'sum'
        alpha_min, alpha_max : float
            Min/max bounds for alpha
        gamma_min, gamma_max : float
            Min/max bounds for gamma
        """
        super(DynamicFocalLoss, self).__init__(alpha, gamma, reduction)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
    
    def update_parameters(self, metrics=None, epoch=None, total_epochs=None):
        """
        Update loss parameters based on validation metrics or training progress
        
        Parameters:
        -----------
        metrics : dict, optional
            Validation metrics (e.g., recall, precision)
        epoch : int, optional
            Current epoch
        total_epochs : int, optional
            Total number of epochs
            
        Returns:
        --------
        dict
            Updated parameters (alpha, gamma)
        """
        # Update alpha based on validation recall
        if metrics is not None and 'recall' in metrics:
            # If recall is low, increase alpha for positive class
            recall = metrics['recall']
            target_recall = 0.8  # Target recall value
            
            # Adjust alpha based on recall difference
            alpha_adj = min(max(0.5 - recall, -0.2), 0.2)  # Limit adjustment to [-0.2, 0.2]
            new_alpha = min(max(self.alpha + alpha_adj, self.alpha_min), self.alpha_max)
            
            self.alpha = new_alpha
        
        # Update gamma based on training progress
        if epoch is not None and total_epochs is not None:
            # Start with lower gamma and increase over time
            progress = epoch / total_epochs
            new_gamma = self.gamma_min + progress * (self.gamma_max - self.gamma_min)
            
            self.gamma = new_gamma
            
        return {'alpha': self.alpha, 'gamma': self.gamma}


class CommunityPreservationLoss(nn.Module):
    """
    Community Preservation Loss
    - Ensures embeddings of proteins in the same community are similar
    - Preserves protein complex structures in the embedding space
    """
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Initialize community preservation loss
        
        Parameters:
        -----------
        margin : float
            Margin for triplet loss
        reduction : str
            Reduction method: 'none', 'mean', 'sum'
        """
        super(CommunityPreservationLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embeddings, communities):
        """
        Forward pass
        
        Parameters:
        -----------
        embeddings : torch.Tensor
            Node embeddings (num_nodes, embed_dim)
        communities : torch.Tensor
            Community assignments (num_nodes,)
            
        Returns:
        --------
        torch.Tensor
            Community preservation loss
        """
        device = embeddings.device
        num_nodes = embeddings.size(0)
        
        # Get unique communities
        unique_communities = torch.unique(communities)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # Skip if no communities
        if len(unique_communities) <= 1:
            return loss
        
        # Calculate pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # For each community, calculate intra-community and inter-community distances
        valid_triplets = 0
        for comm in unique_communities:
            # Get nodes in this community
            comm_mask = (communities == comm)
            
            # Skip if community has too few members
            if comm_mask.sum() < 2:
                continue
            
            # Get other communities
            other_comms = unique_communities[unique_communities != comm]
            
            # For each node in this community
            for i in range(num_nodes):
                if not comm_mask[i]:
                    continue
                    
                # Find positive pairs (same community)
                pos_indices = torch.where(comm_mask)[0]
                pos_indices = pos_indices[pos_indices != i]  # Exclude self
                
                # Skip if no positive pairs
                if len(pos_indices) == 0:
                    continue
                
                # Sample negative pairs (different communities)
                neg_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                for other_comm in other_comms:
                    neg_mask = neg_mask | (communities == other_comm)
                neg_indices = torch.where(neg_mask)[0]
                
                # Skip if no negative pairs
                if len(neg_indices) == 0:
                    continue
                
                # Calculate triplet loss
                # For each positive pair, find the closest negative pair
                for pos_idx in pos_indices:
                    pos_dist = dist_matrix[i, pos_idx]
                    neg_dists = dist_matrix[i, neg_indices]
                    min_neg_dist, min_neg_idx = torch.min(neg_dists, dim=0)
                    
                    # Triplet loss
                    triplet_loss = F.relu(pos_dist - min_neg_dist + self.margin)
                    loss += triplet_loss
                    valid_triplets += 1
        
        # Apply reduction
        if valid_triplets > 0:
            if self.reduction == 'mean':
                return loss / valid_triplets
            elif self.reduction == 'sum':
                return loss
        
        return loss


class TopoLossV2(nn.Module):
    """
    TopoLoss v2
    - Preserves protein-protein interaction network topology in the embedding space
    - Combines adjacency matrix alignment with community preservation
    - Adds hierarchical constraints for functional relationships
    """
    def __init__(self, adjacency_weight=0.5, community_weight=0.5, reduction='mean'):
        """
        Initialize TopoLoss v2
        
        Parameters:
        -----------
        adjacency_weight : float
            Weight for adjacency matrix alignment loss
        community_weight : float
            Weight for community preservation loss
        reduction : str
            Reduction method: 'none', 'mean', 'sum'
        """
        super(TopoLossV2, self).__init__()
        self.adjacency_weight = adjacency_weight
        self.community_weight = community_weight
        self.reduction = reduction
        
        # Community preservation loss
        self.community_loss = CommunityPreservationLoss(reduction=reduction)
    
    def forward(self, embeddings, adjacency_matrix, communities=None):
        """
        Forward pass
        
        Parameters:
        -----------
        embeddings : torch.Tensor
            Node embeddings (num_nodes, embed_dim)
        adjacency_matrix : torch.Tensor
            Adjacency matrix (num_nodes, num_nodes)
        communities : torch.Tensor, optional
            Community assignments (num_nodes,)
            
        Returns:
        --------
        torch.Tensor
            TopoLoss v2
        """
        device = embeddings.device
        
        # Calculate cosine similarity matrix
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # Adjacency matrix alignment loss
        # Entries should be similar for connected nodes, dissimilar for unconnected
        adj_mask = adjacency_matrix > 0
        pos_loss = F.mse_loss(
            sim_matrix[adj_mask],
            torch.ones_like(sim_matrix[adj_mask]),
            reduction=self.reduction
        )
        
        neg_loss = F.mse_loss(
            sim_matrix[~adj_mask],
            torch.zeros_like(sim_matrix[~adj_mask]),
            reduction=self.reduction
        )
        
        adjacency_loss = pos_loss + neg_loss
        
        # Community preservation loss
        if communities is not None:
            comm_loss = self.community_loss(embeddings, communities)
        else:
            comm_loss = torch.tensor(0.0, device=device)
        
        # Combine losses
        total_loss = self.adjacency_weight * adjacency_loss + self.community_weight * comm_loss
        
        return total_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss for joint optimization
    - Combines multiple loss functions with adaptive weighting
    - Supports uncertainty-based weighting
    """
    def __init__(self, loss_fns, weights=None, uncertainty_weighting=False):
        """
        Initialize multi-task loss
        
        Parameters:
        -----------
        loss_fns : dict
            Dictionary of loss functions {task_name: loss_fn}
        weights : dict, optional
            Initial weights for each loss {task_name: weight}
        uncertainty_weighting : bool
            Whether to use uncertainty-based weighting
        """
        super(MultiTaskLoss, self).__init__()
        self.loss_fns = loss_fns
        self.task_names = list(loss_fns.keys())
        
        # Initialize weights
        if weights is None:
            weights = {task: 1.0 for task in self.task_names}
        self.weights = weights
        
        # Uncertainty weighting
        self.uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting:
            # Learnable log variances for each task
            self.log_vars = nn.Parameter(torch.zeros(len(self.task_names)))
    
    def forward(self, outputs, targets):
        """
        Forward pass
        
        Parameters:
        -----------
        outputs : dict
            Model outputs for each task {task_name: output}
        targets : dict
            Ground truth for each task {task_name: target}
            
        Returns:
        --------
        tuple
            - Total loss (torch.Tensor)
            - Dictionary of individual losses {task_name: loss}
        """
        device = next(iter(outputs.values())).device
        
        # Initialize total loss and individual losses
        total_loss = torch.tensor(0.0, device=device)
        individual_losses = {}
        
        # Calculate loss for each task
        for i, task in enumerate(self.task_names):
            if task in outputs and task in targets:
                # Calculate task loss
                task_loss = self.loss_fns[task](outputs[task], targets[task])
                individual_losses[task] = task_loss.detach()
                
                # Apply weighting
                if self.uncertainty_weighting:
                    # Uncertainty weighting: L = L/σ² + log(σ²)
                    precision = torch.exp(-self.log_vars[i])
                    total_loss += precision * task_loss + self.log_vars[i]
                else:
                    # Fixed weighting
                    total_loss += self.weights[task] * task_loss
        
        return total_loss, individual_losses
    
    def update_weights(self, validation_losses):
        """
        Update loss weights based on validation performance
        
        Parameters:
        -----------
        validation_losses : dict
            Validation losses for each task {task_name: loss}
            
        Returns:
        --------
        dict
            Updated weights
        """
        if self.uncertainty_weighting:
            # Weights are learned automatically
            return {task: torch.exp(-self.log_vars[i]).item() 
                   for i, task in enumerate(self.task_names)}
        
        # Calculate inverse task rates
        # Tasks with higher losses get higher weights
        total_loss = sum(validation_losses.values())
        if total_loss > 0:
            loss_ratios = {task: loss / total_loss for task, loss in validation_losses.items()}
            
            # Adjust weights: tasks with high losses get more weight
            new_weights = {}
            for task in self.task_names:
                if task in loss_ratios:
                    # Smooth adjustment to avoid large fluctuations
                    adj_factor = 1.0 + (loss_ratios[task] - 1.0/len(validation_losses)) * 0.5
                    new_weights[task] = self.weights[task] * adj_factor
                else:
                    new_weights[task] = self.weights[task]
            
            # Normalize weights to sum to number of tasks
            weight_sum = sum(new_weights.values())
            if weight_sum > 0:
                norm_factor = len(self.task_names) / weight_sum
                self.weights = {task: weight * norm_factor for task, weight in new_weights.items()}
        
        return self.weights


def get_loss_function(loss_type, **kwargs):
    """
    Factory function to create loss functions
    
    Parameters:
    -----------
    loss_type : str
        Type of loss function: 'focal', 'dynamic_focal', 'topo', 'community', 'multi_task'
    **kwargs : dict
        Additional parameters for the loss function
        
    Returns:
    --------
    nn.Module
        Loss function
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'dynamic_focal':
        return DynamicFocalLoss(**kwargs)
    elif loss_type == 'topo':
        return TopoLossV2(**kwargs)
    elif loss_type == 'community':
        return CommunityPreservationLoss(**kwargs)
    elif loss_type == 'multi_task':
        return MultiTaskLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
