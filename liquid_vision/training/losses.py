"""
Specialized loss functions for liquid neural networks and temporal processing.
Includes temporal consistency losses and liquid state regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import math


class TemporalLoss(nn.Module):
    """
    Temporal consistency loss for sequence prediction.
    Encourages smooth temporal transitions in liquid states.
    """
    
    def __init__(
        self,
        base_loss: str = "cross_entropy",
        temporal_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        
        self.temporal_weight = temporal_weight
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight
        
        # Base loss function
        if base_loss == "cross_entropy":
            self.base_loss = nn.CrossEntropyLoss()
        elif base_loss == "mse":
            self.base_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
            
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        temporal_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporal loss.
        
        Args:
            outputs: Model predictions [batch_size, seq_len, num_classes] or [batch_size, num_classes]
            targets: Ground truth labels
            hidden_states: Liquid hidden states for regularization
            temporal_targets: Temporal consistency targets
            
        Returns:
            Combined loss value
        """
        # Base prediction loss
        if outputs.dim() == 3:  # Sequence output
            # Reshape for loss computation
            batch_size, seq_len, num_classes = outputs.shape
            outputs_flat = outputs.view(-1, num_classes)
            
            if targets.dim() == 2:  # Sequence targets
                targets_flat = targets.view(-1)
            else:  # Single target per sequence
                targets_flat = targets.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1)
                
            base_loss = self.base_loss(outputs_flat, targets_flat)
        else:
            base_loss = self.base_loss(outputs, targets)
            
        total_loss = base_loss
        
        # Temporal smoothness loss
        if outputs.dim() == 3 and self.smoothness_weight > 0:
            # Encourage smooth changes in predictions over time
            diff = outputs[:, 1:] - outputs[:, :-1]
            smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=-1))
            total_loss = total_loss + self.smoothness_weight * smoothness_loss
            
        # Temporal consistency loss
        if temporal_targets is not None and self.consistency_weight > 0:
            consistency_loss = F.mse_loss(outputs, temporal_targets)
            total_loss = total_loss + self.consistency_weight * consistency_loss
            
        # Hidden state regularization
        if hidden_states is not None and self.temporal_weight > 0:
            reg_loss = 0
            for state in hidden_states:
                if state is not None:
                    # L2 regularization on hidden states
                    reg_loss = reg_loss + torch.mean(state ** 2)
                    
            total_loss = total_loss + self.temporal_weight * reg_loss
            
        return total_loss


class LiquidLoss(nn.Module):
    """
    Specialized loss function for liquid neural networks.
    Includes liquid state dynamics regularization and stability constraints.
    """
    
    def __init__(
        self,
        base_loss: str = "cross_entropy",
        stability_weight: float = 0.01,
        sparsity_weight: float = 0.001,
        energy_weight: float = 0.005,
        diversity_weight: float = 0.01,
    ):
        super().__init__()
        
        self.stability_weight = stability_weight
        self.sparsity_weight = sparsity_weight
        self.energy_weight = energy_weight
        self.diversity_weight = diversity_weight
        
        # Base loss
        if base_loss == "cross_entropy":
            self.base_loss = nn.CrossEntropyLoss()
        elif base_loss == "mse":
            self.base_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
            
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        liquid_states: Optional[List[torch.Tensor]] = None,
        recurrent_weights: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute liquid loss with regularization terms.
        
        Args:
            outputs: Model predictions
            targets: Ground truth labels
            liquid_states: Current liquid hidden states
            recurrent_weights: Recurrent connection weights
            
        Returns:
            Combined loss value
        """
        # Base prediction loss
        base_loss = self.base_loss(outputs, targets)
        total_loss = base_loss
        
        if liquid_states is not None:
            # Stability regularization - encourage bounded states
            if self.stability_weight > 0:
                stability_loss = 0
                for state in liquid_states:
                    if state is not None:
                        # Penalize states with large magnitude
                        stability_loss = stability_loss + torch.mean(torch.clamp(torch.abs(state) - 1.0, min=0) ** 2)
                        
                total_loss = total_loss + self.stability_weight * stability_loss
                
            # Sparsity regularization - encourage sparse activations
            if self.sparsity_weight > 0:
                sparsity_loss = 0
                for state in liquid_states:
                    if state is not None:
                        # L1 penalty on activations
                        sparsity_loss = sparsity_loss + torch.mean(torch.abs(state))
                        
                total_loss = total_loss + self.sparsity_weight * sparsity_loss
                
            # Energy regularization - minimize total energy
            if self.energy_weight > 0:
                energy_loss = 0
                for state in liquid_states:
                    if state is not None:
                        # Total energy of liquid
                        energy_loss = energy_loss + torch.sum(state ** 2)
                        
                total_loss = total_loss + self.energy_weight * energy_loss
                
            # Diversity regularization - encourage diverse representations
            if self.diversity_weight > 0:
                diversity_loss = 0
                for state in liquid_states:
                    if state is not None and state.size(0) > 1:  # Need multiple samples
                        # Correlation matrix between samples
                        state_centered = state - torch.mean(state, dim=0, keepdim=True)
                        correlation = torch.mm(state_centered, state_centered.t())
                        
                        # Penalize high correlation between different samples
                        mask = torch.eye(state.size(0), device=state.device) == 0
                        diversity_loss = diversity_loss + torch.mean(correlation[mask] ** 2)
                        
                total_loss = total_loss + self.diversity_weight * diversity_loss
                
        # Recurrent weight regularization
        if recurrent_weights is not None and self.stability_weight > 0:
            weight_reg = 0
            for weights in recurrent_weights:
                if weights is not None:
                    # Spectral radius constraint (approximate)
                    eigenvals = torch.linalg.eigvals(weights)
                    max_eigenval = torch.max(torch.abs(eigenvals))
                    weight_reg = weight_reg + F.relu(max_eigenval - 0.95) ** 2
                    
            total_loss = total_loss + self.stability_weight * weight_reg
            
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative representations.
    Useful for self-supervised learning with event data.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0,
        distance_metric: str = "cosine",  # "cosine", "euclidean"
    ):
        super().__init__()
        
        self.temperature = temperature
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Class labels for automatic pair generation
            positive_pairs: Indices of positive pairs [num_pos_pairs, 2]
            negative_pairs: Indices of negative pairs [num_neg_pairs, 2]
            
        Returns:
            Contrastive loss value
        """
        if positive_pairs is None and negative_pairs is None:
            if labels is None:
                raise ValueError("Must provide either labels or explicit pairs")
            return self._compute_supervised_contrastive_loss(embeddings, labels)
        else:
            return self._compute_pairwise_contrastive_loss(
                embeddings, positive_pairs, negative_pairs
            )
            
    def _compute_supervised_contrastive_loss(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised contrastive loss using labels."""
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        if self.distance_metric == "cosine":
            embeddings = F.normalize(embeddings, dim=1)
            
        # Compute similarity matrix
        if self.distance_metric == "cosine":
            similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        else:  # euclidean
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
            similarity_matrix = -dist_matrix / self.temperature
            
        # Create positive and negative masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix)
        
        # Mask out diagonal
        exp_sim = exp_sim * (1 - torch.eye(batch_size, device=exp_sim.device))
        
        # Positive pairs
        pos_sim = exp_sim * mask
        
        # All pairs (for normalization)
        all_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # Contrastive loss
        pos_pairs_per_sample = torch.sum(mask, dim=1)
        valid_samples = pos_pairs_per_sample > 0
        
        if not valid_samples.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        log_prob = torch.log(pos_sim / (all_sim + 1e-8))
        loss = -torch.sum(log_prob * mask, dim=1) / (pos_pairs_per_sample + 1e-8)
        
        return torch.mean(loss[valid_samples])
        
    def _compute_pairwise_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        positive_pairs: Optional[torch.Tensor],
        negative_pairs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute contrastive loss using explicit pairs."""
        total_loss = 0.0
        num_pairs = 0
        
        if positive_pairs is not None:
            pos_emb1 = embeddings[positive_pairs[:, 0]]
            pos_emb2 = embeddings[positive_pairs[:, 1]]
            
            if self.distance_metric == "cosine":
                pos_dist = 1 - F.cosine_similarity(pos_emb1, pos_emb2)
            else:
                pos_dist = F.pairwise_distance(pos_emb1, pos_emb2, p=2)
                
            pos_loss = torch.mean(pos_dist ** 2)
            total_loss = total_loss + pos_loss
            num_pairs += 1
            
        if negative_pairs is not None:
            neg_emb1 = embeddings[negative_pairs[:, 0]]
            neg_emb2 = embeddings[negative_pairs[:, 1]]
            
            if self.distance_metric == "cosine":
                neg_dist = 1 - F.cosine_similarity(neg_emb1, neg_emb2)
            else:
                neg_dist = F.pairwise_distance(neg_emb1, neg_emb2, p=2)
                
            neg_loss = torch.mean(F.relu(self.margin - neg_dist) ** 2)
            total_loss = total_loss + neg_loss
            num_pairs += 1
            
        return total_loss / max(num_pairs, 1)


class EventSequenceLoss(nn.Module):
    """
    Loss function for event sequence prediction tasks.
    Handles variable-length sequences and temporal alignment.
    """
    
    def __init__(
        self,
        base_loss: str = "cross_entropy",
        sequence_weight: float = 1.0,
        alignment_weight: float = 0.1,
        length_penalty: float = 0.01,
    ):
        super().__init__()
        
        self.sequence_weight = sequence_weight
        self.alignment_weight = alignment_weight
        self.length_penalty = length_penalty
        
        if base_loss == "cross_entropy":
            self.base_loss = nn.CrossEntropyLoss(ignore_index=-1)
        elif base_loss == "mse":
            self.base_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
            
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prediction_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sequence loss with alignment.
        
        Args:
            predictions: Predicted sequences [batch_size, max_seq_len, num_classes]
            targets: Target sequences [batch_size, max_seq_len]
            prediction_lengths: Actual lengths of predictions
            target_lengths: Actual lengths of targets
            
        Returns:
            Sequence loss value
        """
        batch_size, max_seq_len, num_classes = predictions.shape
        
        # Base sequence loss
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        base_loss = self.base_loss(predictions_flat, targets_flat)
        total_loss = self.sequence_weight * base_loss
        
        # Length penalty
        if prediction_lengths is not None and target_lengths is not None and self.length_penalty > 0:
            length_diff = torch.abs(prediction_lengths.float() - target_lengths.float())
            length_loss = torch.mean(length_diff)
            total_loss = total_loss + self.length_penalty * length_loss
            
        # Temporal alignment loss (simplified)
        if self.alignment_weight > 0:
            # Encourage consistent predictions across time
            if predictions.size(1) > 1:
                temporal_diff = predictions[:, 1:] - predictions[:, :-1]
                alignment_loss = torch.mean(torch.sum(temporal_diff ** 2, dim=-1))
                total_loss = total_loss + self.alignment_weight * alignment_loss
                
        return total_loss


def create_loss_function(
    loss_type: str,
    num_classes: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function
        num_classes: Number of classes (for classification)
        **kwargs: Additional loss-specific arguments
        
    Returns:
        Configured loss function
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "mse":
        return nn.MSELoss(**kwargs)
    elif loss_type == "temporal":
        return TemporalLoss(**kwargs)
    elif loss_type == "liquid":
        return LiquidLoss(**kwargs)
    elif loss_type == "contrastive":
        return ContrastiveLoss(**kwargs)
    elif loss_type == "sequence":
        return EventSequenceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")