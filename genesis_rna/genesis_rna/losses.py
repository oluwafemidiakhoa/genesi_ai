"""
Loss Functions for Genesis RNA Multi-Task Learning

Implements loss functions for:
1. Masked Language Modeling (MLM)
2. Secondary structure prediction
3. Base-pair prediction
4. Combined multi-task loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: Optional[int] = None
) -> torch.Tensor:
    """
    Masked Language Modeling (MLM) loss.

    Computes cross-entropy loss for masked token prediction.
    Ignores positions with label -100.

    Args:
        logits: Predicted logits [batch_size, seq_len, vocab_size]
        labels: True token IDs [batch_size, seq_len]
                -100 for positions to ignore

    Returns:
        Scalar loss tensor
    """
    # Flatten for cross-entropy
    # CE expects [N, C] and [N]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    return loss


def structure_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: Optional[int] = None
) -> torch.Tensor:
    """
    Secondary structure prediction loss.

    Computes cross-entropy loss for structure type classification.
    Ignores positions with label -100.

    Args:
        logits: Predicted logits [batch_size, seq_len, num_labels]
        labels: True structure labels [batch_size, seq_len]
                -100 for positions to ignore

    Returns:
        Scalar loss tensor
    """
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    return loss


def pair_loss(
    pair_logits: torch.Tensor,
    pair_matrix: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Base-pair prediction loss.

    Computes binary cross-entropy loss for base-pair adjacency prediction.

    Args:
        pair_logits: Predicted pair scores [batch_size, seq_len, seq_len]
        pair_matrix: Ground truth pairs [batch_size, seq_len, seq_len]
                    0 or 1 for each pair
        attention_mask: Optional mask [batch_size, seq_len]
                       to ignore padding positions

    Returns:
        Scalar loss tensor
    """
    # Binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(
        pair_logits,
        pair_matrix,
        reduction='none'
    )

    # Mask out padding positions
    if attention_mask is not None:
        # Create pairwise mask: both positions must be valid
        mask_2d = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)  # [B, L, L]
        loss = loss * mask_2d.float()

        # Average over valid positions only
        num_valid = mask_2d.sum() + 1e-8
        loss = loss.sum() / num_valid
    else:
        loss = loss.mean()

    return loss


def contact_loss(
    contact_logits: torch.Tensor,
    contact_matrix: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    weight_positive: float = 2.0
) -> torch.Tensor:
    """
    Contact map prediction loss with class imbalance handling.

    Since most positions do NOT form contacts, we use weighted BCE
    to balance positive and negative examples.

    Args:
        contact_logits: Predicted contact scores [batch_size, seq_len, seq_len]
        contact_matrix: Ground truth contacts [batch_size, seq_len, seq_len]
        attention_mask: Optional mask [batch_size, seq_len]
        weight_positive: Weight for positive examples (contacts)

    Returns:
        Scalar loss tensor
    """
    # Create position-wise weights (higher weight for positive contacts)
    pos_weight = torch.ones_like(contact_matrix) * weight_positive

    loss = F.binary_cross_entropy_with_logits(
        contact_logits,
        contact_matrix,
        pos_weight=pos_weight,
        reduction='none'
    )

    # Mask padding
    if attention_mask is not None:
        mask_2d = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
        loss = loss * mask_2d.float()
        loss = loss.sum() / (mask_2d.sum() + 1e-8)
    else:
        loss = loss.mean()

    return loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for Genesis RNA pretraining.

    Combines:
    1. MLM loss (primary task)
    2. Structure prediction loss
    3. Base-pair prediction loss

    Each task can be weighted differently.
    """

    def __init__(
        self,
        mlm_weight: float = 1.0,
        structure_weight: float = 0.5,
        pair_weight: float = 0.1,
        use_focal_loss_for_pairs: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            mlm_weight: Weight for MLM loss
            structure_weight: Weight for structure prediction loss
            pair_weight: Weight for base-pair prediction loss
            use_focal_loss_for_pairs: Use focal loss for pairs (handles imbalance)
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.mlm_weight = mlm_weight
        self.structure_weight = structure_weight
        self.pair_weight = pair_weight
        self.use_focal_loss_for_pairs = use_focal_loss_for_pairs

        # Create focal loss if enabled
        if use_focal_loss_for_pairs:
            self.focal_loss_fn = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs dictionary with:
                - mlm_logits
                - struct_logits
                - pair_logits
            batch: Batch dictionary with:
                - mlm_labels
                - struct_labels
                - pair_matrix
                - attention_mask

        Returns:
            Dictionary with:
                - loss: Combined loss
                - mlm_loss: MLM loss component
                - structure_loss: Structure loss component
                - pair_loss: Pair loss component
        """
        # MLM loss
        loss_mlm = mlm_loss(outputs["mlm_logits"], batch["mlm_labels"])

        # Structure loss
        loss_struct = structure_loss(outputs["struct_logits"], batch["struct_labels"])

        # Pair loss (use focal loss if enabled for better handling of imbalance)
        if self.use_focal_loss_for_pairs:
            loss_pair = self.focal_loss_fn(
                outputs["pair_logits"],
                batch["pair_matrix"],
                batch.get("attention_mask")
            )
        else:
            loss_pair = pair_loss(
                outputs["pair_logits"],
                batch["pair_matrix"],
                batch.get("attention_mask")
            )

        # Combined weighted loss
        total_loss = (
            self.mlm_weight * loss_mlm +
            self.structure_weight * loss_struct +
            self.pair_weight * loss_pair
        )

        return {
            "loss": total_loss,
            "mlm_loss": loss_mlm,
            "structure_loss": loss_struct,
            "pair_loss": loss_pair,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Useful for structure prediction where some classes are rare.
    FL(p_t) = -α(1-p_t)^γ log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor in [0, 1]
            gamma: Focusing parameter (γ ≥ 0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [batch_size, seq_len, num_classes]
            labels: True labels [batch_size, seq_len]

        Returns:
            Scalar loss
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get prob of correct class
        labels_one_hot = F.one_hot(labels.clamp(min=0), num_classes=logits.size(-1))
        pt = (probs * labels_one_hot).sum(dim=-1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Cross entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )

        # Apply focal weight
        focal_loss = focal_weight.view(-1) * ce_loss

        # Filter out ignored indices
        valid_mask = (labels.view(-1) != -100)
        focal_loss = focal_loss[valid_mask].mean()

        return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for handling severe class imbalance in pair prediction.

    Most RNA positions don't pair, so we need to focus on hard examples.
    FL(p_t) = -α(1-p_t)^γ log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        """
        Args:
            alpha: Weight for positive class (higher = more weight on actual pairs)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted pair scores [batch_size, seq_len, seq_len]
            labels: Ground truth pairs [batch_size, seq_len, seq_len]
            attention_mask: Optional mask [batch_size, seq_len]

        Returns:
            Scalar loss
        """
        # Get probabilities
        probs = torch.sigmoid(logits)

        # Compute focal weight
        # For positive examples: (1-p)^gamma
        # For negative examples: p^gamma
        focal_weight = torch.where(
            labels == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma
        )

        # Compute base BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction='none'
        )

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply alpha balancing (higher weight for positive pairs)
        alpha_weight = torch.where(
            labels == 1,
            torch.ones_like(labels) * self.alpha,
            torch.ones_like(labels) * (1 - self.alpha)
        )
        focal_loss = alpha_weight * focal_loss

        # Mask padding
        if attention_mask is not None:
            mask_2d = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
            focal_loss = focal_loss * mask_2d.float()
            loss = focal_loss.sum() / (mask_2d.sum() + 1e-8)
        else:
            loss = focal_loss.mean()

        return loss


def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for all tasks.

    Args:
        outputs: Model outputs
        batch: Ground truth batch

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # MLM accuracy
    mlm_preds = outputs["mlm_logits"].argmax(dim=-1)
    mlm_labels = batch["mlm_labels"]
    mlm_mask = (mlm_labels != -100)
    if mlm_mask.sum() > 0:
        mlm_correct = (mlm_preds[mlm_mask] == mlm_labels[mlm_mask]).float().sum()
        mlm_total = mlm_mask.sum()
        metrics["mlm_accuracy"] = (mlm_correct / mlm_total).item()

    # Structure accuracy
    struct_preds = outputs["struct_logits"].argmax(dim=-1)
    struct_labels = batch["struct_labels"]
    struct_mask = (struct_labels != -100)
    if struct_mask.sum() > 0:
        struct_correct = (struct_preds[struct_mask] == struct_labels[struct_mask]).float().sum()
        struct_total = struct_mask.sum()
        metrics["structure_accuracy"] = (struct_correct / struct_total).item()

    # Pair prediction metrics (precision, recall, F1)
    pair_preds = (torch.sigmoid(outputs["pair_logits"]) > 0.5).float()
    pair_labels = batch["pair_matrix"]

    if "attention_mask" in batch:
        mask_2d = batch["attention_mask"].unsqueeze(-1) * batch["attention_mask"].unsqueeze(-2)
        pair_preds = pair_preds * mask_2d.float()
        pair_labels = pair_labels * mask_2d.float()

    tp = (pair_preds * pair_labels).sum()
    fp = (pair_preds * (1 - pair_labels)).sum()
    fn = ((1 - pair_preds) * pair_labels).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics["pair_precision"] = precision.item()
    metrics["pair_recall"] = recall.item()
    metrics["pair_f1"] = f1.item()

    return metrics
