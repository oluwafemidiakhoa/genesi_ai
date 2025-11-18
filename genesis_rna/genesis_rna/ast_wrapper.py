"""
Adaptive Sparse Training (AST) Integration Wrapper

Provides integration with the adaptive-sparse-training library for
energy-efficient training via adaptive sample selection.

Core concept:
1. Compute per-sample loss/importance scores
2. AST controller selects subset of "important" samples
3. Backprop only on selected samples → reduced FLOPs & energy

References:
- Your adaptive-sparse-training library on PyPI
- PI controller for adaptive activation rate
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
import numpy as np


class PIController:
    """
    Proportional-Integral (PI) Controller for adaptive threshold adjustment.

    Dynamically adjusts the selection threshold to maintain a target
    activation rate (fraction of samples selected per batch).
    """

    def __init__(
        self,
        target_activation: float = 0.4,
        kp: float = 0.01,
        ki: float = 0.001,
        threshold_init: float = 0.5,
    ):
        """
        Args:
            target_activation: Target fraction of samples to select (0-1)
            kp: Proportional gain
            ki: Integral gain
            threshold_init: Initial threshold value
        """
        self.target_activation = target_activation
        self.kp = kp
        self.ki = ki
        self.threshold = threshold_init
        self.integral = 0.0

    def update(self, current_activation: float) -> float:
        """
        Update threshold based on current activation rate.

        Args:
            current_activation: Fraction of samples actually selected

        Returns:
            Updated threshold
        """
        # Error: difference from target
        error = self.target_activation - current_activation

        # Update integral term
        self.integral += error

        # PI control
        adjustment = self.kp * error + self.ki * self.integral

        # Update threshold
        self.threshold += adjustment

        # Clamp threshold to reasonable range
        self.threshold = np.clip(self.threshold, 0.01, 0.99)

        return self.threshold

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0


class ASTSampleSelector:
    """
    Adaptive Sparse Training sample selector.

    Selects a subset of training samples based on importance scores,
    with adaptive threshold control to maintain target activation rate.

    This is a placeholder implementation that demonstrates the interface.
    When the actual adaptive-sparse-training library is available, this
    class can be replaced or extended with the real implementation.
    """

    def __init__(
        self,
        target_activation: float = 0.4,
        selection_mode: str = "loss",
        controller_kp: float = 0.01,
        controller_ki: float = 0.001,
        use_pi_controller: bool = True,
    ):
        """
        Args:
            target_activation: Target fraction of samples to activate (0-1)
            selection_mode: How to compute importance scores
                          - "loss": Select by loss magnitude
                          - "gradient": Select by gradient norm
                          - "uncertainty": Select by prediction uncertainty
            controller_kp: Proportional gain for PI controller
            controller_ki: Integral gain for PI controller
            use_pi_controller: Whether to use adaptive threshold control
        """
        self.target_activation = target_activation
        self.selection_mode = selection_mode
        self.use_pi_controller = use_pi_controller

        # PI controller for adaptive threshold
        if use_pi_controller:
            self.controller = PIController(
                target_activation=target_activation,
                kp=controller_kp,
                ki=controller_ki,
            )
        else:
            self.controller = None

        # Statistics
        self.total_samples_seen = 0
        self.total_samples_selected = 0
        self.history = []

    def compute_importance_scores(
        self,
        sample_losses: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute importance scores for each sample.

        Args:
            sample_losses: Per-sample loss values [batch_size]
            gradients: Optional per-sample gradient norms [batch_size]
            predictions: Optional prediction logits for uncertainty

        Returns:
            Importance scores [batch_size]
        """
        if self.selection_mode == "loss":
            # Higher loss = more important
            scores = sample_losses

        elif self.selection_mode == "gradient":
            # Higher gradient norm = more important
            if gradients is None:
                raise ValueError("Gradients required for gradient-based selection")
            scores = gradients

        elif self.selection_mode == "uncertainty":
            # Higher uncertainty = more important
            if predictions is None:
                raise ValueError("Predictions required for uncertainty-based selection")
            # Use entropy as uncertainty measure
            probs = torch.softmax(predictions, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            scores = entropy

        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

        return scores

    def select_indices(
        self,
        sample_losses: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select indices of samples to train on.

        Args:
            sample_losses: Per-sample losses [batch_size]
            gradients: Optional gradient norms [batch_size]
            predictions: Optional prediction logits

        Returns:
            Indices of selected samples (long tensor)
        """
        batch_size = sample_losses.size(0)

        # Compute importance scores
        scores = self.compute_importance_scores(
            sample_losses,
            gradients,
            predictions
        )

        # Determine selection threshold
        if self.use_pi_controller and self.controller is not None:
            # Use adaptive threshold from PI controller
            threshold = self.controller.threshold
            threshold_value = torch.quantile(scores, 1.0 - threshold)
            selected_mask = scores >= threshold_value

            # Update controller based on actual activation
            actual_activation = selected_mask.float().mean().item()
            self.controller.update(actual_activation)

        else:
            # Fixed top-k selection
            k = max(1, int(self.target_activation * batch_size))
            _, top_indices = torch.topk(scores, k)
            selected_mask = torch.zeros_like(scores, dtype=torch.bool)
            selected_mask[top_indices] = True

        # Get selected indices
        selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)

        # Update statistics
        self.total_samples_seen += batch_size
        self.total_samples_selected += selected_indices.size(0)

        # Log history
        self.history.append({
            'batch_size': batch_size,
            'num_selected': selected_indices.size(0),
            'activation_rate': selected_indices.size(0) / batch_size,
            'mean_score': scores.mean().item(),
            'threshold': threshold if self.use_pi_controller else None,
        })

        return selected_indices

    def get_stats(self) -> Dict[str, float]:
        """Get selection statistics"""
        if self.total_samples_seen == 0:
            return {}

        return {
            'total_samples_seen': self.total_samples_seen,
            'total_samples_selected': self.total_samples_selected,
            'overall_activation_rate': self.total_samples_selected / self.total_samples_seen,
            'target_activation_rate': self.target_activation,
            'current_threshold': self.controller.threshold if self.controller else None,
        }

    def reset_stats(self):
        """Reset statistics"""
        self.total_samples_seen = 0
        self.total_samples_selected = 0
        self.history = []
        if self.controller:
            self.controller.reset()


# TODO: Integration with actual adaptive-sparse-training library
#
# Once the library is installed and documented, replace the above
# implementation with calls to the real AST API. The interface should
# remain similar:
#
# Example usage pattern:
# ```
# from adaptive_sparse_training import ASTEngine, PIController
#
# ast_engine = ASTEngine(
#     target_activation=0.4,
#     controller_type='pi',
#     energy_tracking=True,
# )
#
# # In training loop:
# selected_idx = ast_engine.select_samples(
#     losses=sample_losses,
#     gradients=sample_gradients,
# )
#
# # Train only on selected samples
# loss = compute_loss(outputs[selected_idx], labels[selected_idx])
# loss.backward()
# ```


def compute_per_sample_loss(
    loss_fn,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = 'none'
) -> torch.Tensor:
    """
    Compute per-sample loss (helper function).

    Args:
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss)
        outputs: Model outputs
        labels: Ground truth labels
        reduction: Reduction mode ('none' for per-sample)

    Returns:
        Per-sample losses
    """
    # This is a simplified helper - actual implementation will depend
    # on the specific loss function and task
    if reduction != 'none':
        raise ValueError("reduction must be 'none' for per-sample loss")

    # For cross-entropy on sequences:
    # outputs: [B, L, C], labels: [B, L]
    # We want loss per sample (average over sequence)

    if len(outputs.shape) == 3:  # [B, L, C]
        batch_size, seq_len, num_classes = outputs.shape
        # Compute token-level loss
        token_losses = loss_fn(
            outputs.view(-1, num_classes),
            labels.view(-1)
        ).view(batch_size, seq_len)

        # Average over sequence (ignoring padding if needed)
        mask = (labels != -100).float()
        sample_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    else:  # Simple case: [B, C]
        sample_losses = loss_fn(outputs, labels)

    return sample_losses


class ASTTrainingWrapper:
    """
    High-level wrapper for AST-enabled training.

    Provides a simple interface for training with adaptive sample selection.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        ast_selector: ASTSampleSelector,
        device: str = 'cuda',
    ):
        """
        Args:
            model: Neural network model
            optimizer: Optimizer
            loss_fn: Loss function
            ast_selector: AST sample selector
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.ast_selector = ast_selector
        self.device = device

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        compute_sample_losses: bool = True,
    ) -> Dict[str, float]:
        """
        Perform one training step with AST sample selection.

        Args:
            batch: Batch of training data
            compute_sample_losses: Whether to compute per-sample losses for selection

        Returns:
            Dictionary with loss and metrics
        """
        # Forward pass (all samples)
        outputs = self.model(
            batch['input_ids'],
            attention_mask=batch.get('attention_mask')
        )

        # Compute per-sample losses for selection
        with torch.no_grad():
            # Simplified: use MLM loss as selection criterion
            # In practice, you might want to use combined loss or other metrics
            token_losses = torch.nn.functional.cross_entropy(
                outputs['mlm_logits'].view(-1, outputs['mlm_logits'].size(-1)),
                batch['mlm_labels'].view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(outputs['mlm_logits'].size(0), -1)

            # Average over sequence
            mask = (batch['mlm_labels'] != -100).float()
            sample_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Select samples
        selected_idx = self.ast_selector.select_indices(sample_losses)

        if selected_idx.numel() == 0:
            # No samples selected - skip this batch
            return {
                'loss': 0.0,
                'num_selected': 0,
                'activation_rate': 0.0,
            }

        # Compute loss only on selected samples
        loss_dict = self.loss_fn(
            {k: v[selected_idx] for k, v in outputs.items()},
            {k: v[selected_idx] if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
        )

        loss = loss_dict['loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Return metrics
        return {
            'loss': loss.item(),
            'num_selected': selected_idx.size(0),
            'activation_rate': selected_idx.size(0) / batch['input_ids'].size(0),
            **{k: v.item() if torch.is_tensor(v) else v
               for k, v in loss_dict.items() if k != 'loss'}
        }
