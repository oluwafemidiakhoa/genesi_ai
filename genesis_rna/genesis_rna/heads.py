"""
Prediction Heads for Genesis RNA

Multi-task prediction heads for:
1. Masked Language Modeling (MLM)
2. Secondary structure prediction
3. Base-pair prediction
"""

import torch
import torch.nn as nn
from typing import Optional


class MLMHead(nn.Module):
    """
    Masked Language Modeling (MLM) head.

    Predicts masked nucleotide tokens from hidden states.
    Similar to BERT's MLM head architecture.

    Architecture:
        Hidden → Dense → GELU → LayerNorm → Output projection → Vocab logits
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model: Hidden dimension size
            vocab_size: Size of token vocabulary
        """
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Encoder outputs [batch_size, seq_len, d_model]

        Returns:
            Logits over vocabulary [batch_size, seq_len, vocab_size]
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class StructureHead(nn.Module):
    """
    Secondary structure prediction head.

    Predicts RNA secondary structure type for each position:
    - NONE (unpaired)
    - STEM (base-paired helix)
    - LOOP (hairpin loop)
    - BULGE (bulge/internal loop)
    - HAIRPIN (hairpin loop)

    This is a token-level classification task.
    """

    def __init__(self, d_model: int, num_labels: int):
        """
        Args:
            d_model: Hidden dimension size
            num_labels: Number of structure types (default: 5)
        """
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Encoder outputs [batch_size, seq_len, d_model]

        Returns:
            Structure logits [batch_size, seq_len, num_labels]
        """
        x = self.dropout(hidden_states)
        logits = self.classifier(x)
        return logits


class PairHead(nn.Module):
    """
    Base-pair prediction head.

    Predicts which positions form base pairs (Watson-Crick or wobble pairs).
    Uses a bilinear scoring function to compute pair probabilities.

    Output is a symmetric matrix where entry (i, j) represents the
    probability that positions i and j form a base pair.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        """
        Args:
            d_model: Hidden dimension size
            hidden_dim: Intermediate projection dimension (default: d_model)
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model

        # Project hidden states for pairing scores
        self.proj_left = nn.Linear(d_model, hidden_dim, bias=False)
        self.proj_right = nn.Linear(d_model, hidden_dim, bias=False)

        # Optional: add a learned scaling factor
        self.scale = nn.Parameter(torch.ones(1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        nn.init.normal_(self.proj_left.weight, std=0.02)
        nn.init.normal_(self.proj_right.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Encoder outputs [batch_size, seq_len, d_model]

        Returns:
            Pair logits [batch_size, seq_len, seq_len]
            Entry (b, i, j) is the score for position i pairing with position j
        """
        # Project hidden states
        h_left = self.proj_left(hidden_states)   # [B, L, H]
        h_right = self.proj_right(hidden_states)  # [B, L, H]

        # Compute bilinear pairing scores: h_left @ h_right^T
        # This gives a matrix where entry (i,j) = h_left[i] · h_right[j]
        pair_scores = torch.matmul(h_left, h_right.transpose(-2, -1))  # [B, L, L]

        # Apply scaling
        pair_scores = pair_scores * self.scale

        # Optionally symmetrize (since base pairing is symmetric)
        # pair_scores = (pair_scores + pair_scores.transpose(-2, -1)) / 2

        return pair_scores


class ContactMapHead(nn.Module):
    """
    Alternative contact map prediction head using outer product.

    This head is more parameter-efficient than PairHead and explicitly
    constructs pairwise features via outer product.
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model: Hidden dimension size
        """
        super().__init__()

        # Reduce dimension for efficiency
        self.proj = nn.Linear(d_model, d_model // 4)
        self.activation = nn.ReLU()

        # Final classifier for pair
        self.classifier = nn.Linear(1, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Encoder outputs [batch_size, seq_len, d_model]

        Returns:
            Contact map logits [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = hidden_states.size()

        # Project to lower dimension
        h = self.proj(hidden_states)  # [B, L, d//4]
        h = self.activation(h)

        # Compute outer product for all pairs
        # h[i] * h[j] gives pairwise features
        h_i = h.unsqueeze(2)  # [B, L, 1, d//4]
        h_j = h.unsqueeze(1)  # [B, 1, L, d//4]

        # Element-wise product
        pairwise = h_i * h_j  # [B, L, L, d//4]

        # Reduce to scalar score per pair
        contact_scores = pairwise.sum(dim=-1, keepdim=True)  # [B, L, L, 1]
        contact_scores = self.classifier(contact_scores)      # [B, L, L, 1]
        contact_scores = contact_scores.squeeze(-1)           # [B, L, L]

        return contact_scores


class MutationEffectHead(nn.Module):
    """
    Mutation effect prediction head (for fine-tuning).

    Predicts the effect of mutations on RNA function, stability, or binding.
    Can be used for:
    - Fitness prediction
    - Stability change (ΔΔG)
    - Binding affinity change

    This is typically a sequence-level regression task.
    """

    def __init__(self, d_model: int, pooling: str = "mean"):
        """
        Args:
            d_model: Hidden dimension size
            pooling: How to pool sequence representations ("mean", "max", "cls")
        """
        super().__init__()
        self.pooling = pooling

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Encoder outputs [batch_size, seq_len, d_model]
            attention_mask: Binary mask [batch_size, seq_len]

        Returns:
            Effect predictions [batch_size, 1]
        """
        # Pool sequence representation
        if self.pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()
                summed = (hidden_states * mask_expanded).sum(dim=1)
                pooled = summed / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.pooling == "max":
            pooled = hidden_states.max(dim=1)[0]

        elif self.pooling == "cls":
            # Use [CLS] token (first position)
            pooled = hidden_states[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Predict effect
        effect = self.regressor(pooled)
        return effect
