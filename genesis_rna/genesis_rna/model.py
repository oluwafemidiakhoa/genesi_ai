"""
Model Architecture for Genesis RNA

Implements a Transformer-based encoder with RNA-specific features:
- RNA-aware positional encodings
- Multi-head self-attention
- Support for secondary structure and base-pair prediction
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .config import GenesisRNAConfig
from .heads import MLMHead, StructureHead, PairHead


class RNAEmbedding(nn.Module):
    """
    RNA sequence embeddings with positional encoding.

    Combines:
    - Token embeddings (nucleotides + special tokens)
    - Positional embeddings (learned or sinusoidal)
    - Layer normalization and dropout
    """

    def __init__(self, cfg: GenesisRNAConfig):
        super().__init__()
        self.cfg = cfg

        # Token embeddings
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Positional embeddings
        if cfg.use_rotary_embeddings:
            # Rotary embeddings will be applied in attention
            self.pos_embeddings = None
        else:
            # Standard learned positional embeddings
            self.pos_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.d_model)

        self.layer_norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embeddings.weight, std=self.cfg.initializer_range)
        if self.pos_embeddings is not None:
            nn.init.normal_(self.pos_embeddings.weight, std=self.cfg.initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.size()

        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Add positional embeddings
        if self.pos_embeddings is not None:
            position_ids = torch.arange(
                seq_len,
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            pos_embeds = self.pos_embeddings(position_ids)
            embeddings = token_embeds + pos_embeds
        else:
            embeddings = token_embeds

        # Normalize and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.

    Architecture:
        1. Multi-head self-attention
        2. Add & Norm
        3. Feedforward network
        4. Add & Norm
    """

    def __init__(self, cfg: GenesisRNAConfig):
        super().__init__()
        self.cfg = cfg

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.attention_dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_ff),
            nn.GELU(),
            nn.Dropout(cfg.activation_dropout),
            nn.Linear(cfg.dim_ff, cfg.d_model),
        )

        # Dropout
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.cfg.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [seq_len, seq_len] (optional)
            key_padding_mask: Padding mask [batch_size, seq_len] (optional)
                            True for positions to ignore

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Multi-head attention with residual
        attn_output, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.dropout1(attn_output)
        hidden_states = self.norm1(hidden_states)

        # Feedforward with residual
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + self.dropout2(ffn_output)
        hidden_states = self.norm2(hidden_states)

        return hidden_states


class GenesisRNAEncoder(nn.Module):
    """
    RNA Transformer Encoder.

    Multi-layer Transformer encoder for processing RNA sequences.
    """

    def __init__(self, cfg: GenesisRNAConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.embeddings = RNAEmbedding(cfg)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Binary mask [batch_size, seq_len]
                          1 for real tokens, 0 for padding

        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        # Get embeddings
        hidden_states = self.embeddings(input_ids)

        # Prepare padding mask for attention
        # PyTorch MultiheadAttention uses True for positions to IGNORE
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                key_padding_mask=key_padding_mask
            )

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        return hidden_states


class GenesisRNAModel(nn.Module):
    """
    Complete Genesis RNA model with multi-task heads.

    Combines:
    1. GenesisRNAEncoder (Transformer backbone)
    2. MLMHead (masked language modeling)
    3. StructureHead (secondary structure prediction)
    4. PairHead (base-pair prediction)

    Example:
        >>> config = GenesisRNAConfig()
        >>> model = GenesisRNAModel(config)
        >>> input_ids = torch.randint(0, 9, (4, 128))
        >>> outputs = model(input_ids)
        >>> mlm_logits = outputs["mlm_logits"]  # [4, 128, vocab_size]
    """

    def __init__(self, cfg: GenesisRNAConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.encoder = GenesisRNAEncoder(cfg)

        # Task-specific heads
        self.mlm_head = MLMHead(cfg.d_model, cfg.vocab_size)
        self.struct_head = StructureHead(cfg.d_model, cfg.structure_num_labels)
        self.pair_head = PairHead(cfg.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        Forward pass with multi-task outputs.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Binary attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return encoder hidden states

        Returns:
            Dictionary with keys:
                - mlm_logits: MLM predictions [batch_size, seq_len, vocab_size]
                - struct_logits: Structure predictions [batch_size, seq_len, num_struct_labels]
                - pair_logits: Pair predictions [batch_size, seq_len, seq_len]
                - hidden_states: Encoder outputs [batch_size, seq_len, d_model] (if requested)
        """
        # Encode
        hidden_states = self.encoder(input_ids, attention_mask)

        # Compute task-specific predictions
        mlm_logits = self.mlm_head(hidden_states)
        struct_logits = self.struct_head(hidden_states)
        pair_logits = self.pair_head(hidden_states)

        outputs = {
            "mlm_logits": mlm_logits,
            "struct_logits": struct_logits,
            "pair_logits": pair_logits,
        }

        if return_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get sequence embeddings (for downstream tasks).

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        return self.encoder(input_ids)

    def save_pretrained(self, save_path: str):
        """Save model weights and config"""
        # Save config as dict for better serialization
        config_dict = self.cfg.to_dict() if hasattr(self.cfg, 'to_dict') else self.cfg
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config_dict,
        }, save_path)

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(load_path, map_location=device)

        # Handle config - convert dict to GenesisRNAConfig if needed
        config = checkpoint['config']
        if isinstance(config, dict):
            config = GenesisRNAConfig.from_dict(config)

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)
