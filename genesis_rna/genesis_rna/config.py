"""
Configuration Module for Genesis RNA

Defines model architecture and training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenesisRNAConfig:
    """
    Configuration for Genesis RNA model architecture.

    This config defines the Transformer-based encoder architecture for
    RNA sequence modeling with support for multi-task learning.

    Attributes:
        vocab_size: Size of the token vocabulary
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dim_ff: Feedforward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        structure_num_labels: Number of structure labels (e.g., STEM, LOOP, BULGE)
        use_rotary_embeddings: Whether to use rotary positional embeddings
        attention_type: Type of attention mechanism ('standard', 'linear', 'flash')
        layer_norm_eps: Epsilon for layer normalization
        initializer_range: Standard deviation for weight initialization
    """

    # Vocabulary
    vocab_size: int = 9  # 4 nucleotides + N + 4 special tokens

    # Architecture dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dim_ff: int = 2048
    max_len: int = 512

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0

    # Task-specific
    structure_num_labels: int = 5  # NONE, STEM, LOOP, BULGE, HAIRPIN

    # Positional encoding
    use_rotary_embeddings: bool = False
    max_position_embeddings: int = 512

    # Attention settings
    attention_type: str = "standard"  # 'standard', 'linear', 'flash'

    # Normalization
    layer_norm_eps: float = 1e-12

    # Initialization
    initializer_range: float = 0.02

    # Model type identifier
    model_type: str = "genesis_rna"

    def __post_init__(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"

    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        return self.d_model // self.n_heads

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """
    Training configuration for Genesis RNA pretraining.

    Attributes:
        batch_size: Training batch size
        learning_rate: Peak learning rate
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps (overrides num_epochs if set)
        weight_decay: Weight decay coefficient
        gradient_clip_norm: Maximum gradient norm for clipping
        mlm_probability: Probability of masking tokens for MLM
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log metrics every N steps
        output_dir: Directory for saving checkpoints and logs
    """

    # Batch and optimization
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 10000
    max_steps: Optional[int] = None
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    # MLM settings
    mlm_probability: float = 0.15

    # Multi-task loss weights
    mlm_loss_weight: float = 1.0
    structure_loss_weight: float = 0.5
    pair_loss_weight: float = 0.1

    # Checkpointing and logging
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 100
    output_dir: str = "./checkpoints"

    # Device and mixed precision
    device: str = "cuda"
    mixed_precision: bool = True
    fp16: bool = True

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2

    # AST settings
    use_ast: bool = True
    ast_target_activation: float = 0.4
    ast_controller_kp: float = 0.01
    ast_controller_ki: float = 0.001

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate training configuration"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.mlm_probability < 1, "mlm_probability must be in (0, 1)"


@dataclass
class GenesisRNAConfigSmall(GenesisRNAConfig):
    """Small model configuration for testing and development"""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dim_ff: int = 1024
    max_len: int = 512


@dataclass
class GenesisRNAConfigBase(GenesisRNAConfig):
    """Base model configuration (default)"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dim_ff: int = 2048
    max_len: int = 512


@dataclass
class GenesisRNAConfigLarge(GenesisRNAConfig):
    """Large model configuration for high-capacity training"""
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dim_ff: int = 3072
    max_len: int = 1024
