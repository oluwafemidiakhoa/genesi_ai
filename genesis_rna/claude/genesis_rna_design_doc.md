# Genesis RNA: RNA Foundation Model with Adaptive Sparse Training

## Vision: What is genesis_rna?

A general-purpose RNA foundation model (RiNALMo-style) trained with **Adaptive Sparse Training (AST)** for energy-efficient pretraining, then specialized into agents inside GENESIS-AI for:

- RNA design
- Mutation effect prediction
- RNA-protein interaction prediction
- mRNA therapeutic optimization

### Core Ideas

**Architecture**: Transformer + RNA-aware positional embeddings + secondary-structure embeddings

**Pretraining Tasks**:
1. Masked nucleotide modeling (MLM)
2. Paired-base prediction
3. Structure-context prediction (loop/stem/bulge)
4. Mutation effect prediction (Δfitness / Δstability)
5. RNA-protein binding prediction

**Efficiency**: Train with Adaptive Sparse Training library (`adaptive-sparse-training`), using energy-aware sample selection.

---

## Project Structure

```
genesis_rna/
  genesis_rna/               # Main Python package
    __init__.py             # Package initialization
    config.py               # Model and training configurations
    tokenization.py         # RNA tokenizer (A, C, G, U, N + special tokens)
    data.py                 # Dataset classes and data loading
    model.py                # Transformer encoder architecture
    heads.py                # Task-specific prediction heads
    losses.py               # Multi-task loss functions
    ast_wrapper.py          # AST integration (sample selection)
    train_pretrain.py       # Main pretraining script
    train_finetune.py       # Fine-tuning script (TODO)
    eval_benchmarks.py      # Evaluation on benchmarks (TODO)

  scripts/                  # Utility scripts
    download_rnacentral.sh  # Download RNAcentral data
    preprocess_rna.py       # Preprocess RNA sequences

  experiments/              # Experiment configurations
    config_pretrain_small.yaml
    config_pretrain_base.yaml
    config_pretrain_large.yaml

  tests/                    # Unit tests
    test_tokenization.py
    test_model_forward.py
    test_data.py

  claude/                   # Documentation for Claude
    genesis_rna_design_doc.md  # This file

  requirements.txt          # Python dependencies
  README.md                 # Project overview
```

---

## Architecture Details

### 1. Tokenization (`tokenization.py`)

**Vocabulary**:
- Nucleotides: `A`, `C`, `G`, `U`, `N` (unknown)
- Special tokens: `[PAD]`, `[MASK]`, `[CLS]`, `[SEP]`
- Total vocab size: 9

**Key Features**:
- `encode()`: Convert RNA sequence to token IDs
- `random_mask()`: BERT-style masking (80% [MASK], 10% random, 10% original)
- Handles padding and truncation to max sequence length

### 2. Model Architecture (`model.py`)

#### RNAEmbedding
- Token embeddings (nucleotides + special tokens)
- Positional embeddings (learned or sinusoidal)
- Layer normalization + dropout

#### TransformerBlock
- Multi-head self-attention
- Feedforward network (Linear → GELU → Linear)
- Pre-normalization (LayerNorm before attention/FFN)
- Residual connections

#### GenesisRNAEncoder
- Stack of N transformer blocks
- Final layer normalization
- Output: Hidden states `[batch_size, seq_len, d_model]`

#### GenesisRNAModel
- Combines encoder + task-specific heads
- Multi-task outputs:
  - MLM logits `[B, L, vocab_size]`
  - Structure logits `[B, L, num_struct_labels]`
  - Pair logits `[B, L, L]` (base-pairing matrix)

### 3. Prediction Heads (`heads.py`)

#### MLMHead
- Predicts masked nucleotides
- Architecture: Dense → GELU → LayerNorm → Decoder

#### StructureHead
- Predicts secondary structure type (NONE, STEM, LOOP, BULGE, HAIRPIN)
- Simple linear classifier

#### PairHead
- Predicts base-pair adjacency matrix
- Bilinear scoring: `score(i,j) = h_i^T W h_j`

#### MutationEffectHead (for fine-tuning)
- Predicts mutation effect (Δfitness, ΔΔG, etc.)
- Sequence-level regression with pooling

### 4. Loss Functions (`losses.py`)

- **MLM Loss**: Cross-entropy over masked positions
- **Structure Loss**: Cross-entropy for structure labels
- **Pair Loss**: Binary cross-entropy for base pairs
- **MultiTaskLoss**: Weighted combination of all losses

### 5. Data Loading (`data.py`)

#### RNAPretrainSample
- Dataclass for single RNA sequence
- Fields: `seq`, `struct_labels`, `pair_indices`, `metadata`

#### RNAPretrainDataset
- PyTorch Dataset for pretraining
- Applies random masking
- Returns: `input_ids`, `mlm_labels`, `struct_labels`, `pair_matrix`, `attention_mask`

### 6. Adaptive Sparse Training (`ast_wrapper.py`)

**Core Concept**: Train only on "important" samples per batch → reduce FLOPs & energy

#### ASTSampleSelector
- Computes importance scores (loss-based, gradient-based, or uncertainty-based)
- Selects top-k samples using PI controller
- Dynamically adjusts threshold to maintain target activation rate

#### PIController
- Proportional-Integral controller
- Maintains stable activation rate (e.g., 40% of samples)
- Adjusts threshold based on error: `threshold += kp * error + ki * integral`

**Integration**:
1. Forward pass on full batch
2. Compute per-sample losses
3. AST selector chooses important samples
4. Backprop only on selected samples

---

## Training Pipeline

### Pretraining (`train_pretrain.py`)

**Command**:
```bash
python -m genesis_rna.train_pretrain \
  --model_size base \
  --batch_size 16 \
  --num_epochs 10 \
  --learning_rate 1e-4 \
  --use_ast \
  --ast_target_activation 0.4 \
  --output_dir ./checkpoints
```

**Features**:
- Multi-task learning (MLM + structure + pairing)
- AST-enabled sample selection
- Mixed precision training (FP16)
- Learning rate warmup + cosine decay
- Gradient clipping
- Checkpoint management
- Validation evaluation

**Key Hyperparameters**:
- `d_model`: 512 (base), 256 (small), 768 (large)
- `n_layers`: 8 (base), 4 (small), 12 (large)
- `n_heads`: 8 (base), 4 (small), 12 (large)
- `max_len`: 512 tokens
- `mlm_probability`: 0.15
- `ast_target_activation`: 0.4 (train on 40% of samples)

---

## Configuration System

### GenesisRNAConfig (model architecture)
```python
d_model: int = 512           # Hidden dimension
n_heads: int = 8             # Attention heads
n_layers: int = 8            # Transformer layers
dim_ff: int = 2048           # FFN dimension
max_len: int = 512           # Max sequence length
dropout: float = 0.1         # Dropout rate
vocab_size: int = 9          # Token vocabulary size
structure_num_labels: int = 5 # Structure types
```

### TrainingConfig (training hyperparameters)
```python
batch_size: int = 16
learning_rate: float = 1e-4
num_epochs: int = 10
warmup_steps: int = 10000
weight_decay: float = 0.01
gradient_clip_norm: float = 1.0
mlm_probability: float = 0.15

# Multi-task loss weights
mlm_loss_weight: float = 1.0
structure_loss_weight: float = 0.5
pair_loss_weight: float = 0.1

# AST settings
use_ast: bool = True
ast_target_activation: float = 0.4
ast_controller_kp: float = 0.01
ast_controller_ki: float = 0.001
```

---

## AST Integration Details

### How AST Works in Genesis RNA

1. **Forward Pass (All Samples)**:
   ```python
   outputs = model(input_ids, attention_mask)
   # outputs: mlm_logits, struct_logits, pair_logits
   ```

2. **Compute Per-Sample Losses**:
   ```python
   # Token-level MLM loss
   token_losses = F.cross_entropy(..., reduction='none')
   # Average over sequence → per-sample loss
   sample_losses = token_losses.mean(dim=1)  # [batch_size]
   ```

3. **AST Sample Selection**:
   ```python
   selected_idx = ast_selector.select_indices(sample_losses)
   # selected_idx: indices of "important" samples
   # Typically 40% of batch size
   ```

4. **Backprop on Selected Samples**:
   ```python
   loss = compute_loss(
       outputs[selected_idx],
       labels[selected_idx]
   )
   loss.backward()
   optimizer.step()
   ```

### Benefits of AST

- **Energy Efficiency**: ~60% reduction in FLOPs (with 40% activation)
- **Faster Training**: Skip low-importance samples
- **Similar Performance**: Focuses on hard examples
- **Adaptive**: PI controller maintains stable activation rate

---

## Data Sources (TODO)

### RNAcentral
- Comprehensive RNA sequence database
- Millions of sequences from multiple species
- Download: `scripts/download_rnacentral.sh`

### Rfam
- RNA families with structure annotations
- Alignment + consensus structures
- Use for structure prediction task

### lncRNAdb
- Long non-coding RNAs
- Functional annotations

### Custom Datasets
- mRNA stability data
- RNA-protein binding (RBP-Seq)
- Mutation effect data (DMS-seq, fitness landscapes)

---

## Fine-tuning for Downstream Tasks

After pretraining, the encoder can be fine-tuned for specific tasks:

### 1. Mutation Effect Prediction
- Add `MutationEffectHead`
- Train on fitness landscapes / ΔΔG data
- Use MSE or MAE loss

### 2. RNA-Protein Binding
- Add binary classification head
- Train on RBP binding data
- Use BCE loss

### 3. RNA Design
- Use as scoring function for RL/diffusion
- Generate sequences with desired properties
- Optimize via gradient-based search

### 4. mRNA Therapeutic Optimization
- Multi-objective optimization
  - Expression level
  - Stability (half-life)
  - Immunogenicity (low)
- Use encoder + custom heads

---

## Evaluation Benchmarks

### Pretraining Metrics
- MLM accuracy (nucleotide prediction)
- Structure prediction accuracy
- Base-pair F1 score

### Downstream Benchmarks
- **RNASolo**: Structure prediction benchmark
- **RNAcompete**: RNA-protein binding
- **Fitness landscapes**: Mutation effect prediction
- **mRNA stability**: UTR design evaluation

---

## Installation & Setup

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install AST library
pip install adaptive-sparse-training
```

### Quick Start
```bash
# Test with dummy data
python -m genesis_rna.train_pretrain \
  --use_dummy_data \
  --model_size small \
  --batch_size 8 \
  --num_epochs 2

# Pretrain on real data
python -m genesis_rna.train_pretrain \
  --data_path /path/to/rna_data \
  --config experiments/config_pretrain_base.yaml \
  --output_dir ./checkpoints
```

---

## Roadmap

### Phase 1: Foundation (Current)
- [x] Core architecture implementation
- [x] Multi-task pretraining
- [x] AST integration
- [x] Basic training pipeline
- [ ] Real data loaders (RNAcentral, Rfam)
- [ ] Evaluation metrics

### Phase 2: Optimization
- [ ] Rotary positional embeddings
- [ ] Flash attention integration
- [ ] Gradient checkpointing
- [ ] Distributed training (DDP)
- [ ] Mixed precision optimization

### Phase 3: Downstream Tasks
- [ ] Fine-tuning scripts
- [ ] Mutation effect prediction
- [ ] RNA-protein binding
- [ ] Structure prediction head
- [ ] Benchmark evaluations

### Phase 4: GENESIS-AI Integration
- [ ] Agent wrappers for each task
- [ ] REST API endpoints
- [ ] Interactive design interface
- [ ] Explainability tools (attention viz)
- [ ] Deployment pipeline

---

## Technical Notes

### Memory Optimization
- Use gradient checkpointing for long sequences (>1024)
- Mixed precision (FP16) reduces memory by ~50%
- AST reduces effective batch size → lower memory

### Computational Requirements
- **Small model**: ~10M params, 1 GPU (8GB VRAM)
- **Base model**: ~50M params, 1-2 GPUs (16GB VRAM)
- **Large model**: ~150M params, 4-8 GPUs (32GB VRAM)

### Training Time Estimates (Base Model)
- 1M sequences, 10 epochs
- Without AST: ~20 hours on 1x A100
- With AST (40%): ~12 hours on 1x A100

---

## References

1. **RiNALMo**: Chen et al., "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions"
2. **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
3. **AST**: Your adaptive-sparse-training library (energy-efficient training)
4. **RNA Structure**: Lorenz et al., "ViennaRNA Package 2.0"

---

## Contact & Contributing

For questions or contributions, see the main repository README.

**Key Design Principles**:
1. Modularity: Each component is self-contained
2. Efficiency: AST + mixed precision for low energy cost
3. Extensibility: Easy to add new tasks/heads
4. Reproducibility: All configs + seeds tracked

---

## Appendix: Key Files Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `config.py` | Model/training configuration | `GenesisRNAConfig`, `TrainingConfig` |
| `tokenization.py` | RNA sequence tokenization | `RNATokenizer`, `random_mask()` |
| `data.py` | Dataset classes | `RNAPretrainDataset`, `RNAPretrainSample` |
| `model.py` | Transformer encoder | `GenesisRNAModel`, `GenesisRNAEncoder` |
| `heads.py` | Task-specific heads | `MLMHead`, `StructureHead`, `PairHead` |
| `losses.py` | Loss functions | `MultiTaskLoss`, `mlm_loss()`, `pair_loss()` |
| `ast_wrapper.py` | AST integration | `ASTSampleSelector`, `PIController` |
| `train_pretrain.py` | Main training script | `main()`, `train_epoch()` |

