# CLAUDE.md - AI Assistant Guide for GENESI AI Repository

**Last Updated:** 2025-11-20
**Repository:** GENESI AI - Genesis RNA Foundation Model for Cancer Research
**Purpose:** Comprehensive guide for AI assistants working on this codebase

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Technologies](#core-technologies)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Testing Guidelines](#testing-guidelines)
7. [Key Files and Modules](#key-files-and-modules)
8. [Common Tasks](#common-tasks)
9. [Checkpoint Management](#checkpoint-management)
10. [Troubleshooting](#troubleshooting)
11. [Important Notes for AI Assistants](#important-notes-for-ai-assistants)

---

## Project Overview

### Mission
GENESI AI is an **AI-powered genomics research platform** focused on RNA analysis and **breast cancer cure research**. The project combines cutting-edge deep learning with domain expertise in RNA biology and cancer genomics.

### Key Innovation
**Genesis RNA** - A transformer-based RNA foundation model trained with **Adaptive Sparse Training (AST)** that achieves:
- 60% reduction in training FLOPs
- 40% faster research iterations
- Better predictions by focusing on difficult cancer variants
- Lower carbon footprint and costs

### Primary Applications
1. **BRCA1/2 Mutation Analysis** - Predict pathogenicity of genetic variants
2. **mRNA Therapeutics Design** - Optimize cancer treatment sequences
3. **Neoantigen Discovery** - Create personalized cancer vaccines
4. **Drug Target Identification** - Find therapeutic opportunities

### Model Architecture
- **Transformer-based** RNA language model inspired by RiNALMo
- **Multi-task learning**: Masked Language Modeling (MLM) + Secondary Structure + Base-Pair Prediction
- **Model sizes**: Small (10M), Base (35M), Large (150M) parameters
- **Training optimization**: Adaptive Sparse Training (AST) with PI controller
- **Production-ready**: Google Colab integration, clinical validation tools

---

## Repository Structure

```
genesi_ai/
├── genesis_rna/                 # Core RNA foundation model package
│   ├── genesis_rna/            # Main Python package
│   │   ├── __init__.py         # Package exports
│   │   ├── model.py            # Transformer architecture (339 lines)
│   │   ├── config.py           # Model & training configs (201 lines)
│   │   ├── tokenization.py     # RNA tokenizer (211 lines)
│   │   ├── heads.py            # Task-specific prediction heads (306 lines)
│   │   ├── data.py             # Dataset classes (358 lines)
│   │   ├── losses.py           # Loss functions incl. Focal Loss (462 lines)
│   │   ├── train_pretrain.py   # Training script (679 lines)
│   │   ├── ast_wrapper.py      # Adaptive Sparse Training (429 lines)
│   │   └── breast_cancer.py    # Cancer analysis tools (544 lines)
│   ├── scripts/                # Data generation scripts
│   ├── tests/                  # Unit tests (3 test files)
│   ├── requirements.txt        # Core dependencies
│   └── *.ipynb                 # Colab training notebooks
│
├── data/                       # Training data (gitignored)
│   ├── human_ncrna/           # Generated ncRNA sequences
│   ├── breast_cancer/         # Cancer-specific datasets
│   │   ├── brca_mutations/    # BRCA variant data
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── tcga/                  # TCGA RNA-seq data (optional)
│
├── checkpoints/               # Model checkpoints (organized by type)
│   ├── pretrained/           # Foundation models
│   │   ├── small/           # 10M parameters
│   │   ├── base/            # 35M parameters (default)
│   │   └── large/           # 150M parameters
│   ├── finetuned/           # Task-specific models
│   │   ├── brca_mutations/  # BRCA variant prediction
│   │   ├── structure/       # Structure prediction
│   │   └── pairing/         # Base-pair prediction
│   └── experiments/         # Experimental runs (dated folders)
│
├── configs/                  # Training configuration files (YAML)
│   └── train_t4_optimized.yaml  # T4 GPU optimized config
│
├── scripts/                  # Utility scripts
│   ├── download_brca_variants.py    # Fetch BRCA variants from ClinVar
│   ├── evaluate_cancer_model.py     # Comprehensive evaluation
│   ├── split_dataset.py             # Train/test splitting
│   ├── visualize_metrics.py         # Training visualization
│   └── download_tcga_data.py        # TCGA data downloader
│
├── examples/                 # Example usage scripts
│   ├── breast_cancer_demo.py
│   ├── colab_train_ncrna.py
│   └── train_with_ncrna.py
│
├── *.ipynb                   # Jupyter notebooks (4 total)
│   ├── breast_cancer_colab.ipynb              # Main cancer workflow
│   └── genesis_rna/genesis_rna_colab_training.ipynb
│
└── Documentation Files
    ├── README.md                    # Main project README
    ├── RESEARCH_WORKFLOW.md         # Complete research workflow guide
    ├── TRAINING_GUIDE.md            # Training instructions
    ├── BREAST_CANCER_RESEARCH.md    # Cancer research guide
    ├── BREAST_CANCER_QUICKSTART.md  # Quick start for cancer research
    ├── AST_CANCER_IMPACT.md         # AST impact analysis
    ├── IMPROVEMENTS.md              # Training improvements log
    ├── CHECKPOINT_FIX_NOTES.md      # Checkpoint troubleshooting
    ├── QUICKSTART_T4.md             # T4 GPU quick start
    ├── checkpoints/README.md        # Checkpoint management guide
    └── data/README.md               # Data setup instructions
```

---

## Core Technologies

### Deep Learning Stack
- **PyTorch 2.0+** - Primary deep learning framework
- **Transformers 4.30+** - HuggingFace library for transformer models
- **Mixed Precision Training (FP16)** - Essential for T4 GPU efficiency
- **Adaptive Sparse Training (AST)** - Custom implementation (no external package)

### Bioinformatics
- **BioPython 1.81+** - Biological sequence processing
- **RNA Tokenization** - Custom nucleotide encoding: A, C, G, U, N + special tokens
- **Vocabulary**: `[PAD]`, `[MASK]`, `[CLS]`, `[SEP]`, A, C, G, U, N (total: 9 tokens)

### Data Processing
- **NumPy 1.24+** - Numerical computing
- **Pandas 2.0+** - Data manipulation
- **PyYAML** - Configuration management

### Training Features
- **Multi-task Learning** - Joint training on MLM, structure prediction, and base-pairing
- **Focal Loss** - Handles severe class imbalance in RNA base-pairing
- **PI Controller** - Adaptive threshold control for AST sample selection
- **Cosine Annealing** - Learning rate scheduling for better convergence

### Development & Testing
- **pytest 7.4+** - Unit testing framework
- **Google Colab** - Cloud training environment (main development platform)
- **Google Drive** - Checkpoint storage and sharing

---

## Development Workflows

### 1. Local Development Setup

```bash
# Clone repository
cd genesi_ai

# Install core dependencies
cd genesis_rna
pip install -r requirements.txt

# Install cancer research dependencies (optional)
cd ..
pip install -r requirements_cancer.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Data Preparation

**Option A: Quick Start with Synthetic Data**
```bash
# Generate 5K synthetic ncRNA sequences
cd genesis_rna/scripts
python generate_sample_ncrna.py --output ../../data/human_ncrna --num_samples 5000
```

**Option B: Real BRCA Variant Data**
```bash
# Download BRCA variants from ClinVar API
python scripts/download_brca_variants.py --output data/breast_cancer/brca_mutations
```

**Option C: TCGA Cancer Data (requires authorization)**
```bash
python scripts/download_tcga_data.py --list --cancer_type BRCA
```

### 3. Training Workflow

**Quick Training Test (with dummy data)**
```bash
python -m genesis_rna.train_pretrain \
    --use_dummy_data \
    --model_size small \
    --num_epochs 2
```

**Full Training (with real data)**
```bash
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --data_path data/human_ncrna \
    --output_dir checkpoints/pretrained/base \
    --num_epochs 30
```

**Google Colab Training (recommended)**
```python
# Mount Google Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Train with checkpoint saving to Drive
!python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --data_path ./data/human_ncrna
```

### 4. Evaluation Workflow

```bash
# Split dataset
python scripts/split_dataset.py \
    --input data/breast_cancer/brca_mutations/train.jsonl \
    --train_out data/breast_cancer/train.jsonl \
    --test_out data/breast_cancer/test.jsonl \
    --test_split 0.2

# Evaluate model
python scripts/evaluate_cancer_model.py \
    --model checkpoints/pretrained/base/best_model.pt \
    --test_data data/breast_cancer/test.jsonl
```

### 5. Visualization Workflow

```bash
# Generate training metrics plots
python scripts/visualize_metrics.py \
    --metrics_file checkpoints/pretrained/base/training_metrics.csv \
    --output_dir plots/
```

**Generated plots:**
- `losses.png` - All loss components over time
- `accuracies.png` - MLM, structure, and pair metrics
- `lr_activation.png` - LR schedule and AST activation rate
- `summary.png` - Comprehensive training dashboard

---

## Code Conventions

### File Organization
- **Models**: Core model code in `genesis_rna/genesis_rna/*.py`
- **Scripts**: Utilities in `scripts/*.py` (data processing, evaluation)
- **Configs**: YAML files in `configs/`
- **Tests**: Unit tests in `genesis_rna/tests/`
- **Notebooks**: Research workflows as `.ipynb` files

### Naming Conventions

**Files:**
- Python modules: lowercase with underscores (`train_pretrain.py`)
- Classes: PascalCase (`GenesisRNAModel`)
- Functions: lowercase with underscores (`get_lr_scheduler`)
- Constants: UPPERCASE (`VOCAB_SIZE`)

**Checkpoints:**
```
{model_size}_{training_type}_{metric}_{value}_epoch{N}.pt

Examples:
- base_pretrain_valloss_2.63_epoch10.pt
- small_pretrain_best_model.pt
- large_finetune_structure_f1_0.85_epoch5.pt
```

### Code Style
- **Indentation**: 4 spaces (not tabs)
- **Line length**: Prefer <100 characters where reasonable
- **Docstrings**: Include for public classes and functions
- **Type hints**: Use where helpful (not strictly enforced)
- **Comments**: Explain "why" not "what"

### Configuration Management
- **YAML configs** for training parameters (`configs/train_t4_optimized.yaml`)
- **Config classes** in `genesis_rna/config.py` with dataclass-style structure
- **Command-line overrides** supported via argparse

### Import Style
```python
# Standard library
import os
import json
from typing import Optional, Dict, List

# Third-party
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel

# Local imports
from genesis_rna.config import GenesisRNAConfig
from genesis_rna.tokenization import RNATokenizer
```

---

## Testing Guidelines

### Test Location
All tests are in: `genesis_rna/tests/`

### Test Files
1. **`test_tokenization.py`** - RNA tokenizer tests
   - Token encoding/decoding
   - Special token handling ([PAD], [MASK], [CLS], [SEP])
   - Masking strategies (BERT-style: 80% [MASK], 10% random, 10% keep)

2. **`test_data.py`** - Dataset tests
   - Data loading
   - Batch processing
   - Label generation for multi-task learning

3. **`test_model_forward.py`** - Model architecture tests
   - `test_rna_embedding()`: Embedding layer
   - `test_transformer_block()`: Single transformer block
   - `test_encoder()`: Full encoder
   - `test_full_model()`: End-to-end forward pass
   - Tests with/without attention masks
   - Shape validation, NaN detection

### Running Tests

```bash
# Run all tests
cd genesis_rna
pytest tests/

# Run specific test file
pytest tests/test_model_forward.py

# Run with coverage
pytest tests/ --cov=genesis_rna --cov-report=html

# Verbose output
pytest tests/ -v
```

### Testing Philosophy
- **Unit tests** for core components (tokenizer, data, model)
- **Integration tests** for full forward pass
- **No tests for training scripts** (manual validation instead)
- Use **small model configs** for fast testing (`GenesisRNAConfigSmall`)
- **Fixtures** for reusable test components

### Writing New Tests
```python
import pytest
import torch
from genesis_rna import GenesisRNAModel, GenesisRNAConfig

def test_new_feature():
    """Test description"""
    # Setup
    config = GenesisRNAConfig.from_pretrained('small')
    model = GenesisRNAModel(config)

    # Execute
    input_ids = torch.randint(0, 9, (2, 10))
    outputs = model(input_ids)

    # Assert
    assert outputs.mlm_logits.shape == (2, 10, 9)
    assert not torch.isnan(outputs.mlm_logits).any()
```

---

## Key Files and Modules

### Core Model Files (genesis_rna/genesis_rna/)

#### `model.py` (339 lines) - Transformer Architecture
**Key Classes:**
- `RNAEmbedding`: Token + positional embeddings with layer norm
- `TransformerBlock`: Multi-head self-attention + feedforward network
- `GenesisRNAEncoder`: Multi-layer transformer encoder (4-12 layers)
- `GenesisRNAModel`: Complete model with multi-task heads

**Important Methods:**
- `forward()`: Main forward pass with optional masking
- `from_pretrained()`: Load model from checkpoint (handles dict/Config conversion)

**Line References:**
- RNAEmbedding: `model.py:20-40`
- TransformerBlock: `model.py:42-120`
- GenesisRNAEncoder: `model.py:122-200`
- GenesisRNAModel: `model.py:202-339`

#### `config.py` (201 lines) - Configuration Management
**Key Classes:**
- `GenesisRNAConfig`: Model architecture configuration
  - vocab_size: 9 (4 nucleotides + N + 4 special tokens)
  - d_model: 256-768, n_heads: 4-12, n_layers: 4-12
- `TrainingConfig`: Training hyperparameters
  - AST settings, loss weights, learning rate schedules

**Pre-defined Configs:**
- `GenesisRNAConfigSmall`: 10M params (d_model=256, n_layers=4, n_heads=4)
- `GenesisRNAConfigBase`: 35M params (d_model=512, n_layers=8, n_heads=8)
- `GenesisRNAConfigLarge`: 150M params (d_model=768, n_layers=12, n_heads=12)

**Important Functions:**
- `from_dict()`: Convert dict to Config object (critical for checkpoint loading)
- `to_dict()`: Serialize config for checkpointing

#### `tokenization.py` (211 lines) - RNA Tokenizer
**Key Class:** `RNATokenizer`

**Vocabulary (9 tokens):**
- Special tokens: `[PAD]` (0), `[MASK]` (1), `[CLS]` (2), `[SEP]` (3)
- Nucleotides: A (4), C (5), G (6), U (7), N (8)

**Important Methods:**
- `encode()`: Sequence → token IDs (with optional [CLS]/[SEP])
- `decode()`: Token IDs → sequence
- `random_mask()`: BERT-style masking (80% [MASK], 10% random, 10% keep)
- `batch_encode()`: Batch processing with padding

#### `heads.py` (306 lines) - Task-Specific Prediction Heads
**Key Classes:**
- `MLMHead`: Masked language modeling (nucleotide prediction)
- `StructureHead`: Secondary structure classification (STEM, LOOP, BULGE, HAIRPIN)
- `PairHead`: Base-pair prediction using bilinear scoring
- `MutationEffectHead`: Variant effect regression (for fine-tuning)

**Important:** PairHead uses bilinear scoring matrix for all-pairs prediction

#### `data.py` (358 lines) - Dataset Classes
**Key Classes:**
- `RNAPretrainSample`: Data structure for RNA sequences
- `RNAPretrainDataset`: PyTorch dataset for multi-task learning
  - Supports MLM, structure labels, pair matrices
  - Handles batching, padding, masking

**Important Methods:**
- `__getitem__()`: Returns sample with masked inputs and labels
- `collate_fn()`: Custom collation for variable-length sequences

#### `losses.py` (462 lines) - Loss Functions
**Key Classes:**
- `BinaryFocalLoss`: Handles severe class imbalance in pair prediction
  - alpha=0.75 (prioritize positive pairs)
  - gamma=2.0 (focus on hard examples)
- `MultiTaskLoss`: Combines MLM + structure + pairing losses

**Important Parameters:**
- `mlm_loss_weight`: 1.0
- `structure_loss_weight`: 0.8
- `pair_loss_weight`: 3.0 (aggressive for better structure prediction)

**Focal Loss Formula:**
```
FL(p_t) = -α(1-p_t)^γ * BCE(p_t)
```

#### `train_pretrain.py` (679 lines) - Training Script
**Key Features:**
- AST integration for sample selection
- Mixed precision training (FP16)
- Checkpointing, logging, metrics tracking
- Learning rate scheduling (cosine, linear)
- Google Colab compatibility

**Important Functions:**
- `get_lr_scheduler()`: Creates LR scheduler (cosine/linear/constant)
- `train_epoch()`: Single epoch training loop with AST
- `evaluate()`: Validation loop with comprehensive metrics

**Command-line Arguments:**
- `--config`: YAML config file (recommended)
- `--model_size`: small/base/large
- `--use_ast`: Enable Adaptive Sparse Training
- `--use_dummy_data`: Quick test with synthetic data
- `--resume_from`: Continue from checkpoint

#### `ast_wrapper.py` (429 lines) - Adaptive Sparse Training
**Key Classes:**
- `PIController`: Proportional-Integral controller for adaptive thresholds
- `ASTSampleSelector`: Selects important samples based on loss

**How AST Works:**
1. Compute loss for all samples
2. PI controller adjusts threshold to target activation rate (e.g., 40%)
3. Select samples with loss > threshold
4. Train only on selected samples (saves ~60% FLOPs)

**Configuration:**
- `ast_target_activation`: 0.4 (train on 40% of samples)
- `ast_controller_kp`: 0.01 (proportional gain)
- `ast_controller_ki`: 0.001 (integral gain)

#### `breast_cancer.py` (544 lines) - Cancer Analysis Tools
**Key Classes:**
- `BreastCancerAnalyzer`: Main interface for cancer variant analysis
  - `predict_variant_effect()`: Pathogenicity prediction
  - Supports 10 cancer genes (BRCA1, BRCA2, TP53, HER2, etc.)
- `TherapeuticmRNA`: mRNA therapeutic design
- `Neoantigen`: Personalized vaccine design

**Usage Example:**
```python
from genesis_rna.breast_cancer import BreastCancerAnalyzer

analyzer = BreastCancerAnalyzer('checkpoints/finetuned/brca/best_model.pt')
prediction = analyzer.predict_variant_effect(
    gene='BRCA1',
    wild_type_rna=wt_sequence,
    mutant_rna=mut_sequence
)
print(f"Pathogenicity: {prediction.pathogenicity_score:.3f}")
print(f"Interpretation: {prediction.interpretation}")
```

### Configuration Files

#### `configs/train_t4_optimized.yaml`
**T4 GPU optimized configuration (16GB VRAM)**

**Key Settings:**
```yaml
model:
  d_model: 256
  n_heads: 4
  n_layers: 4
  max_len: 512

training:
  batch_size: 32          # Optimized for T4
  learning_rate: 0.0002
  num_epochs: 30          # AST makes this efficient
  fp16: true              # Essential for T4 Tensor Cores

  # Multi-task loss weights
  mlm_loss_weight: 1.0
  structure_loss_weight: 0.8
  pair_loss_weight: 3.0   # Aggressive for structure prediction

  # Focal loss for class imbalance
  use_focal_loss_for_pairs: true
  focal_alpha: 0.75
  focal_gamma: 2.0

  # AST settings
  use_ast: true
  ast_target_activation: 0.4  # Train on 40% of samples

  # Learning rate scheduling
  lr_scheduler_type: "cosine"
  min_lr_ratio: 0.05      # Decay to 5% of peak LR
```

### Utility Scripts

#### `scripts/download_brca_variants.py`
Fetches BRCA1/2 variants from NCBI ClinVar API
- Creates synthetic datasets for testing
- Generates training data in JSONL format

#### `scripts/evaluate_cancer_model.py`
Comprehensive evaluation framework
- Clinical metrics: sensitivity, specificity, PPV, NPV
- AUC-ROC, AUC-PR calculations
- VUS (Variant of Uncertain Significance) analysis

#### `scripts/visualize_metrics.py`
Training metrics visualization
- Loss curves, accuracy plots
- Generates 4 types of plots (losses, accuracies, LR/activation, summary)

---

## Common Tasks

### Task 1: Adding a New Model Feature

**Example: Adding a new prediction head**

1. **Define the head in `heads.py`:**
```python
class NewTaskHead(nn.Module):
    """New task-specific prediction head"""
    def __init__(self, d_model, num_outputs):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.output = nn.Linear(d_model, num_outputs)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        return self.output(x)
```

2. **Add to model in `model.py`:**
```python
class GenesisRNAModel(nn.Module):
    def __init__(self, config):
        # ... existing code ...
        self.new_task_head = NewTaskHead(config.d_model, config.new_task_outputs)

    def forward(self, input_ids, ...):
        # ... existing code ...
        new_task_logits = self.new_task_head(hidden_states)
        return outputs
```

3. **Add loss computation in `losses.py`:**
```python
class MultiTaskLoss(nn.Module):
    def forward(self, outputs, labels):
        # ... existing losses ...
        new_task_loss = self.new_task_criterion(outputs.new_task_logits, labels.new_task)
        total_loss += self.new_task_weight * new_task_loss
```

4. **Write tests in `tests/test_model_forward.py`:**
```python
def test_new_task_head():
    config = GenesisRNAConfig.from_pretrained('small')
    model = GenesisRNAModel(config)
    input_ids = torch.randint(0, 9, (2, 10))
    outputs = model(input_ids)
    assert outputs.new_task_logits.shape == (2, 10, config.new_task_outputs)
```

### Task 2: Modifying Training Configuration

**Example: Adding a new hyperparameter**

1. **Update `config.py`:**
```python
@dataclass
class TrainingConfig:
    # ... existing parameters ...
    new_parameter: float = 0.5
```

2. **Update YAML config (`configs/train_t4_optimized.yaml`):**
```yaml
training:
  # ... existing settings ...
  new_parameter: 0.5
```

3. **Use in training script (`train_pretrain.py`):**
```python
def train_epoch(model, dataloader, optimizer, config):
    new_param_value = config.new_parameter
    # Use in training loop
```

### Task 3: Adding Data Processing Script

**Example: New data downloader**

1. **Create script in `scripts/download_new_data.py`:**
```python
#!/usr/bin/env python3
"""Download new RNA dataset"""
import argparse
import requests
from pathlib import Path

def download_data(output_dir, num_samples=1000):
    """Download data from source"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download logic here
    print(f"Downloaded {num_samples} samples to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()

    download_data(args.output, args.num_samples)
```

2. **Make executable:**
```bash
chmod +x scripts/download_new_data.py
```

3. **Document in README or relevant guide**

### Task 4: Debugging Training Issues

**Common issue: Pair F1 score drops to 0%**

1. **Check loss weights in config:**
```yaml
training:
  pair_loss_weight: 3.0  # Should be significant (not 0.1)
```

2. **Verify focal loss is enabled:**
```yaml
training:
  use_focal_loss_for_pairs: true
  focal_alpha: 0.75
  focal_gamma: 2.0
```

3. **Check metrics CSV for trends:**
```bash
python scripts/visualize_metrics.py \
    --metrics_file checkpoints/training_metrics.csv \
    --output_dir debug_plots/
```

4. **Adjust AST activation rate (train on more samples):**
```yaml
training:
  ast_target_activation: 0.5  # Increase from 0.4
```

See `IMPROVEMENTS.md` for detailed troubleshooting guide.

### Task 5: Fine-tuning for New Task

**Example: Fine-tune for structure prediction**

1. **Prepare task-specific data:**
```bash
# Create JSONL with structure labels
python scripts/prepare_structure_data.py \
    --input data/structures.csv \
    --output data/structure/train.jsonl
```

2. **Create fine-tuning script (or modify `train_pretrain.py`):**
```python
# Load pre-trained model
checkpoint = torch.load('checkpoints/pretrained/base/best_model.pt')
model = GenesisRNAModel.from_pretrained('checkpoints/pretrained/base/best_model.pt')

# Freeze encoder (optional)
for param in model.encoder.parameters():
    param.requires_grad = False

# Train only task head
optimizer = AdamW(model.structure_head.parameters(), lr=1e-5)
```

3. **Train with lower learning rate:**
```bash
python -m genesis_rna.train_finetune \
    --task structure \
    --pretrained_model checkpoints/pretrained/base/best_model.pt \
    --train_data data/structure/train.jsonl \
    --learning_rate 1e-5 \
    --num_epochs 10
```

---

## Checkpoint Management

### Checkpoint Structure
Each `.pt` file contains:
```python
{
    'epoch': int,                    # Training epoch number
    'step': int,                     # Global training step
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'config': {                      # Training configuration
        'model': {...},              # Model config (dict or Config object)
        'training': {...}            # Training config
    }
}
```

### Loading Checkpoints

**Recommended Method (handles config conversion):**
```python
from genesis_rna import GenesisRNAModel

model = GenesisRNAModel.from_pretrained(
    'checkpoints/pretrained/base/best_model.pt',
    device='cuda'
)
```

**Manual Loading:**
```python
import torch
from genesis_rna import GenesisRNAModel, GenesisRNAConfig

checkpoint = torch.load('checkpoints/pretrained/base/best_model.pt')

# Handle dict or Config object (important!)
model_config_dict = checkpoint['config']['model']
if isinstance(model_config_dict, dict):
    model_config = GenesisRNAConfig.from_dict(model_config_dict)
else:
    model_config = model_config_dict

model = GenesisRNAModel(model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Resuming Training
```python
checkpoint = torch.load('checkpoints/checkpoint_epoch_5.pt')

# Restore model
model_config = GenesisRNAConfig.from_dict(checkpoint['config']['model'])
model = GenesisRNAModel(model_config)
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer
optimizer = AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue from next epoch
start_epoch = checkpoint['epoch'] + 1
```

### Google Drive Integration

**Standard Colab checkpoint path:**
```
/content/drive/MyDrive/genesis_rna_checkpoints/
```

**Save to Drive during training:**
```bash
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --data_path ./data/human_ncrna
```

**Resume from Drive checkpoint:**
```bash
python -m genesis_rna.train_pretrain \
    --resume_from /content/drive/MyDrive/genesis_rna_checkpoints/checkpoint_epoch_5.pt \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints
```

### Checkpoint Naming Convention
```
{model_size}_{training_type}_{metric}_{value}_epoch{N}.pt

Examples:
- base_pretrain_valloss_2.63_epoch10.pt
- small_pretrain_best_model.pt
- large_finetune_structure_f1_0.85_epoch5.pt
```

### Best Practices
1. **Save regularly**: Every 5 epochs for long runs, every epoch for short runs
2. **Keep best model**: Always save based on validation loss
3. **Document metadata**: Create `metadata.json` with training details
4. **Clean up**: Delete intermediate checkpoints to save space
5. **Version control**: Do NOT commit `.pt` files to git (too large)

See `checkpoints/README.md` for comprehensive guide.

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes during forward/backward pass

**Solutions:**
```bash
# Reduce batch size
--batch_size 16  # or even 8

# Use gradient accumulation
--gradient_accumulation_steps 4

# Ensure FP16 is enabled
--fp16

# Reduce sequence length
--max_len 256  # default is 512
```

#### 2. Pair F1 Score Drops to 0%

**Symptoms:**
- Pair F1 starts at ~0.2% and drops to 0.00% by epoch 4
- Pair precision/recall both near zero

**Root Cause:** Severe class imbalance (most RNA positions don't form pairs)

**Solutions:**
1. Verify focal loss is enabled:
```yaml
training:
  use_focal_loss_for_pairs: true
  focal_alpha: 0.75
  focal_gamma: 2.0
```

2. Increase pair loss weight:
```yaml
training:
  pair_loss_weight: 3.0  # Not 0.1!
```

3. Adjust focal loss parameters:
```yaml
training:
  focal_alpha: 0.85  # Increase to prioritize pairs more
```

4. Reduce AST activation (train on more samples):
```yaml
training:
  ast_target_activation: 0.5  # From 0.4
```

See `IMPROVEMENTS.md` for detailed analysis.

#### 3. Checkpoint Loading Errors

**Symptoms:**
- `TypeError: GenesisRNAModel() expects Config, got dict`
- `AttributeError: 'dict' object has no attribute 'vocab_size'`

**Root Cause:** Config stored as dict in checkpoint, not Config object

**Solution:** Use `from_pretrained()` or manual conversion:
```python
# Option 1: Use from_pretrained (recommended)
model = GenesisRNAModel.from_pretrained('checkpoint.pt', device='cuda')

# Option 2: Manual conversion
checkpoint = torch.load('checkpoint.pt')
config_dict = checkpoint['config']['model']
config = GenesisRNAConfig.from_dict(config_dict)
model = GenesisRNAModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

See `CHECKPOINT_FIX_NOTES.md` for more details.

#### 4. Slow Training on T4 GPU

**Symptoms:**
- Training slower than expected
- GPU utilization < 80% (check with `nvidia-smi`)

**Solutions:**
1. Ensure FP16 is enabled (critical for T4 Tensor Cores):
```yaml
training:
  fp16: true
  mixed_precision: true
```

2. Increase batch size:
```yaml
training:
  batch_size: 48  # or 64 if memory allows
```

3. Optimize data loading:
```yaml
training:
  num_workers: 2
  prefetch_factor: 2
```

4. Use T4-optimized config:
```bash
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml
```

#### 5. Loss Explodes (NaN)

**Symptoms:**
- Loss becomes NaN after a few iterations
- Gradients explode

**Solutions:**
1. Reduce learning rate:
```yaml
training:
  learning_rate: 0.0001  # From 0.0002
```

2. Increase warmup steps:
```yaml
training:
  warmup_steps: 1000  # From 500
```

3. Verify gradient clipping is enabled:
```yaml
training:
  gradient_clip_norm: 1.0  # Should be set
```

4. Check for bugs in custom loss functions

#### 6. DataLoader Worker Warnings

**Symptoms:**
```
UserWarning: This DataLoader will create 4 worker processes in total.
Our suggested max number of worker in current system is 2.
```

**Solution:** Reduce worker count:
```yaml
training:
  num_workers: 2  # Not 4
```

#### 7. ImportError or ModuleNotFoundError

**Symptoms:**
- `ModuleNotFoundError: No module named 'genesis_rna'`
- Import errors when running scripts

**Solutions:**
1. Install in editable mode:
```bash
cd genesis_rna
pip install -e .
```

2. Or run as module:
```bash
python -m genesis_rna.train_pretrain  # Not: python train_pretrain.py
```

3. Check PYTHONPATH includes repo root:
```bash
export PYTHONPATH=/path/to/genesi_ai:$PYTHONPATH
```

### Getting Help

1. **Check documentation:**
   - `README.md` - Project overview
   - `TRAINING_GUIDE.md` - Training instructions
   - `IMPROVEMENTS.md` - Known issues and fixes
   - `checkpoints/README.md` - Checkpoint issues

2. **Check logs:**
   - Training metrics CSV: `checkpoints/training_metrics.csv`
   - Console output during training
   - Generated plots from `visualize_metrics.py`

3. **Verify setup:**
   - GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Dependencies: `pip list | grep torch`
   - Data files: `ls data/human_ncrna/`

---

## Important Notes for AI Assistants

### Critical Points

1. **Config Type Handling**
   - Checkpoints may store config as `dict` or `GenesisRNAConfig` object
   - Always use `GenesisRNAConfig.from_dict()` when loading from dict
   - The `from_pretrained()` method handles this automatically
   - This is a common source of bugs - see `CHECKPOINT_FIX_NOTES.md`

2. **Multi-task Learning**
   - Model trains on 3 tasks simultaneously: MLM, structure prediction, base-pairing
   - Loss weights are critical: `mlm_loss_weight=1.0, structure_loss_weight=0.8, pair_loss_weight=3.0`
   - Focal loss is essential for pair prediction (severe class imbalance)

3. **Adaptive Sparse Training (AST)**
   - AST is a custom implementation (no external package)
   - Located in `ast_wrapper.py`
   - Uses PI controller to dynamically select ~40% of samples
   - Must track activation rate (should hover around target, e.g., 0.4)

4. **Google Colab Environment**
   - Primary development/training platform
   - Always mount Google Drive for checkpoint persistence
   - Standard checkpoint path: `/content/drive/MyDrive/genesis_rna_checkpoints/`
   - Free T4 GPU is sufficient for training

5. **T4 GPU Optimization**
   - FP16 is **essential** for T4 Tensor Cores (not optional)
   - Batch size 32 is optimal for 16GB VRAM
   - Use `configs/train_t4_optimized.yaml`

6. **RNA Tokenization**
   - Vocabulary size is 9 (not 4): `[PAD], [MASK], [CLS], [SEP], A, C, G, U, N`
   - BERT-style masking: 80% [MASK], 10% random, 10% keep
   - U (not T) - this is RNA, not DNA

7. **Clinical Focus**
   - Project is specifically for **breast cancer cure research**
   - BRCA1/2 mutations are primary use case
   - Clinical metrics (sensitivity, specificity, PPV, NPV) are critical
   - VUS (Variant of Uncertain Significance) reclassification is a key goal

### Code Modification Guidelines

**When modifying core model code:**
1. Always write tests in `genesis_rna/tests/`
2. Ensure backward compatibility with existing checkpoints
3. Update relevant documentation (README, TRAINING_GUIDE, etc.)
4. Test with small model first (`GenesisRNAConfigSmall`)
5. Verify shapes and check for NaN in outputs

**When adding new features:**
1. Follow existing code structure and naming conventions
2. Add configuration parameters to `config.py`
3. Update YAML configs if needed
4. Document in docstrings and comments
5. Consider impact on checkpoint loading

**When fixing bugs:**
1. Check if issue is documented in `IMPROVEMENTS.md` or `CHECKPOINT_FIX_NOTES.md`
2. Write a test that reproduces the bug
3. Verify fix doesn't break existing tests
4. Update documentation to prevent recurrence

### Understanding Project Context

**This is a research project with clinical applications:**
- Code quality is important but not production-grade
- Focus is on rapid iteration and experimentation
- Google Colab is the primary environment (not local machines)
- Documentation is extensive but may have some inconsistencies
- Some features are planned but not yet implemented (e.g., full fine-tuning scripts)

**Key stakeholders:**
- Researchers using the model for cancer variant analysis
- Data scientists training and evaluating models
- Clinicians potentially using predictions in genetic counseling

**Project goals:**
1. Build effective RNA foundation model
2. Predict pathogenicity of cancer mutations (especially BRCA1/2)
3. Enable faster, more accessible cancer research (via AST efficiency)
4. Provide tools for therapeutic design and neoantigen discovery
5. Publish results and open-source the platform

### Repository Maintenance

**Files to NOT modify without good reason:**
- `genesis_rna/__init__.py` - Package exports (keep minimal)
- `genesis_rna/tokenization.py` - Stable, well-tested
- Test files - Only add, don't break existing tests
- `configs/train_t4_optimized.yaml` - Carefully tuned

**Files frequently modified:**
- `train_pretrain.py` - Training improvements
- `losses.py` - Loss function tuning
- `config.py` - Adding new parameters
- Documentation files - Keep up to date

**Files to create for new features:**
- New scripts in `scripts/`
- New notebooks for workflows
- New configs in `configs/`
- Documentation in root (e.g., `NEW_FEATURE_GUIDE.md`)

### Expected Metrics

**Pre-training (after 30 epochs with T4 config):**
- MLM Accuracy: >35%
- Structure Accuracy: >85%
- Pair F1: >2% (with focal loss)
- Pair Precision: >5%
- Pair Recall: >10%
- Validation Loss: <1.8

**Fine-tuning (BRCA variant prediction):**
- AUC-ROC: >0.85 (goal: >0.90 for clinical use)
- Sensitivity: >0.90 (critical - minimize false negatives)
- Specificity: >0.85
- VUS Reclassification: >30% with confidence

**Training efficiency (with AST):**
- Activation Rate: ~40% (should hover around target)
- FLOPs Reduction: ~60%
- Training Time: 2-4 hours on T4 for 30 epochs (small/base models)

### Common Patterns

**Running training:**
```bash
# Always prefer config files
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir checkpoints/pretrained/base

# Not: python train_pretrain.py (import issues)
```

**Loading models:**
```python
# Always use from_pretrained
model = GenesisRNAModel.from_pretrained('checkpoint.pt', device='cuda')

# Not: manual loading (unless you need optimizer state)
```

**Adding dependencies:**
```bash
# Add to genesis_rna/requirements.txt
# Or to requirements_cancer.txt for cancer-specific tools
# Then: pip install -r requirements.txt
```

**Handling data:**
```bash
# Data directory is gitignored
# Always generate or download data locally
# Document data sources in data/README.md
```

### Questions to Ask

When implementing new features, consider:
1. Does this require changes to checkpoint format? (backward compatibility)
2. Will this work on Google Colab? (primary environment)
3. Does this need new dependencies? (update requirements.txt)
4. Should this be configurable? (add to config.py)
5. Does this affect multi-task learning? (check all 3 tasks)
6. Is this compatible with AST? (sample selection may affect it)
7. Does this need documentation? (likely yes)

### Getting Started (For AI Assistants)

When asked to work on this repository:

1. **First, understand the task:**
   - Is it a bug fix, new feature, or documentation update?
   - Which files are involved?
   - Are there existing examples or similar code?

2. **Check existing documentation:**
   - Is this task documented in any of the guides?
   - Are there known issues or solutions in `IMPROVEMENTS.md`?

3. **Locate relevant code:**
   - Core model: `genesis_rna/genesis_rna/*.py`
   - Scripts: `scripts/*.py`
   - Configs: `configs/*.yaml`
   - Tests: `genesis_rna/tests/*.py`

4. **Make changes following conventions:**
   - Follow existing code style
   - Add tests if modifying model code
   - Update documentation if changing APIs
   - Test with small model/data first

5. **Verify the change:**
   - Run relevant tests: `pytest genesis_rna/tests/`
   - Check with dummy data: `--use_dummy_data`
   - Review checkpoint loading if config changes

---

## Summary

GENESI AI is a production-ready research platform for RNA-based cancer cure research. The codebase is well-structured with:

- **Clear separation**: Core model, utilities, configs, tests, documentation
- **Extensive documentation**: 9+ markdown files covering all aspects
- **Google Colab focus**: Optimized for accessible cloud training
- **Clinical applications**: Real-world breast cancer variant analysis
- **Research innovations**: Adaptive Sparse Training for efficiency

**Key strengths:**
- Multi-task learning with focal loss for imbalanced data
- Comprehensive checkpoint management with Google Drive integration
- Well-tested core components (tokenizer, data, model)
- Extensive documentation and troubleshooting guides

**Areas requiring care:**
- Config type handling (dict vs Config object)
- Checkpoint backward compatibility
- Multi-task loss weight tuning
- AST activation rate monitoring

**Primary workflow:**
1. Generate/download data → 2. Train with T4 config → 3. Evaluate on cancer variants → 4. Visualize metrics → 5. Fine-tune for specific tasks

When in doubt, refer to:
- `README.md` for overview
- `RESEARCH_WORKFLOW.md` for complete research workflow
- `IMPROVEMENTS.md` for known issues and solutions
- `checkpoints/README.md` for checkpoint management
- This file (`CLAUDE.md`) for development conventions

---

**Document Version:** 1.0
**Last Updated:** 2025-11-20
**Maintained By:** Project maintainers
**For Questions:** See documentation files or create GitHub issue
