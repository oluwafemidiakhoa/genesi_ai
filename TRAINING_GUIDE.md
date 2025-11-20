# Genesis RNA Training Guide

This guide explains how to train the Genesis RNA model with real ncRNA data.

## Quick Start

### 1. Preprocess ncRNA Data (Already Done! ✅)

You've already processed the data:
- **Input**: `Homo_sapiens.GRCh38.ncrna.fa` (203,749 sequences)
- **Output**: `../data/human_ncrna/sequences.pkl` (43,050 valid sequences)
- **Filtered**: 160,699 sequences (outside length range 50-512)

### 2. Train the Model

Now you can train the model with your processed data!

#### Option A: Using the Simple Training Script

```bash
# From the repository root
python examples/train_with_ncrna.py \
    --data_path ../data/human_ncrna/sequences.pkl \
    --num_epochs 10 \
    --batch_size 16
```

#### Option B: Using the Module Directly

```bash
python -m genesis_rna.train_pretrain \
    --data_path ../data/human_ncrna/sequences.pkl \
    --output_dir checkpoints/pretrained/base \
    --model_size base \
    --batch_size 16 \
    --num_epochs 10 \
    --use_ast
```

#### Option C: Google Colab

```python
# In a Colab notebook
%cd /content/genesi_ai
!python examples/colab_train_ncrna.py
```

### 3. Training Options

#### Fast Training (for testing)
```bash
python examples/train_with_ncrna.py \
    --data_path ../data/human_ncrna/sequences.pkl \
    --max_samples 5000 \
    --num_epochs 3 \
    --batch_size 32
```

#### Full Training (recommended)
```bash
python examples/train_with_ncrna.py \
    --data_path ../data/human_ncrna/sequences.pkl \
    --num_epochs 10 \
    --batch_size 16 \
    --use_ast
```

#### High-Quality Training
```bash
python examples/train_with_ncrna.py \
    --data_path ../data/human_ncrna/sequences.pkl \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --use_ast
```

## What the Training Does

The training script will:

1. ✅ Load your processed ncRNA sequences (43,050 samples)
2. ✅ Split into train (90%) and validation (10%)
3. ✅ Train with multi-task learning:
   - Masked language modeling (MLM)
   - Secondary structure prediction
   - Base-pair prediction
4. ✅ Use Adaptive Sparse Training (AST) to save ~60% computation
5. ✅ Save the best model to `checkpoints/pretrained/base/best_model.pt`

## Expected Output

During training, you'll see:

```
Loading data from ../data/human_ncrna/sequences.pkl...
Loaded 43,050 sequences
Split into 38,745 train and 4,305 validation samples

Using device: cuda
Model parameters: 42,356,736

======================================================================
GENESIS RNA TRAINING - OPTIMIZED WITH AST
======================================================================
Epochs: 10
Total training steps: 24,216

OPTIMIZATION STRATEGY:
  • AST Target Activation: 40%
    → Training on hardest 40% of samples
    → Saving ~60% computation!
======================================================================

Epoch 0: 100%|████████████| 2422/2422 [12:34<00:00, 3.21it/s]
  loss: 2.3456
  mlm_loss: 1.8234
  structure_loss: 0.3122
  pair_loss: 0.2100
  activation_rate: 0.4023 (AST: 40.2% samples trained)

Validation:
  loss: 2.1234
  mlm_accuracy: 0.6789
  structure_accuracy: 0.5432
  pair_f1: 0.4321

✅ Saved checkpoint to checkpoints/pretrained/base/best_model.pt
```

## After Training

Once training is complete, you can use the model for:

### 1. Breast Cancer Analysis

```python
from genesis_rna.breast_cancer import BreastCancerAnalyzer

analyzer = BreastCancerAnalyzer('checkpoints/pretrained/base/best_model.pt')

# Predict variant pathogenicity
prediction = analyzer.predict_variant_effect(
    gene='BRCA1',
    wt_sequence='AUGGGCUUCCGU...',
    mut_sequence='AUGGGCUUCCGU...'
)

print(f"Pathogenicity: {prediction.pathogenicity_score:.3f}")
print(f"Interpretation: {prediction.interpretation}")
```

### 2. Run the Demo

```bash
python examples/breast_cancer_analysis.py
```

### 3. Fine-tune for Specific Tasks

```bash
python -m genesis_rna.train_finetune \
    --pretrained_model checkpoints/pretrained/base/best_model.pt \
    --task structure_prediction \
    --data_path data/rfam/structured.pkl
```

## Troubleshooting

### Error: "Data file not found"

Make sure the data path is correct. If you preprocessed with:
```bash
python genesis_rna/scripts/preprocess_rna.py \
    --input ../Homo_sapiens.GRCh38.ncrna.fa \
    --output ../data/human_ncrna
```

Then your data is at: `../data/human_ncrna/sequences.pkl`

### Error: "CUDA out of memory"

Reduce batch size:
```bash
python examples/train_with_ncrna.py \
    --batch_size 8  # or 4
```

### Error: "Model file not found"

The model is only created AFTER successful training. Check:
1. Training completed without errors
2. You're using the correct path: `checkpoints/pretrained/base/best_model.pt`

## Advanced Configuration

### Custom Training Config

Create a YAML config file:

```yaml
# config/my_training.yaml
model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_len: 512

training:
  batch_size: 16
  num_epochs: 10
  learning_rate: 1e-4
  use_ast: true
  ast_target_activation: 0.4
  mlm_loss_weight: 1.0
  structure_loss_weight: 0.5
  pair_loss_weight: 2.0
```

Then train with:
```bash
python -m genesis_rna.train_pretrain \
    --config config/my_training.yaml \
    --data_path ../data/human_ncrna/sequences.pkl
```

## Performance Expectations

### Training Time (Tesla T4 GPU)

- **Fast (5K samples, 3 epochs)**: ~30 minutes
- **Standard (43K samples, 10 epochs)**: ~3-4 hours
- **High-quality (43K samples, 20 epochs)**: ~6-8 hours

### Expected Metrics

After 10 epochs with AST:
- **MLM Accuracy**: ~70-75%
- **Structure Accuracy**: ~55-60%
- **Pair F1**: ~40-45%

## Next Steps

1. ✅ Train the model (you're here!)
2. 📊 Evaluate on breast cancer variants
3. 🧬 Fine-tune for specific applications
4. 🚀 Deploy for research use

## Need Help?

- Check the full documentation: `docs/`
- See examples: `examples/`
- Read the research guide: `BREAST_CANCER_RESEARCH.md`

Happy training! 🧬🎗️
