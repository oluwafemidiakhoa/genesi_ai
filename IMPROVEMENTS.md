# Genesis RNA Training Improvements

## Overview

This document describes the major improvements made to the Genesis RNA training pipeline to address pair prediction issues and optimize for T4 GPU training.

## Problems Addressed

1. **Pair Prediction Failing** - F1 score dropped to 0% by epoch 4
2. **Class Imbalance** - Most RNA positions don't form pairs, causing severe imbalance
3. **DataLoader Warnings** - Worker count exceeded system recommendations
4. **Deprecated APIs** - Using old PyTorch AMP API
5. **Suboptimal LR Scheduling** - Linear decay not ideal for convergence
6. **No Metrics Tracking** - Difficult to analyze training progress

## Key Improvements

### 1. Binary Focal Loss for Pair Prediction ✨

**Problem**: Pair prediction uses binary cross-entropy, which struggles with severe class imbalance (most positions don't pair).

**Solution**: Implemented Binary Focal Loss that:
- Focuses learning on hard-to-classify examples
- Down-weights easy negative examples (non-pairing positions)
- Up-weights actual base pairs with `alpha=0.75`
- Uses `gamma=2.0` to focus on hard examples

**Code**: `genesis_rna/losses.py:BinaryFocalLoss`

**Expected Impact**: Pair F1 score should stay >1% and improve over training

### 2. Cosine Annealing LR Scheduler 📉

**Problem**: Linear decay can be suboptimal, especially near convergence.

**Solution**: Implemented cosine annealing with warmup:
- Smooth learning rate decay following cosine curve
- Decays from peak LR to 10% of peak (configurable)
- Better convergence properties than linear decay
- Warmup period for stability

**Code**: `genesis_rna/train_pretrain.py:get_lr_scheduler()`

**Configuration**:
```yaml
lr_scheduler_type: "cosine"  # or "linear", "constant"
min_lr_ratio: 0.1  # Minimum LR = 10% of peak
```

### 3. Comprehensive Metrics Logging 📊

**Problem**: No systematic way to track and visualize training progress.

**Solution**: Added CSV logger and visualization tools:
- Logs all metrics per epoch to CSV
- Tracks: losses, accuracies, LR, activation rate
- Visualization script creates 4 types of plots

**Files**:
- Logger: `genesis_rna/train_pretrain.py:MetricsLogger`
- Visualizer: `scripts/visualize_metrics.py`

**Usage**:
```bash
# After training, generate visualizations
python scripts/visualize_metrics.py \
    --metrics_file /path/to/training_metrics.csv \
    --output_dir /path/to/plots
```

**Generated Plots**:
1. `losses.png` - All loss components over time
2. `accuracies.png` - MLM, structure, and pair metrics
3. `lr_activation.png` - LR schedule and AST activation rate
4. `summary.png` - Comprehensive training dashboard

### 4. T4 GPU Optimization ⚡

**Problem**: Training not optimized for T4's 16GB VRAM and Tensor Cores.

**Solution**: Created T4-optimized configuration:
- Batch size: 32 (increased from 16 for better utilization)
- FP16 enabled (essential for T4 Tensor Cores)
- Optimized worker count: 2
- Reduced warmup: 500 steps
- Higher learning rate: 2e-4 (for larger effective batch)

**File**: `configs/train_t4_optimized.yaml`

**Usage**:
```bash
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --data_path ./data/rnacentral_processed \
    --num_epochs 10
```

### 5. Bug Fixes 🐛

- Fixed DataLoader worker count (4 → 2)
- Updated deprecated `torch.cuda.amp` → `torch.amp`
- Updated `GradScaler()` → `GradScaler('cuda')`
- Updated `autocast()` → `autocast('cuda')`

## Configuration Changes

### New Training Config Parameters

```python
# Learning rate scheduling
lr_scheduler_type: str = "cosine"  # 'linear', 'cosine', 'constant'
min_lr_ratio: float = 0.1  # Minimum LR as ratio of peak LR

# Focal loss settings for pair prediction
use_focal_loss_for_pairs: bool = True
focal_alpha: float = 0.75  # Weight for positive pairs
focal_gamma: float = 2.0  # Focusing parameter
```

### Updated Loss Weights

```python
mlm_loss_weight: 1.0          # Unchanged
structure_loss_weight: 0.8    # Increased from 0.5
pair_loss_weight: 1.5         # Increased from 0.1 (15x!)
```

## How to Use

### Option 1: Use T4-Optimized Config (Recommended)

```bash
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --data_path ./data/rnacentral_processed
```

### Option 2: Command-Line Arguments

```bash
python -m genesis_rna.train_pretrain \
    --model_size small \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 0.0002 \
    --ast_target_activation 0.4 \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --data_path ./data/rnacentral_processed \
    --use_ast
```

### Monitoring Training

1. **During Training**: Watch the progress bar for:
   - Loss trends
   - Activation rate (should hover around 40%)
   - Learning rate

2. **After Each Epoch**: Check printed metrics:
   - Pair F1, precision, recall
   - Structure and MLM accuracy
   - Individual loss components

3. **After Training**: Generate visualizations:
   ```bash
   python scripts/visualize_metrics.py \
       --metrics_file /content/drive/MyDrive/genesis_rna_checkpoints/training_metrics.csv \
       --output_dir /content/drive/MyDrive/genesis_rna_plots
   ```

## Expected Results

### Before Improvements
- Pair F1: 0.27% → **0.00%** (collapsed by epoch 4)
- Val Loss: 3.11 → 2.02
- Structure Accuracy: 11.83% → 80.63%

### After Improvements (Expected)
- Pair F1: **Should maintain >1% and improve**
- Pair Precision: **>5%** by end of training
- Pair Recall: **>10%** by end of training
- Val Loss: **<1.8** with better convergence
- Structure Accuracy: **>85%** maintained
- MLM Accuracy: **>35%** by end

## What Each Improvement Does

| Improvement | Impact | Expected Gain |
|------------|--------|---------------|
| Focal Loss | Handles class imbalance | +5-10% pair F1 |
| Higher pair weight (1.5x) | More gradient signal | +3-5% pair F1 |
| Cosine LR schedule | Better convergence | -5-10% final loss |
| T4 optimization | 2x faster training | 50% speedup |
| Metrics logging | Better insights | Easier debugging |

## Files Modified

1. `genesis_rna/genesis_rna/losses.py`
   - Added `BinaryFocalLoss` class
   - Updated `MultiTaskLoss` to support focal loss

2. `genesis_rna/genesis_rna/train_pretrain.py`
   - Added `MetricsLogger` class
   - Updated `get_lr_scheduler()` for cosine annealing
   - Fixed deprecated PyTorch AMP APIs
   - Added metrics logging calls

3. `genesis_rna/genesis_rna/config.py`
   - Added focal loss parameters
   - Added LR scheduler configuration
   - Updated loss weights
   - Reduced worker count

## Files Added

1. `configs/train_t4_optimized.yaml` - T4 GPU optimized configuration
2. `scripts/visualize_metrics.py` - Training metrics visualization tool
3. `scripts/test_training.sh` - Quick test script
4. `IMPROVEMENTS.md` - This document

## Troubleshooting

### If pair F1 is still 0%

1. Check loss weights are correct (pair_loss_weight should be 1.5)
2. Verify focal loss is enabled: `use_focal_loss_for_pairs: true`
3. Try increasing focal_alpha to 0.85
4. Try reducing AST target activation to 0.5 (train on more samples)

### If training is too slow on T4

1. Increase batch size to 48-64 (if VRAM allows)
2. Reduce number of workers to 1
3. Ensure FP16 is enabled
4. Check GPU utilization with `nvidia-smi`

### If loss explodes

1. Reduce learning rate to 1e-4
2. Increase warmup steps to 1000
3. Check gradient clipping is enabled (should be 1.0)

## Next Steps

1. **Run Full Training**: Use the T4-optimized config for 10-20 epochs
2. **Monitor Pair Metrics**: Watch for steady improvement in pair F1
3. **Visualize Results**: Use the visualization script after training
4. **Fine-tune**: Adjust focal_alpha and pair_loss_weight based on results
5. **Evaluate**: Test on held-out sequences to verify generalization

## Technical Details

### Focal Loss Formula

For binary classification:

```
FL(p_t) = -α(1-p_t)^γ * BCE(p_t)

where:
  p_t = probability of correct class
  α = balancing factor (0.75 for pairs)
  γ = focusing parameter (2.0)
```

### Cosine Annealing Schedule

```
lr_t = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(π * t/T))

where:
  t = current step (after warmup)
  T = total steps
  lr_min = min_lr_ratio * lr_max
  lr_max = peak learning rate
```

## Questions?

If you encounter issues or have questions:
1. Check the training_metrics.csv for detailed metrics
2. Generate visualizations to identify problems
3. Review the AST statistics printed after each epoch
4. Check GPU memory usage with `nvidia-smi`

## Summary

These improvements address the critical pair prediction issue while also:
- Modernizing the codebase (PyTorch AMP)
- Optimizing for your hardware (T4 GPU)
- Adding professional monitoring tools
- Improving training efficiency

The combination of focal loss + increased pair weight should prevent the pair F1 from collapsing to 0%, which was the main issue in your previous training run.

Good luck with your training! 🚀
