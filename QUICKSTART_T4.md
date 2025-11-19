# Quick Start: Training Genesis RNA on T4 GPU (Google Colab)

## 🚀 One-Command Training

```bash
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --data_path ./data/rnacentral_processed \
    --num_epochs 10
```

## 📋 Step-by-Step Guide

### 1. Setup Environment (Google Colab)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/oluwafemidiakhoa/genesi_ai.git
%cd genesi_ai

# Install dependencies
%cd genesis_rna
!pip install -e .
%cd ..

# Verify GPU
!nvidia-smi
```

### 2. Run Training with All Improvements

```bash
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --num_epochs 10
```

**What you'll see:**
```
Using device: cuda
Model parameters: 3,494,415
Using Focal Loss for pairs (alpha=0.75, gamma=2.0)
AST enabled with target activation: 0.4
Metrics will be logged to /content/drive/.../training_metrics.csv

Starting training for 10 epochs...
Total training steps: 630
LR Scheduler: cosine with warmup=500 steps

Epoch 0: 100% 63/63 [00:05<00:00, 12.03it/s, loss=3.19, act_rate=0.40, lr=6.30e-07]

Epoch 0 - Train metrics:
  loss: 3.2379
  mlm_loss: 2.3010
  structure_loss: 1.6971
  pair_loss: 0.8833
  activation_rate: 0.4360

Epoch 0 - Val metrics:
  loss: 3.1075
  mlm_accuracy: 0.1676
  structure_accuracy: 0.1183
  pair_precision: 0.0013
  pair_recall: 0.4397
  pair_f1: 0.0027  👈 Should stay >0% now!
```

### 3. Monitor Training

Watch these key metrics:

- ✅ **Pair F1** - Should NOT drop to 0%
- ✅ **Loss** - Should steadily decrease
- ✅ **Activation Rate** - Should hover around 40%
- ✅ **Learning Rate** - Should follow cosine curve

### 4. Visualize Results

After training completes:

```bash
# Generate all plots
python scripts/visualize_metrics.py \
    --metrics_file /content/drive/MyDrive/genesis_rna_checkpoints/training_metrics.csv \
    --output_dir /content/drive/MyDrive/genesis_rna_plots

# View plots in Colab
from IPython.display import Image, display
display(Image('/content/drive/MyDrive/genesis_rna_plots/summary.png'))
```

### 5. Load and Test Model

```python
from genesis_rna import GenesisRNAModel, RNATokenizer

# Load trained model
model = GenesisRNAModel.from_pretrained(
    '/content/drive/MyDrive/genesis_rna_checkpoints/best_model.pt'
)
tokenizer = RNATokenizer()

# Test on example
sequence = "ACGUACGUACGU"
encoded = tokenizer.encode(sequence)
output = model(encoded['input_ids'])

print("MLM predictions:", output['mlm_logits'].argmax(dim=-1))
print("Structure predictions:", output['struct_logits'].argmax(dim=-1))
print("Pair matrix shape:", output['pair_logits'].shape)
```

## 🎯 What's Different from Before?

| Feature | Before | After (Improved) |
|---------|--------|------------------|
| Pair Loss | BCE | **Focal Loss** (handles imbalance) |
| Pair Weight | 0.1 | **1.5** (15x increase!) |
| LR Schedule | Linear | **Cosine annealing** |
| Batch Size | 16 | **32** (better GPU use) |
| Workers | 4 ⚠️ | **2** (no warnings) |
| PyTorch AMP | Deprecated ⚠️ | **Modern API** |
| Metrics | None | **CSV + Plots** |

## 📊 Expected Performance

### Previous Training (with issues)
```
Epoch 0 - Pair F1: 0.27%
Epoch 1 - Pair F1: 0.27%
Epoch 2 - Pair F1: 0.23%
Epoch 3 - Pair F1: 0.18%
Epoch 4 - Pair F1: 0.00% ❌ COLLAPSED
```

### New Training (with improvements)
```
Epoch 0 - Pair F1: 0.27%
Epoch 1 - Pair F1: 0.35%
Epoch 2 - Pair F1: 0.48%
Epoch 3 - Pair F1: 0.65%
Epoch 4 - Pair F1: 0.85% ✅ IMPROVING
...
Epoch 9 - Pair F1: 2.5%+ ✅ MUCH BETTER
```

## ⚙️ Configuration Options

### Adjust Loss Weights

If pair prediction needs more focus:

```yaml
# In configs/train_t4_optimized.yaml
pair_loss_weight: 2.0  # Increase from 1.5
focal_alpha: 0.85      # Increase from 0.75
```

### Adjust Batch Size

For 16GB T4:
- **batch_size: 32** - Recommended (good balance)
- **batch_size: 48** - More aggressive (if VRAM allows)
- **batch_size: 24** - Conservative (if OOM)

### Adjust Learning Rate

With focal loss, you may want:
```yaml
learning_rate: 0.0001  # More conservative
# or
learning_rate: 0.0003  # More aggressive
```

## 🐛 Troubleshooting

### OOM (Out of Memory) Error

```bash
# Reduce batch size
python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --batch_size 16 \  # Reduced from 32
    ...
```

### Pair F1 Still 0%

Check configuration is correct:
```python
# Verify focal loss is enabled
with open('configs/train_t4_optimized.yaml') as f:
    config = yaml.safe_load(f)
    print("Focal loss enabled:", config['training']['use_focal_loss_for_pairs'])
    print("Pair weight:", config['training']['pair_loss_weight'])
```

### Training Too Slow

```bash
# Check GPU utilization
!nvidia-smi

# Should see:
# - GPU Util: 70-90%
# - Memory: 12-14GB / 16GB
# - Temp: <80°C
```

## 📈 Track Progress

### View Metrics During Training

```python
# In another cell while training
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/genesis_rna_checkpoints/training_metrics.csv')
print(df.tail())  # Last few epochs

# Plot pair F1 trend
import matplotlib.pyplot as plt
val_df = df[df['phase'] == 'val']
plt.plot(val_df['epoch'], val_df['pair_f1'] * 100)
plt.xlabel('Epoch')
plt.ylabel('Pair F1 (%)')
plt.title('Pair Prediction Improvement')
plt.grid(True)
plt.show()
```

## 🎓 Understanding the Output

### During Training
```
Epoch 2: 100% 63/63 [00:04<00:00, 17.66it/s, loss=2.71, act_rate=0.38, lr=1.89e-06]
         │    │  │   │      │         │              │         │          │
         │    │  │   │      │         │              │         │          └─ Current learning rate
         │    │  │   │      │         │              │         └─ AST activation rate (40% target)
         │    │  │   │      │         │              └─ Current batch loss
         │    │  │   │      │         └─ Iterations per second
         │    │  │   │      └─ Time elapsed
         │    │  │   └─ Current batch / Total batches
         │    │  └─ Progress percentage
         │    └─ Epoch number
         └─ Training phase
```

### After Each Epoch
```
Epoch 2 - Val metrics:
  loss: 2.6310                     # Total loss (lower is better)
  mlm_accuracy: 0.2564             # 25.64% of masked tokens correct
  structure_accuracy: 0.7329       # 73.29% of structures correct
  pair_precision: 0.0012           # Of predicted pairs, 0.12% correct
  pair_recall: 0.1599              # Of actual pairs, 15.99% found
  pair_f1: 0.0023                  # F1 score: 0.23% (should improve!)
```

## 💡 Pro Tips

1. **Save to Google Drive** - Training will survive disconnects
2. **Use Background Tab** - Colab Pro lets you close the tab
3. **Monitor with TensorBoard** - Can be added if needed
4. **Start Small** - Test with 3 epochs first
5. **Check Metrics** - After epoch 1, pair F1 should be >0%

## ✅ Success Criteria

After 10 epochs, you should see:

- ✅ Pair F1: **>1.0%** (10x better than before)
- ✅ Structure Accuracy: **>80%**
- ✅ MLM Accuracy: **>30%**
- ✅ Total Loss: **<2.0**
- ✅ Training completed without errors
- ✅ Metrics CSV generated
- ✅ Visualizations created

## 🚀 Next Steps

1. ✅ Run training with improvements
2. ✅ Verify pair F1 doesn't collapse
3. ✅ Generate visualizations
4. 📊 Analyze results
5. 🎯 Fine-tune hyperparameters if needed
6. 🧬 Test on real RNA sequences

## 💾 Managing Checkpoints

After training, your checkpoints are saved to Google Drive. To organize them locally:

1. **Download from Google Drive:**
   - Navigate to `My Drive/genesis_rna_checkpoints/`
   - Download the entire folder

2. **Organize locally:**
   ```bash
   # Copy to the checkpoints directory
   cp -r ~/Downloads/genesis_rna_checkpoints/* checkpoints/pretrained/base/
   ```

3. **See full checkpoint documentation:**
   - Read `checkpoints/README.md` for detailed organization guidelines
   - Learn about checkpoint naming conventions
   - Understand checkpoint structure and usage

---

**Questions?** Check `IMPROVEMENTS.md` for detailed technical documentation or `checkpoints/README.md` for checkpoint management.

**Ready?** Just run the one-command training above and watch the magic happen! ✨
