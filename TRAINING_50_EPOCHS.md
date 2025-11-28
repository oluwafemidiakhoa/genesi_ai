# Clinical-Grade Training Guide: 50 Epochs

**Developer:** Oluwafemi Idiakhoa
**Training Time:** 6-8 hours on T4 GPU
**Model:** BASE (35M parameters)
**Configuration:** `configs/clinical_grade.yaml`

---

## Training Overview

This guide walks you through training the Genesis RNA model for **50 epochs** using clinical-grade hyperparameters based on published genomics research.

### What to Expect

| Metric | Value |
|--------|-------|
| **Training Time** | 6-8 hours (T4 GPU) |
| **GPU Required** | T4 (16GB VRAM minimum) |
| **Epochs** | 50 (with early stopping) |
| **Checkpoints** | Saved every 5 epochs |
| **Final Model Size** | ~140 MB |
| **Training Data** | 50K+ ncRNA sequences |

---

## Step-by-Step Instructions

### 1. Open Colab Notebook

Go to: [Genesis RNA Colab](https://colab.research.google.com/github/oluwafemidiakhoa/genesi_ai/blob/main/genesis_rna/breast_cancer_research_colab.ipynb)

### 2. Connect GPU

**CRITICAL:** You need a GPU for training.

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

Verify GPU:
```python
!nvidia-smi
```

You should see: **Tesla T4** with **15GB** available memory

### 3. Mount Google Drive

Run Cell 4 to mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Why:** Model checkpoints will be saved to Drive (they're too large for Colab's temporary storage)

### 4. Run Setup Cells

Run these cells in order:
- **Cell 3:** Check GPU
- **Cell 4:** Mount Drive
- **Cell 5:** Clone repository
- **Cell 6:** Install dependencies (~3 minutes)

### 5. Download Real ncRNA Data

Run **Cell 10** to download real human ncRNA from Ensembl:

```bash
wget ftp://ftp.ensembl.org/pub/current_fasta/homo_sapiens/ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz
```

**Size:** ~50MB compressed → ~150MB uncompressed
**Time:** ~2 minutes
**Sequences:** 50,000+ real human ncRNA

### 6. Start 50-Epoch Training

Run **Cell 11** (Clinical-Grade Training):

```python
!python -m genesis_rna.train_pretrain \
    --config /content/genesi_ai/configs/clinical_grade.yaml \
    --data_path ../data/human_ncrna \
    --output_dir "{CHECKPOINT_DIR}"
```

**This will take 6-8 hours.** Here's what happens:

---

## Training Timeline (50 Epochs)

### Hour 0-1: Initialization and Warmup
```
Epoch 1/50:   ████░░░░░░░░░░░░░░░░  5%
Training Loss: 4.23 → 3.87
Learning Rate: 0.0 → 3e-4 (warmup)
AST Activation: ~35-45% (stabilizing)
```

**What's happening:**
- Model learning basic RNA patterns
- Learning rate warming up from 0 to 3e-4
- AST controller adjusting to target 40% activation

### Hour 1-3: Main Training (Epochs 5-20)
```
Epoch 10/50:  ████████░░░░░░░░░░░░  20%
Training Loss: 2.89 → 2.45
Validation Loss: 2.73 → 2.52
Learning Rate: 3e-4 → 2e-4 (cosine decay)
AST Activation: ~40% (stable)
```

**What's happening:**
- Loss decreasing steadily
- Checkpoints saved every 5 epochs
- Early stopping monitoring validation loss

### Hour 3-6: Convergence (Epochs 20-40)
```
Epoch 30/50:  ████████████░░░░░░░░  60%
Training Loss: 2.12 → 1.95
Validation Loss: 2.21 → 2.15
Learning Rate: 1.5e-4 → 5e-5 (cosine)
```

**What's happening:**
- Loss improvements slowing down
- Learning rate decaying
- Model converging to optimal performance

### Hour 6-8: Final Epochs (40-50)
```
Epoch 50/50:  ████████████████████  100%
Training Loss: 1.87
Validation Loss: 2.08
Learning Rate: 6e-6 (minimum)
```

**Early stopping may trigger around epoch 40-45 if validation loss stops improving.**

---

## Monitoring Progress

### Real-Time Monitoring

Watch for these logs:

```
Epoch 10/50 - Train Loss: 2.45, Val Loss: 2.52, LR: 2.1e-4
  MLM Accuracy: 38.2%
  AST Activation: 39.8%
  Time: 8m 24s

Checkpoint saved: epoch_10_valloss_2.52.pt
```

### Key Metrics to Watch

1. **Validation Loss** (most important)
   - Should decrease from ~4.0 to ~2.0
   - If it stops decreasing for 10 epochs → early stopping

2. **MLM Accuracy**
   - Should increase from ~20% to ~40%
   - Higher = better RNA understanding

3. **AST Activation Rate**
   - Should hover around 40%
   - Too high (>60%) = controller issue
   - Too low (<20%) = not enough hard samples

4. **Learning Rate**
   - Starts at 3e-4 (after warmup)
   - Decays to 6e-6 by epoch 50

### Warning Signs

🚨 **Stop training if you see:**

- **NaN loss** → Something broke, restart with lower LR
- **Validation loss increasing** → Overfitting (early stopping will handle this)
- **GPU out of memory** → Reduce batch_size from 48 to 32
- **AST activation = 0%** → AST controller failed

---

## Checkpoints Saved

### During Training

Every 5 epochs:
```
/content/drive/MyDrive/breast_cancer_research/checkpoints/clinical_grade/
  ├── checkpoint_epoch_5.pt
  ├── checkpoint_epoch_10.pt
  ├── checkpoint_epoch_15.pt
  ├── checkpoint_epoch_20.pt
  ├── checkpoint_epoch_25.pt
  ├── checkpoint_epoch_30.pt
  ├── checkpoint_epoch_35.pt
  ├── checkpoint_epoch_40.pt
  ├── checkpoint_epoch_45.pt
  └── checkpoint_epoch_50.pt
```

### Best Models (Top-3)

```
├── best_val_loss.pt              ← MAIN MODEL (use this!)
├── checkpoint_epoch_38_valloss_2.08.pt
├── checkpoint_epoch_42_valloss_2.09.pt
└── checkpoint_epoch_35_valloss_2.11.pt
```

**Always use `best_val_loss.pt` for inference!**

---

## After Training Completes

### 1. Verify Training Success

Look for this output:
```
✓ Training completed successfully
  Model: best_val_loss.pt
  Size: 142.35 MB
  Final epoch: 43 (early stopped)
  Best val loss: 2.08
  Ensemble models: 3 checkpoints saved
```

### 2. View Training Curves

The notebook automatically generates plots:

```
training_curves.png saved to:
/content/drive/MyDrive/breast_cancer_research/checkpoints/clinical_grade/
```

**Plots include:**
- Loss curves (train vs validation)
- Learning rate schedule
- AST activation rate

### 3. Next Steps

Now you can:
1. **Analyze BRCA variants** (Cell 16-20)
2. **Download the trained model** from Google Drive
3. **Share the model** (upload to HuggingFace Hub)
4. **Fine-tune** on specific variant datasets

---

## Expected Final Performance

Based on evidence from similar models:

| Metric | Expected Value | Comparison |
|--------|---------------|------------|
| **Final Train Loss** | 1.8 - 2.0 | Lower is better |
| **Final Val Loss** | 2.0 - 2.2 | Should be close to train loss |
| **MLM Accuracy** | 38 - 42% | Competitive with DNABERT |
| **Training Time** | 6-8 hours | T4 GPU |

**After fine-tuning on BRCA variants:**
- AUC-ROC: 0.82 - 0.88 (target: >0.85)
- Sensitivity: 0.88 - 0.92 (target: >0.90)
- Specificity: 0.84 - 0.88 (target: >0.85)

---

## Troubleshooting

### Issue: "Out of Memory" Error

**Solution 1:** Reduce batch size
```yaml
# In clinical_grade.yaml, change:
batch_size: 32  # from 48
```

**Solution 2:** Restart Colab runtime
```
Runtime → Restart runtime → Run all cells again
```

### Issue: Training is Very Slow

**Check:**
1. Is GPU connected? Run `!nvidia-smi`
2. Is FP16 enabled? Should see "Using mixed precision: True"
3. Is data on local disk (not Drive)? ncRNA should be in `/content/genesi_ai/data/`

**Expected speed:** ~10 minutes per epoch (batch_size=48, T4 GPU)

### Issue: Validation Loss Not Decreasing

**Possible causes:**
1. Learning rate too high → Lower to 1e-4
2. Dataset too small → Use full Ensembl ncRNA
3. Early in training → Wait until epoch 15-20

**Early stopping will handle this automatically.**

### Issue: Colab Disconnects

**Colab free tier:** 12-hour limit, may disconnect

**Solutions:**
1. **Use Colab Pro** ($10/month) → 24-hour sessions
2. **Keep browser active** → Move mouse occasionally
3. **Resume training:** Use `--resume_from` flag (if implemented)

**Checkpoints are saved every 5 epochs, so you won't lose much progress!**

---

## Cost Analysis

### Free Colab (T4 GPU)
- **Cost:** $0
- **Limit:** 12 hours/session (may disconnect)
- **Risk:** May need to restart if session expires

**Recommendation:** Start training, keep browser active

### Colab Pro ($10/month)
- **Cost:** $10/month
- **Limit:** 24 hours/session
- **Benefit:** Uninterrupted 6-8 hour training

**Worth it for:** Serious research, multiple training runs

### Local GPU (if you have one)
- **RTX 3090 / 4090:** ~4-5 hours
- **V100:** ~3-4 hours
- **A100:** ~2-3 hours

---

## Research Use Disclaimer

**⚠️ IMPORTANT**

This model is for **RESEARCH USE ONLY**. After training:
- NOT for clinical diagnosis
- NOT for patient management
- NOT for treatment decisions

For clinical variant interpretation, consult board-certified genetic counselors and follow ACMG/AMP guidelines.

---

## Questions?

**If training fails or you encounter issues:**

1. Check this guide's Troubleshooting section
2. Read the error message carefully
3. Open a GitHub issue with:
   - Error message
   - Epoch where it failed
   - GPU type (run `!nvidia-smi`)

**Developer:** Oluwafemi Idiakhoa
**Institution:** Genesis AI Research
**Last Updated:** January 27, 2025
