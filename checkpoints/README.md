# Genesis RNA Checkpoints Organization

This directory contains trained model checkpoints for Genesis RNA. Checkpoints are organized by training type and purpose.

## Directory Structure

```
checkpoints/
├── pretrained/          # Pre-trained foundation models
│   ├── small/          # Small model checkpoints (10M params)
│   ├── base/           # Base model checkpoints (35M params)
│   └── large/          # Large model checkpoints (150M params)
├── finetuned/          # Fine-tuned models for specific tasks
│   ├── structure/      # Structure prediction models
│   ├── pairing/        # Base pairing prediction models
│   └── custom/         # Custom fine-tuned models
└── experiments/        # Experimental training runs
    └── YYYY-MM-DD_description/  # Dated experiment folders
```

## Checkpoint Naming Convention

Checkpoints should follow this naming pattern:

```
{model_size}_{training_type}_{metric}_{value}_epoch{N}.pt

Examples:
- base_pretrain_valloss_2.63_epoch10.pt
- small_pretrain_best_model.pt
- large_finetune_structure_f1_0.85_epoch5.pt
```

## Google Drive Integration

### For Google Colab Training

When training on Google Colab, checkpoints are automatically saved to Google Drive:

```python
# In Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Train with checkpoint saving to Drive
!python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \
    --num_epochs 10
```

**Standard Google Drive path:**
```
/content/drive/MyDrive/genesis_rna_checkpoints/
```

### Downloading from Google Drive

If you've trained on Colab and want to download checkpoints:

1. **Via Google Drive Web Interface:**
   - Navigate to `My Drive/genesis_rna_checkpoints/`
   - Right-click the folder → Download
   - Extract to your local repository under `checkpoints/`

2. **Via Google Drive Desktop:**
   - Sync folder: `genesis_rna_checkpoints`
   - Copy to: `<repo>/checkpoints/pretrained/<model_size>/`

3. **Via `gdown` (programmatic):**
   ```bash
   # Install gdown
   pip install gdown

   # Download checkpoint (requires file ID from shareable link)
   gdown <file_id> -O checkpoints/pretrained/base/best_model.pt
   ```

### Uploading to Google Drive

To continue training with existing checkpoints on Colab:

1. Upload checkpoint to Google Drive
2. Reference in training command:
   ```bash
   python -m genesis_rna.train_pretrain \
       --config configs/train_t4_optimized.yaml \
       --resume_from /content/drive/MyDrive/genesis_rna_checkpoints/checkpoint_epoch_5.pt \
       --output_dir /content/drive/MyDrive/genesis_rna_checkpoints
   ```

## Checkpoint Contents

Each checkpoint file (`.pt`) contains:

```python
{
    'epoch': int,                    # Training epoch number
    'step': int,                     # Global training step
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'config': {                      # Training configuration
        'model': {...},              # Model config
        'training': {...}            # Training config
    }
}
```

## Loading Checkpoints

### Load for Inference (Recommended)

**Use the built-in `from_pretrained()` method:**

```python
from genesis_rna import GenesisRNAModel, RNATokenizer

# Load model (handles config conversion automatically)
model = GenesisRNAModel.from_pretrained(
    'checkpoints/pretrained/base/best_model.pt',
    device='cuda'  # or 'cpu'
)

# Use for predictions
tokenizer = RNATokenizer()
sequence = "ACGUACGUACGU"
inputs = tokenizer.encode(sequence)
outputs = model(inputs['input_ids'])
```

### Load for Inference (Manual)

**If you need manual control over the loading process:**

```python
import torch
from genesis_rna import GenesisRNAModel, GenesisRNAConfig, RNATokenizer

# Load checkpoint
checkpoint = torch.load('checkpoints/pretrained/base/best_model.pt')

# Create model with saved config
model_config_dict = checkpoint['config']['model']
model_config = GenesisRNAConfig.from_dict(model_config_dict)
model = GenesisRNAModel(model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for predictions
tokenizer = RNATokenizer()
sequence = "ACGUACGUACGU"
inputs = tokenizer.encode(sequence)
outputs = model(inputs['input_ids'])
```

### Resume Training

```python
import torch
from genesis_rna import GenesisRNAModel, GenesisRNAConfig
from torch.optim import AdamW

# Load checkpoint
checkpoint = torch.load('checkpoints/pretrained/base/checkpoint_epoch_5.pt')

# Restore model
model_config = GenesisRNAConfig.from_dict(checkpoint['config']['model'])
model = GenesisRNAModel(model_config)
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer
optimizer = AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training from epoch N
start_epoch = checkpoint['epoch'] + 1
```

## Best Practices

### 1. Regular Checkpointing

Save checkpoints at regular intervals:
- Every 5 epochs for long training runs
- Every epoch for short runs (<10 epochs)
- Always save the best model based on validation loss

### 2. Checkpoint Retention

Keep these checkpoints:
- **Best model** (`best_model.pt`) - Always keep
- **Latest checkpoint** - For resuming training
- **Milestone checkpoints** - Every 10-20 epochs for long runs
- **Final checkpoint** - Last epoch of training

Delete intermediate checkpoints to save space.

### 3. Metadata Documentation

Create a `metadata.json` alongside checkpoints:

```json
{
    "model_size": "base",
    "training_type": "pretrain",
    "dataset": "RNACentral_100k",
    "epochs_trained": 50,
    "best_val_loss": 2.63,
    "training_date": "2024-01-15",
    "gpu_type": "T4",
    "training_time_hours": 4.5,
    "notes": "Trained with AST, focal loss for pairs"
}
```

### 4. Version Control

**DO NOT commit checkpoint files to Git** - they are too large.

Instead:
- Keep checkpoints in Google Drive or cloud storage
- Document checkpoint locations in this README
- Share checkpoint links in documentation

## Available Checkpoints

### Pre-trained Models

| Model | Size | Epochs | Val Loss | Pair F1 | Location | Date |
|-------|------|--------|----------|---------|----------|------|
| Small | 10M  | 10     | TBD      | TBD     | Google Drive | TBD |
| Base  | 35M  | 10     | TBD      | TBD     | Google Drive | TBD |
| Large | 150M | -      | -        | -       | Not trained | - |

### Fine-tuned Models

*No fine-tuned models available yet.*

## Storage Guidelines

### Local Storage

- **Development:** Keep 1-2 recent checkpoints locally (~100-500MB each)
- **Production:** Download only the best models needed for inference

### Cloud Storage (Google Drive)

- **Training checkpoints:** Keep in `genesis_rna_checkpoints/` folder
- **Archived runs:** Move old experiments to dated folders
- **Shared models:** Create shareable links for distribution

### Recommended Storage Structure in Google Drive

```
My Drive/
└── genesis_rna_checkpoints/
    ├── current_training/          # Active training run
    │   ├── best_model.pt
    │   ├── checkpoint_epoch_10.pt
    │   ├── training_metrics.csv
    │   └── config.json
    ├── archived/                  # Completed training runs
    │   ├── 2024-01-15_base_pretrain/
    │   ├── 2024-01-20_small_structure/
    │   └── ...
    └── production/                # Production-ready models
        ├── base_pretrain_v1.0.pt
        └── small_pretrain_v1.0.pt
```

## Troubleshooting

### Checkpoint File Too Large

If checkpoint files are too large:
1. Use `torch.save()` with `_use_new_zipfile_serialization=True`
2. Save only model weights, not optimizer state for inference
3. Use model quantization for deployment

### Corrupted Checkpoint

If loading fails:
```python
try:
    checkpoint = torch.load('checkpoint.pt', map_location='cpu')
except Exception as e:
    print(f"Checkpoint corrupted: {e}")
    # Fall back to previous checkpoint
```

### Out of Disk Space

Google Drive free tier: 15GB
- Each checkpoint: ~100-500MB (depending on model size)
- Monitor space usage regularly
- Archive old experiments

## Contributing

When adding new checkpoints:
1. Follow the naming convention
2. Update the "Available Checkpoints" table above
3. Include metadata.json with training details
4. Test checkpoint loading before sharing

## Questions?

- Training issues: See `QUICKSTART_T4.md`
- Model architecture: See `genesis_rna/README.md`
- Technical details: See `IMPROVEMENTS.md`
