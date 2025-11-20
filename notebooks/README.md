# Genesis RNA Notebooks

This directory contains Jupyter notebooks and scripts for using Genesis RNA for various RNA analysis tasks.

## 📋 Contents

### BRCA1 Variant Analysis

- **`brca1_variant_analysis.ipynb`** - Interactive notebook for analyzing BRCA1 variants
- **`brca1_variant_analysis.py`** - Standalone Python script version

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
# Install dependencies
pip install -r ../genesis_rna/requirements.txt

# Launch Jupyter
jupyter notebook

# Open: brca1_variant_analysis.ipynb
# Run all cells in order!
```

### Option 2: Python Script

```bash
# Install dependencies
pip install -r ../genesis_rna/requirements.txt

# Run the script
python brca1_variant_analysis.py
```

## ⚠️ Common Error: `NameError: name 'analyzer' is not defined`

### Problem

If you get this error:
```
NameError: name 'analyzer' is not defined
```

### Solution

You must **initialize the analyzer before using it**. You cannot skip directly to the analysis code!

**Step-by-step fix:**

1. **Import dependencies:**
   ```python
   import torch
   from genesis_rna.model import GenesisRNAModel
   from genesis_rna.config import GenesisRNAConfig
   from genesis_rna.tokenization import RNATokenizer
   from genesis_rna.breast_cancer import BreastCancerAnalyzer
   ```

2. **Initialize model and tokenizer:**
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Create model config
   config = GenesisRNAConfig(
       vocab_size=32,
       d_model=256,
       n_layers=4,
       n_heads=4,
       dim_ff=1024,
       max_len=512,
       dropout=0.1,
       structure_num_labels=3
   )

   # Initialize model
   model = GenesisRNAModel(config)
   model.to(device)
   model.eval()

   # Initialize tokenizer
   tokenizer = RNATokenizer()
   ```

3. **Create the analyzer instance:**
   ```python
   # THIS IS THE KEY STEP!
   analyzer = BreastCancerAnalyzer(model, tokenizer, device=device)
   ```

4. **Now you can use it:**
   ```python
   pred = analyzer.predict_variant_effect(
       gene='BRCA1',
       wild_type_rna=wt_brca1,
       mutant_rna=mut_brca1,
       variant_id='BRCA1:c.5266dupC'
   )
   ```

## 📚 Using a Trained Model

The examples above use **randomly initialized weights** (demo mode). For real research, use a trained checkpoint:

### Loading a Pre-trained Model

```python
# Load from checkpoint
model = GenesisRNAModel.from_pretrained(
    'path/to/checkpoint.pt',
    device='cuda'
)

# Create analyzer with trained model
analyzer = BreastCancerAnalyzer(model, tokenizer, device='cuda')
```

### Training Your Own Model

See the main Genesis RNA documentation for training:

1. **Pre-training:** `genesis_rna/genesis_rna_colab_training.ipynb`
2. **Fine-tuning on BRCA data:** `genesis_rna/breast_cancer_research_colab.ipynb`

## 🧬 Example: Complete BRCA1 Analysis

Here's a complete working example:

```python
import torch
from genesis_rna.model import GenesisRNAModel
from genesis_rna.config import GenesisRNAConfig
from genesis_rna.tokenization import RNATokenizer
from genesis_rna.breast_cancer import BreastCancerAnalyzer

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model (demo mode)
config = GenesisRNAConfig(
    vocab_size=32, d_model=256, n_layers=4,
    n_heads=4, dim_ff=1024, max_len=512,
    dropout=0.1, structure_num_labels=3
)
model = GenesisRNAModel(config).to(device).eval()
tokenizer = RNATokenizer()

# Create analyzer
analyzer = BreastCancerAnalyzer(model, tokenizer, device=device)

# Sequences
wt_brca1 = "AUGGGCUUCCGUGUCCAGCUCCUGGGAGCUGCUGGUGGCGGCGGCCGCGGG..."
mut_brca1 = "AUGGGCUUCCGUGUCCAGCUCCUGGGAGCUGCUGGUGGCGGCGGCCGCGGG..."

# Analyze
prediction = analyzer.predict_variant_effect(
    gene='BRCA1',
    wild_type_rna=wt_brca1,
    mutant_rna=mut_brca1,
    variant_id='BRCA1:c.5266dupC'
)

# Results
print(f"Pathogenicity: {prediction.pathogenicity_score:.3f}")
print(f"Interpretation: {prediction.interpretation}")
```

## 📖 Additional Resources

- **Main repository:** [oluwafemidiakhoa/genesi_ai](https://github.com/oluwafemidiakhoa/genesi_ai)
- **Full demo notebook:** `breast_cancer_colab.ipynb` (no training required)
- **Research notebook:** `genesis_rna/breast_cancer_research_colab.ipynb` (includes training)
- **Quick start guide:** `BREAST_CANCER_QUICKSTART.md`

## 💡 Tips

1. **Always run notebook cells in order** - each cell depends on previous ones
2. **Check that `analyzer` is defined** before trying to use it
3. **Use GPU if available** for faster inference
4. **For production:** Train/load a real model instead of using random weights
5. **Validate predictions** against ClinVar or experimental data

## 🆘 Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`
```bash
pip install torch transformers datasets biopython
```

### `ModuleNotFoundError: No module named 'genesis_rna'`
```bash
# Make sure you're in the right directory
cd genesi_ai
# Or add to Python path
export PYTHONPATH="/path/to/genesi_ai/genesis_rna:$PYTHONPATH"
```

### `RuntimeError: CUDA out of memory`
```python
# Use CPU instead
device = 'cpu'
```

Or reduce batch size / sequence length.

## 📝 License

See main repository for license information.
