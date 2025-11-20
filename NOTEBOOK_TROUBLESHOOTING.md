# Notebook Troubleshooting Guide

## Common Error: `NameError: name 'analyzer' is not defined`

### The Problem

This error occurs when you try to use the `analyzer` object before it has been initialized. It typically looks like this:

```python
======================================================================
BRCA1 Pathogenic Variant Analysis
======================================================================
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipython-input-XXXXX.py in <cell line: 0>()
     10 # Analyze
---> 11 pred = analyzer.predict_variant_effect(
     12     gene='BRCA1',
     13     wild_type_rna=wt_brca1,

NameError: name 'analyzer' is not defined
```

### Root Cause

Jupyter notebooks execute cells independently. If you:
1. **Skip cells** - You didn't run the cell that creates the `analyzer`
2. **Run cells out of order** - You ran analysis cells before initialization cells
3. **Restart the kernel** - The notebook kernel crashed or was restarted, clearing all variables

### The Solution

#### ✅ Option 1: Run All Cells (RECOMMENDED)

The safest way to run any notebook:

1. Click **Runtime → Run all** (Google Colab)
   - OR **Kernel → Restart & Run All** (Jupyter)
2. Wait for all cells to complete
3. All variables will be properly initialized

#### ✅ Option 2: Run Cells in Order

If you want to run cells manually:

**For `notebooks/brca1_variant_analysis.ipynb`:**
1. Cell 2-3: Setup and imports
2. Cell 5: Define VariantPrediction
3. Cell 7: Define BreastCancerAnalyzer class
4. Cell 9: Initialize model and tokenizer
5. Cell 11: **Initialize the analyzer** ← This creates `analyzer`
6. Cell 13: Run analysis (uses `analyzer`)

**For `breast_cancer_colab.ipynb`:**
1. Cell 2: Check GPU
2. Cell 3-4: Clone repo and install dependencies
3. Cell 6: Initialize model and tokenizer
4. Cell 8: **Initialize analyzer and designer** ← This creates `analyzer`
5. Cell 10: Verify setup (optional but recommended)
6. Cell 12: Run BRCA1 analysis (uses `analyzer`)
7. Cell 14: Run therapeutic design (uses `designer`)

#### ✅ Option 3: Re-run Initialization Cells

If you get the error, simply re-run these cells in order:

```python
# 1. Imports (if needed)
import torch
from genesis_rna.model import GenesisRNAModel
from genesis_rna.config import GenesisRNAConfig
from genesis_rna.tokenization import RNATokenizer

# 2. Initialize model
model_config = GenesisRNAConfig(...)
model = GenesisRNAModel(model_config)
tokenizer = RNATokenizer()

# 3. Initialize analyzer
analyzer = BreastCancerAnalyzer(model, tokenizer, device=device)

# 4. Now you can use analyzer
prediction = analyzer.predict_variant_effect(...)
```

### Safety Checks (Added in v2.0)

As of v2.0, our notebooks include built-in safety checks that will give you helpful error messages:

```python
if 'analyzer' not in dir():
    raise RuntimeError(
        "❌ ERROR: Analyzer not initialized!\n\n"
        "You must run ALL previous cells in order before running this cell...\n"
        "HOW TO FIX:\n"
        "  Option 1: Click 'Runtime → Run all' to run everything\n"
        "  Option 2: Run cells 2, 3, 5, 7, 9, 11 in order, then try again\n"
    )
```

These checks will tell you exactly what's missing and how to fix it!

---

## Other Common Errors

### Error: `NameError: name 'model' is not defined`

**Cause:** Model not initialized

**Fix:** Run the model initialization cell:
```python
model_config = GenesisRNAConfig(...)
model = GenesisRNAModel(model_config)
```

### Error: `NameError: name 'designer' is not defined`

**Cause:** mRNA designer not initialized (in `breast_cancer_colab.ipynb`)

**Fix:** Run cell 8 which creates both `analyzer` and `designer`:
```python
analyzer = BreastCancerAnalyzer(model, tokenizer, device=device)
designer = mRNATherapeuticDesigner(model, tokenizer, device=device)
```

### Error: `ModuleNotFoundError: No module named 'genesis_rna'`

**Cause:** Repository not properly installed or path not set

**Fix for Google Colab:**
```python
# Clone repo
!git clone https://github.com/oluwafemidiakhoa/genesi_ai.git
%cd genesi_ai

# Add to path
import sys
sys.path.insert(0, 'genesis_rna')
```

**Fix for local Jupyter:**
```bash
cd genesi_ai/genesis_rna
pip install -e .
```

### Error: `RuntimeError: CUDA out of memory`

**Cause:** Model too large for GPU memory

**Fix:** Use a smaller model or reduce batch size:
```python
# Use small model instead of base/large
model_config = GenesisRNAConfig(
    d_model=256,  # smaller
    n_layers=4,   # fewer layers
    n_heads=4,
    dim_ff=1024
)
```

### Error: `AttributeError: 'dict' object has no attribute 'vocab_size'`

**Cause:** Trying to load a checkpoint with dict config

**Fix:** Use `from_pretrained()` method:
```python
# Correct way
model = GenesisRNAModel.from_pretrained('checkpoint.pt', device='cuda')

# NOT: model = GenesisRNAModel(checkpoint['config']['model'])
```

---

## Best Practices for Jupyter Notebooks

### 1. Always Run Cells in Order

Notebooks are designed to be executed top-to-bottom. Always run cells in order!

### 2. Use "Run All" for Reliability

Before presenting results or sharing notebooks, always do a clean run:
1. **Restart kernel** (clears all variables)
2. **Run all cells** (ensures everything works in order)
3. Verify all outputs are correct

### 3. Check for Initialization

Before running analysis cells, verify your setup:

```python
# Add verification cells
print(f"Model initialized: {'model' in dir()}")
print(f"Tokenizer initialized: {'tokenizer' in dir()}")
print(f"Analyzer initialized: {'analyzer' in dir()}")
```

### 4. Use Descriptive Cell Markers

Our notebooks use clear section headers:
- **Step 1: Setup and Imports**
- **Step 2: Initialize Model**
- **Step 3: Create Analyzer**
- **Step 4: Run Analysis**

Follow this structure when creating new notebooks!

### 5. Add Safety Checks (for developers)

When creating analysis cells, add safety checks:

```python
# SAFETY CHECK
if 'analyzer' not in dir():
    raise RuntimeError("Analyzer not initialized! Run previous cells first.")

# Now safe to use analyzer
prediction = analyzer.predict_variant_effect(...)
```

---

## Quick Reference: Notebook Structure

### `notebooks/brca1_variant_analysis.ipynb`

```
Cell 2-3:  Imports + Setup        → Creates: device
Cell 5:    VariantPrediction      → Creates: VariantPrediction class
Cell 7:    Analyzer Class         → Creates: BreastCancerAnalyzer class
Cell 9:    Model Init             → Creates: model, tokenizer
Cell 11:   Analyzer Init          → Creates: analyzer ⭐
Cell 13:   BRCA1 Analysis         → Uses: analyzer
```

**Critical:** Cell 11 creates `analyzer` - must run before Cell 13!

### `breast_cancer_colab.ipynb`

```
Cell 2:    GPU Check              → Verifies GPU
Cell 3-4:  Setup                  → Clones repo, installs deps
Cell 6:    Model Init             → Creates: model, tokenizer
Cell 8:    Analyzers Init         → Creates: analyzer, designer ⭐
Cell 10:   Verification (optional)→ Checks all initialized
Cell 12:   BRCA1 Analysis         → Uses: analyzer
Cell 14:   Therapeutic Design     → Uses: designer
```

**Critical:** Cell 8 creates `analyzer` and `designer` - must run before Cells 12 and 14!

---

## Getting Help

### If this guide doesn't solve your issue:

1. **Check the error message carefully** - Our safety checks provide detailed instructions
2. **Restart kernel and run all** - 90% of issues are fixed by this
3. **Read the notebook instructions** - Each notebook has a header explaining how to use it
4. **Check documentation:**
   - `README.md` - Project overview
   - `CLAUDE.md` - Comprehensive developer guide
   - `BREAST_CANCER_QUICKSTART.md` - Quick start guide

### Still stuck?

- **GitHub Issues:** https://github.com/oluwafemidiakhoa/genesi_ai/issues
- **Documentation:** See all `.md` files in repository root

---

## Version History

- **v2.0** (2025-11-20): Added safety checks to all notebooks, created this guide
- **v1.0** (2025-11-15): Initial notebooks without safety checks

---

**Remember:** Notebooks are designed to run **top to bottom** in order. When in doubt, click **"Run All"**! 🚀
