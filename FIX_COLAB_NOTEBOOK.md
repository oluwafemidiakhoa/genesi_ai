# 🔧 Fix for Colab Notebook - Analyzer Not Defined Error

## Problem

When running Cell 20 in `breast_cancer_research_colab.ipynb`, you get:
```
NameError: name 'analyzer' is not defined
```

## Solution

The analyzer needs to be initialized in **Cell 17 or 18** before being used. Here's the complete fix:

---

## Option 1: Replace Cell 17 (Recommended)

**Delete the current Cell 17** and replace with this simpler version:

```python
# Load model and initialize analyzer
import torch
import sys
sys.path.insert(0, '/content/genesi_ai/genesis_rna')

from genesis_rna.breast_cancer import BreastCancerAnalyzer

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Find model path (adjust if needed)
MODEL_PATH = f"{DRIVE_DIR}/checkpoints/quick/best_model.pt"

# Alternative paths to try
import os
if not os.path.exists(MODEL_PATH):
    alternative_paths = [
        f"{DRIVE_DIR}/checkpoints/full/best_model.pt",
        "/content/genesi_ai/checkpoints/quick/best_model.pt",
    ]
    for path in alternative_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            break

print(f"📥 Loading model from {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("\n⚠️  Please complete Step 2 (Training) first!")
    print("Or update MODEL_PATH to point to your trained model")
else:
    # Initialize analyzer
    analyzer = BreastCancerAnalyzer(MODEL_PATH, device=device)

    print(f"✅ Analyzer initialized on {device}")
    print(f"\nSupported cancer genes:")
    for gene, desc in analyzer.cancer_genes.items():
        print(f"  • {gene}: {desc}")
```

---

## Option 2: Keep Current Cell 17, Add New Cell After It

If you want to keep Cell 17 as-is, add this **NEW CELL** right after Cell 17:

```python
# Initialize BreastCancerAnalyzer with the loaded model
from genesis_rna.breast_cancer import BreastCancerAnalyzer

analyzer = BreastCancerAnalyzer(MODEL_PATH, device=device)

print(f"✅ Analyzer initialized!")
print(f"\nSupported cancer genes:")
for gene, desc in analyzer.cancer_genes.items():
    print(f"  • {gene}: {desc}")
```

---

## Option 3: Delete Cell 18 Entirely (Cleanest)

Cell 18 has ~200 lines defining `BreastCancerAnalyzer` and other classes. This is unnecessary duplication!

**Better approach:**
1. Install the package properly (add to Cell 6):
   ```python
   # Install genesis_rna package
   %cd /content/genesi_ai/genesis_rna
   !pip install -e . -q
   %cd /content/genesi_ai
   ```

2. Delete Cell 18 completely

3. Use Cell 17 from Option 1 above

This way you use the actual `breast_cancer.py` module instead of duplicating code.

---

## Complete Fixed Workflow

Here's the correct cell sequence:

### Cell 16: Verify Model

```python
import os

# Check if model was trained
MODEL_PATH = f"{DRIVE_DIR}/checkpoints/quick/best_model.pt"

if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model found: {MODEL_PATH}")
    print(f"   Size: {file_size:.2f} MB")
else:
    print(f"❌ Model not found at {MODEL_PATH}")
    print("\n⚠️  Please run Step 2 (Training) first!")
```

### Cell 17: Initialize Analyzer

```python
import torch
import sys
sys.path.insert(0, '/content/genesi_ai/genesis_rna')

from genesis_rna.breast_cancer import BreastCancerAnalyzer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"📥 Loading analyzer from {MODEL_PATH}...")
analyzer = BreastCancerAnalyzer(MODEL_PATH, device=device)

print(f"✅ Analyzer ready on {device}!")
print(f"\nSupported genes: {', '.join(analyzer.cancer_genes.keys())}")
```

### Cell 18: DELETE THIS CELL
(Or keep it only if you want to see the implementation)

### Cell 19: Skip
(Optional - only if you want mRNA designer)

### Cell 20: Analyze BRCA1 (Now Works!)

```python
import json

# Load variants from database
with open('/content/genesi_ai/data/breast_cancer/brca_variants.json') as f:
    variants = json.load(f)

# Get BRCA1 variant
variant_id = 'BRCA1_c.5266dupC'
variant = variants[variant_id]

print("="*70)
print("BRCA1 Pathogenic Variant Analysis")
print("="*70)
print(f"\nVariant: {variant_id}")
print(f"Description: {variant['description']}")

# Analyze
pred = analyzer.predict_variant_effect(
    gene=variant['gene'],
    wild_type_rna=variant['wild_type'],
    mutant_rna=variant['mutant'],
    variant_id=variant_id
)

print(f"\n{'Variant ID:':<30} {pred.variant_id}")
print(f"{'Pathogenicity Score:':<30} {pred.pathogenicity_score:.3f}")
print(f"{'ΔStability (kcal/mol):':<30} {pred.delta_stability:.2f}")
print(f"{'Clinical Interpretation:':<30} {pred.interpretation}")
print(f"{'Confidence:':<30} {pred.confidence:.3f}")

# Validate
known_pathogenic = variant['clinical_significance'] == 'Pathogenic'
predicted_pathogenic = pred.interpretation in ['Pathogenic', 'Likely Pathogenic']
concordant = known_pathogenic == predicted_pathogenic

print(f"\n{'ClinVar Concordance:':<30} {'✅ Match' if concordant else '❌ Mismatch'}")

print("\n📋 Clinical Significance:")
print("  • Known pathogenic frameshift (5382insC)")
print("  • Disrupts DNA repair")
print("  • 65-80% lifetime breast cancer risk")
print("  • Recommend: Enhanced screening + counseling")
```

---

## Quick Test

After fixing, run this in a new cell to verify analyzer works:

```python
# Quick test
print("Testing analyzer...")
print(f"Analyzer type: {type(analyzer)}")
print(f"Device: {analyzer.device}")
print(f"Model loaded: {analyzer.model is not None}")
print("✅ Analyzer working!")
```

---

## Why This Happened

The original notebook had:
- Cell 17: Loads model manually (complex code)
- Cell 18: Defines BreastCancerAnalyzer class inline (~200 lines)
- Cell 19: Creates analyzer instance

But Cell 18/19 weren't being executed or were failing silently.

**Better approach:** Use the actual `genesis_rna.breast_cancer` module!

---

## TL;DR - Fastest Fix

1. Add to Cell 6 (after cloning repo):
   ```python
   %cd /content/genesi_ai/genesis_rna
   !pip install -e . -q
   %cd /content/genesi_ai
   ```

2. Replace Cell 17 with:
   ```python
   from genesis_rna.breast_cancer import BreastCancerAnalyzer
   import torch

   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   analyzer = BreastCancerAnalyzer(MODEL_PATH, device=device)
   print(f"✅ Analyzer ready!")
   ```

3. Delete Cell 18 (duplication)

4. Run Cell 20 (should work now!)

---

🎗️ **Now your notebook will work for curing breast cancer!**
