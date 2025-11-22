# ✅ Fix for mRNATherapeuticDesigner in Colab

**Error:** `AttributeError: 'mRNATherapeuticDesigner' object has no attribute 'design_therapeutic'`

**Root Cause:** The method is called `design()`, not `design_therapeutic()`

---

## 🔧 Quick Fix (Two Steps)

### Step 1: Pull Latest Changes from GitHub

In your Colab notebook, run this cell:

```python
# Pull latest fixes from GitHub
!cd /content/genesi_ai && git pull origin main
```

### Step 2: Reload All Modules

Run this reload cell:

```python
import sys

# Remove cached modules
for module in ['genesis_rna.breast_cancer', 'genesis_rna.model', 'genesis_rna.tokenization', 'genesis_rna.config']:
    if module in sys.modules:
        del sys.modules[module]

# Re-import with fixes
sys.path.insert(0, '/content/genesi_ai/genesis_rna')
from genesis_rna.breast_cancer import BreastCancerAnalyzer, mRNATherapeuticDesigner, NeoantigenDiscovery

# Reinitialize all tools
analyzer = BreastCancerAnalyzer(MODEL_PATH, device=device)
designer = mRNATherapeuticDesigner(MODEL_PATH, device=device)
neoantigen_designer = NeoantigenDiscovery(MODEL_PATH, device=device)

print("✅ All tools reloaded with fixes!")
```

### Step 3: Fix Your Code

**Change this (WRONG):**
```python
therapeutic = designer.design_therapeutic(
    protein_sequence=p53_protein,
    optimize_for='stability',
    target_stability=0.95,
    target_translation=0.90
)
```

**To this (CORRECT):**
```python
therapeutic = designer.design(
    protein_sequence=p53_protein,
    optimization_goals={
        'stability': 0.95,
        'translation': 0.90
    }
)
```

---

## ✅ Complete Working Code

Here's the full corrected cell:

```python
print("="*70)
print("mRNA Therapeutic Design: p53")
print("="*70)

p53_protein = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDD"

print(f"\nTarget: p53 tumor suppressor")
print(f"Length: {len(p53_protein)} amino acids")
print(f"\n⚙️ Designing mRNA...")

# FIXED: Use design() method with optimization_goals dict
therapeutic = designer.design(
    protein_sequence=p53_protein,
    optimization_goals={
        'stability': 0.95,
        'translation': 0.90
    }
)

print(f"\n✅ Design complete!")
print(f"\n{'Property':<30} {'Value'}")
print("="*50)
print(f"{'Length:':<30} {therapeutic.length} nt")
print(f"{'Stability:':<30} {therapeutic.stability_score:.3f}")
print(f"{'Translation:':<30} {therapeutic.translation_score:.3f}")
print(f"{'Immunogenicity:':<30} {therapeutic.immunogenicity_score:.3f}")
print(f"{'Half-life:':<30} {therapeutic.half_life_hours:.1f} hours")

print(f"\n🧬 Sequence (first 100 nt):")
print(f"   {therapeutic.sequence[:100]}...")

# Save to Drive
import json
result = {
    'protein': p53_protein,
    'mrna': therapeutic.sequence,
    'stability': therapeutic.stability_score,
    'translation': therapeutic.translation_score,
    'half_life': therapeutic.half_life_hours
}

with open(f"{DRIVE_DIR}/results/p53_therapeutic.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n💾 Saved to {DRIVE_DIR}/results/p53_therapeutic.json")
```

---

## 📋 Summary of Changes

1. **Method name:** `design_therapeutic()` → `design()`
2. **Parameter format:** Individual parameters → `optimization_goals` dict
3. **Fixed initialization:** `mRNATherapeuticDesigner` now accepts `model_path` string

---

## 🎯 What the Latest Fix Does

The latest GitHub commit (`c12f13e`) fixed:
- `mRNATherapeuticDesigner.__init__` now accepts `model_path: str` (like other classes)
- Loads model internally (same pattern as `BreastCancerAnalyzer` and `NeoantigenDiscovery`)

**Before (broken):**
```python
designer = mRNATherapeuticDesigner(model_object, device)  # Expected model object
```

**After (fixed):**
```python
designer = mRNATherapeuticDesigner(MODEL_PATH, device)  # Accepts model path string ✅
```

---

## 🔍 Quick Verification

After pulling and reloading, test all tools:

```python
print("Testing all tools...")
print(f"✓ analyzer: {type(analyzer)}")
print(f"✓ designer: {type(designer)}")
print(f"✓ neoantigen_designer: {type(neoantigen_designer)}")
print(f"\n✓ designer.design method: {hasattr(designer, 'design')}")
print("\n✅ All tools working!")
```

Expected output:
```
Testing all tools...
✓ analyzer: <class 'genesis_rna.breast_cancer.BreastCancerAnalyzer'>
✓ designer: <class 'genesis_rna.breast_cancer.mRNATherapeuticDesigner'>
✓ neoantigen_designer: <class 'genesis_rna.breast_cancer.NeoantigenDiscovery'>

✓ designer.design method: True

✅ All tools working!
```

---

**Ready to cure cancer!** 🎗️
