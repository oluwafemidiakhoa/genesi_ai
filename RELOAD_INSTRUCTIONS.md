# 🔄 How to Reload Tools in Your Local Notebook

**Issue:** You're running a local Jupyter notebook and getting `NameError` for `analyzer` or `designer` because the module is cached with old code.

**Solution:** Use one of these methods to reload with the latest fixes.

---

## ✅ Method 1: Copy-Paste Reload Cell (Fastest)

**Add this cell to your notebook** (after model loading, before analysis):

```python
# ═══════════════════════════════════════════════════════════════════════
# RELOAD ALL TOOLS WITH FIXES
# ═══════════════════════════════════════════════════════════════════════

import sys
import importlib

# Remove cached modules
modules_to_remove = [
    'genesis_rna.breast_cancer',
    'genesis_rna.model',
    'genesis_rna.tokenization',
    'genesis_rna.config'
]

for module in modules_to_remove:
    if module in sys.modules:
        del sys.modules[module]

# Re-import with fixes
sys.path.insert(0, r'c:\Users\adminidiakhoa\genesi_ai\genesis_rna')
from genesis_rna.breast_cancer import (
    BreastCancerAnalyzer,
    mRNATherapeuticDesigner,
    NeoantigenDesigner
)

# Reinitialize all tools (using your existing MODEL_PATH and device)
analyzer = BreastCancerAnalyzer(MODEL_PATH, device=device)
designer = mRNATherapeuticDesigner(MODEL_PATH, device=device)
neoantigen_designer = NeoantigenDesigner(MODEL_PATH, device=device)

print("✅ All tools reloaded with fixes!")
print(f"   • analyzer - BreastCancerAnalyzer")
print(f"   • designer - mRNATherapeuticDesigner")
print(f"   • neoantigen_designer - NeoantigenDesigner")
print(f"\n   Device: {device}")
print(f"   Model: {MODEL_PATH}")
```

**Then run your analysis cells** - they'll all work now!

---

## ✅ Method 2: Use Helper Script

```python
from reload_analyzer import reload_all_tools

# Reload all tools with fixes
analyzer, designer, neoantigen_designer = reload_all_tools(MODEL_PATH, device=device)

# Now run your analysis!
```

---

## ✅ Method 3: Restart Kernel (Slowest but Most Complete)

1. **Restart your Jupyter kernel:**
   - Menu: Kernel → Restart Kernel
   - Or: Kernel → Restart & Clear Output

2. **Re-run all cells from the beginning**
   - The latest code will be loaded automatically

---

## 📝 What This Fixes

**Before reload:**
- ❌ `IndexError: too many indices for tensor of dimension 1` when calling `analyzer.predict_variant_effect()`
- ❌ `NameError: name 'designer' is not defined` when calling `designer.design_therapeutic()`

**After reload:**
- ✅ `analyzer.predict_variant_effect()` works correctly
- ✅ `designer.design_therapeutic()` works correctly
- ✅ `neoantigen_designer.design_neoantigen_vaccine()` works correctly

---

## 🎯 Your Current Workflow

Based on your session, here's what to do:

1. **You already have:**
   - ✅ Model trained or loaded
   - ✅ `MODEL_PATH` and `device` defined

2. **Add the reload cell above** (Method 1)

3. **Run your analysis cells:**
   - BRCA1 variant analysis (uses `analyzer`) ✅
   - mRNA therapeutic design (uses `designer`) ✅
   - Any neoantigen cells (uses `neoantigen_designer`) ✅

---

## 🔧 Why This Is Needed

When you import a Python module in Jupyter, it gets cached in memory. Even if you update the file on disk (`breast_cancer.py`), Jupyter keeps using the old cached version.

**The reload cell:**
1. Removes the cached modules from memory
2. Forces Python to re-import from the updated files
3. Initializes all tools with the fixed code

---

## 💡 Pro Tip

Add the reload cell as a permanent cell in your notebook (after model loading). That way, any time you update the code, you can just re-run that cell to get the latest version without restarting the kernel!

---

## ✅ Verification

After reloading, test that it works:

```python
# Quick test
print("Testing all tools...")
print(f"✓ analyzer: {type(analyzer)}")
print(f"✓ designer: {type(designer)}")
print(f"✓ neoantigen_designer: {type(neoantigen_designer)}")
print("\n✅ All tools loaded successfully!")
```

Should output:
```
Testing all tools...
✓ analyzer: <class 'genesis_rna.breast_cancer.BreastCancerAnalyzer'>
✓ designer: <class 'genesis_rna.breast_cancer.mRNATherapeuticDesigner'>
✓ neoantigen_designer: <class 'genesis_rna.breast_cancer.NeoantigenDesigner'>

✅ All tools loaded successfully!
```

---

## 🎗️ Ready to Cure Cancer!

Once reloaded, all your analysis cells will work without errors:
- ✅ BRCA variant analysis
- ✅ mRNA therapeutic design
- ✅ Neoantigen vaccine design
- ✅ Clinical interpretation
- ✅ Batch processing

**The tokenizer bug is fixed, and all tools are ready to use!** 🎗️
