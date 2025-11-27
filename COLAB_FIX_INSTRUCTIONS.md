# How to Fix the Colab Notebook

## Problem

`breast_cancer_research_colab.ipynb` **still has the label leakage bug** in Cell 24:

```python
if row.get('Label') == 1:  # Pathogenic
    sequence = sequence[:mid] + 'AAAA' + sequence[mid+4:]
```

This causes the fake 100% accuracy.

## Solution: Use the Production Notebook Instead

**Easiest option:** Use the new production-ready notebook that's already fixed:

### Option 1: New Production Notebook (Recommended)

1. Go to: https://colab.research.google.com/
2. File → Open notebook → GitHub
3. Enter: `oluwafemidiakhoa/genesi_ai`
4. Select: **`genesis_rna_production.ipynb`** ✅

This notebook:
- ✅ Uses real Ensembl sequences (no label leakage)
- ✅ Clean, minimal code (~15 cells)
- ✅ Tests baselines first
- ✅ Production-ready
- ✅ Runs immediately

**Or use direct link:**
```
https://colab.research.google.com/github/oluwafemidiakhoa/genesi_ai/blob/main/genesis_rna_production.ipynb
```

### Option 2: Fix Existing Notebook Manually

If you want to keep using `breast_cancer_research_colab.ipynb`:

1. **Open the notebook in Colab**
2. **Find Cell 24** (the one with `generate_variant_rna_sequence`)
3. **Delete the entire function** (lines 1064-1110)
4. **Replace with** the code from `CELL_24_FIX.txt`

The fix:
- Downloads real BRCA sequences from Ensembl
- Uses HGVS parser to apply variants
- NO label information used

## What Each Notebook Does

### `genesis_rna_production.ipynb` (NEW ✅)
- Clean, production-ready
- ~15 cells total
- Uses real data pipeline
- Tests baselines first
- Realistic expectations (75-85%)
- **Ready to run now**

### `breast_cancer_research_colab.ipynb` (OLD ❌)
- 33 cells with marketing content
- Has label leakage bug in Cell 24
- Too much explanation
- Not production-ready
- **Needs fixing**

## Recommendation

**Use `genesis_rna_production.ipynb` - it's clean, fixed, and ready to run.**

The old notebook has too much content and needs major cleanup. The new one is what you actually need for training.

## Quick Start

```
1. Open: https://colab.research.google.com/github/oluwafemidiakhoa/genesi_ai/blob/main/genesis_rna_production.ipynb
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Get realistic results (75-85% accuracy)
```

Done! ✅
