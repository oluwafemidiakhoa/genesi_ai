# 🎗️ START HERE - Cure Breast Cancer with Genesis RNA

**Welcome!** This is your complete guide to analyzing BRCA1/BRCA2 variants for breast cancer cure research.

---

## ✨ What You Can Do

✅ Analyze pathogenic BRCA1/BRCA2 mutations
✅ Validate predictions against ClinVar database
✅ Measure clinical performance (sensitivity, specificity)
✅ Generate publication-ready visualizations
✅ Identify variants for reclassification

---

## 🚀 Quick Start (Choose One Path)

### Path 1: Simplest - Run One Command ⭐ RECOMMENDED

```bash
python examples/simple_cancer_analysis.py
```

This will:
- Find your trained model (or offer to train one)
- Analyze all 6 BRCA variants
- Show concordance with ClinVar
- Takes 1 minute (if model exists) or 10 minutes (if training)

---

### Path 2: Google Colab - Interactive Research

1. Open `breast_cancer_research_colab.ipynb` in Colab
2. **IMPORTANT:** Follow [FIX_COLAB_NOTEBOOK.md](FIX_COLAB_NOTEBOOK.md) to fix the analyzer error
3. Run cells to train model and analyze variants

**Quick Fix for Colab:**
- Add to Cell 6: `%cd /content/genesi_ai/genesis_rna && !pip install -e . -q && %cd /content/genesi_ai`
- Replace Cell 17 with simple analyzer init (see [FIX_COLAB_NOTEBOOK.md](FIX_COLAB_NOTEBOOK.md))
- Delete Cell 18 (duplicate code)

---

### Path 3: Step-by-Step - Full Control

#### Step 1: Install Package
```bash
cd genesis_rna
pip install -e .
cd ..
```

#### Step 2: Train Model
```bash
python -m genesis_rna.train_pretrain \
    --use_dummy_data \
    --model_size small \
    --num_epochs 5 \
    --output_dir checkpoints/cancer_model
```

#### Step 3: Analyze Variants
```bash
python scripts/batch_variant_analysis.py \
    --model checkpoints/cancer_model/best_model.pt \
    --variants data/breast_cancer/brca_variants.json \
    --output results/analysis.csv
```

#### Step 4: Visualize
```bash
python scripts/visualize_variant_analysis.py \
    --results results/analysis.csv \
    --output plots/
```

---

## 📁 What's in This Repository

### Core Tools
- **`examples/simple_cancer_analysis.py`** - Complete automated workflow
- **`scripts/batch_variant_analysis.py`** - Batch variant processing
- **`scripts/visualize_variant_analysis.py`** - Generate figures
- **`data/breast_cancer/brca_variants.json`** - 6 curated variants

### Documentation
- **[QUICK_START_CANCER_CURE.md](QUICK_START_CANCER_CURE.md)** - Detailed quick start
- **[CANCER_RESEARCH_ENHANCEMENTS.md](CANCER_RESEARCH_ENHANCEMENTS.md)** - Complete feature guide
- **[FIX_COLAB_NOTEBOOK.md](FIX_COLAB_NOTEBOOK.md)** - Fix notebook analyzer error
- **[CLAUDE.md](CLAUDE.md)** - Complete development guide

---

## 🔧 Troubleshooting

### Error: `NameError: name 'analyzer' is not defined`

**In Python script:**
```python
from genesis_rna.breast_cancer import BreastCancerAnalyzer
analyzer = BreastCancerAnalyzer('path/to/model.pt')
```

**In Colab:**
Follow [FIX_COLAB_NOTEBOOK.md](FIX_COLAB_NOTEBOOK.md)

### Error: `ModuleNotFoundError: No module named 'genesis_rna'`

```bash
cd genesis_rna
pip install -e .
cd ..
```

### Error: Model not found

Train a model first:
```bash
python -m genesis_rna.train_pretrain \
    --use_dummy_data \
    --model_size small \
    --num_epochs 3 \
    --output_dir checkpoints/quick_model
```

### Error in `quick_cancer_analysis.py`

Use the simpler version instead:
```bash
python examples/simple_cancer_analysis.py
```

---

## 📊 Understanding Results

### Pathogenicity Score
- **0.0 - 0.2** = Benign
- **0.2 - 0.5** = Uncertain
- **0.5 - 0.8** = Likely Pathogenic
- **0.8 - 1.0** = Pathogenic

### Clinical Metrics
- **Sensitivity** - % pathogenic variants detected (Goal: >90%)
- **Specificity** - % benign variants identified (Goal: >85%)
- **Concordance** - Agreement with ClinVar (Goal: >80%)

### Interpretation
- ✅ **Concordant** - Matches ClinVar (Good!)
- ❌ **Discordant** - Doesn't match (Needs review)

---

## 🎯 Your Workflow

```
1. Train Model (10 min)
   ↓
2. Analyze Variants (1 min)
   ↓
3. Validate vs ClinVar (automatic)
   ↓
4. Visualize Results (1 min)
   ↓
5. Review Discordant Cases
   ↓
6. Add More Variants
   ↓
7. Publish Findings!
```

---

## 📚 Documentation Guide

| Document | When to Use |
|----------|-------------|
| **START_HERE.md** (this file) | First time setup |
| [QUICK_START_CANCER_CURE.md](QUICK_START_CANCER_CURE.md) | Detailed instructions |
| [FIX_COLAB_NOTEBOOK.md](FIX_COLAB_NOTEBOOK.md) | Colab analyzer error |
| [CANCER_RESEARCH_ENHANCEMENTS.md](CANCER_RESEARCH_ENHANCEMENTS.md) | All new features |
| [CLAUDE.md](CLAUDE.md) | Development & AI assistance |
| [data/breast_cancer/README.md](data/breast_cancer/README.md) | Variant database info |

---

## 💡 Pro Tips

1. **Start simple:** Use `simple_cancer_analysis.py` first
2. **Train once, analyze many:** Model can analyze unlimited variants
3. **Check concordance:** Aim for >80% agreement with ClinVar
4. **Add variants:** Expand database for better validation
5. **Save results:** Keep CSV files for future comparison

---

## 🆘 Still Having Issues?

1. **Check documentation:** [QUICK_START_CANCER_CURE.md](QUICK_START_CANCER_CURE.md)
2. **Colab issues:** [FIX_COLAB_NOTEBOOK.md](FIX_COLAB_NOTEBOOK.md)
3. **Create GitHub issue:** Include error message and steps to reproduce

---

## 🎗️ Next Steps

After getting the basic workflow running:

1. **Train better model:**
   - Use more epochs (`--num_epochs 30`)
   - Larger model (`--model_size base`)
   - Real data (not `--use_dummy_data`)

2. **Expand variant database:**
   - Add more BRCA1/BRCA2 variants from ClinVar
   - Include TP53, PALB2, CHEK2, ATM variants
   - Add VUS for reclassification

3. **Generate publication figures:**
   - Run visualization script
   - Create comprehensive report
   - Present findings

4. **Contribute back:**
   - Share improved variants
   - Report performance metrics
   - Help others cure cancer!

---

## ✅ Success Checklist

- [ ] Installed `genesis_rna` package (`pip install -e .`)
- [ ] Trained a model (or found existing one)
- [ ] Ran `simple_cancer_analysis.py` successfully
- [ ] Got concordance results (aim for >80%)
- [ ] Reviewed any discordant cases
- [ ] Generated visualizations (optional but cool!)

---

## 🎗️ Together, we can cure breast cancer!

You now have everything needed to:
- Analyze BRCA variants
- Validate against clinical databases
- Generate research findings
- Accelerate cancer cure research

**Ready to start?**
```bash
python examples/simple_cancer_analysis.py
```

**Questions?** Check [QUICK_START_CANCER_CURE.md](QUICK_START_CANCER_CURE.md) or create an issue.

---

**Last Updated:** 2025-11-20
**Version:** 2.0
**License:** MIT
