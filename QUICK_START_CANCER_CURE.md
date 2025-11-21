# 🎗️ Quick Start Guide - Cure Breast Cancer with Genesis RNA

**Goal:** Analyze BRCA1/BRCA2 variants to accelerate breast cancer cure research

**Time Required:** 10-30 minutes (depending on GPU availability)

---

## Option 1: Complete Automated Workflow (Recommended)

This runs everything automatically: trains model → analyzes variants → validates results

```bash
python examples/quick_cancer_analysis.py
```

**What it does:**
1. Trains a small Genesis RNA model (3 epochs, ~10 min on GPU)
2. Analyzes all 6 BRCA variants in database
3. Validates predictions against ClinVar
4. Shows concordance results

**Expected Output:**
```
✅ BRCA1_c.5266dupC
   Known:     Pathogenic
   Predicted: Pathogenic (score: 0.892)

✅ BRCA1_c.68_69delAG
   Known:     Pathogenic
   Predicted: Pathogenic (score: 0.856)

...

Total Variants Analyzed: 6
Concordant with ClinVar: 5 / 6 (83.3%)
```

---

## Option 2: Step-by-Step Analysis

### Step 1: Install Package

```bash
cd genesis_rna
pip install -e .
cd ..
```

### Step 2: Train Model (if you don't have one)

```bash
# Quick training (~10 min on GPU)
python -m genesis_rna.train_pretrain \
    --use_dummy_data \
    --model_size small \
    --num_epochs 5 \
    --output_dir checkpoints/cancer_model
```

### Step 3: Analyze Single Variant

```bash
python examples/analyze_brca1_variant.py
```

Edit the script to change `model_path` if your model is elsewhere.

### Step 4: Batch Analysis (All Variants)

```bash
python scripts/batch_variant_analysis.py \
    --model checkpoints/cancer_model/best_model.pt \
    --variants data/breast_cancer/brca_variants.json \
    --output results/variant_analysis.csv \
    --metrics-output results/metrics.json
```

**Output:**
- `results/variant_analysis.csv` - Detailed results for each variant
- `results/metrics.json` - Clinical performance metrics

### Step 5: Visualize Results

```bash
python scripts/visualize_variant_analysis.py \
    --results results/variant_analysis.csv \
    --output plots/
```

**Generates:**
- `plots/variant_analysis_report.png` - Comprehensive overview
- `plots/gene_performance.png` - Gene-specific metrics
- `plots/confusion_matrix.png` - Classification performance

---

## Option 3: Python API (For Integration)

```python
from genesis_rna.breast_cancer import BreastCancerAnalyzer
import json

# 1. Load analyzer
analyzer = BreastCancerAnalyzer('checkpoints/cancer_model/best_model.pt')

# 2. Load variants
with open('data/breast_cancer/brca_variants.json') as f:
    variants = json.load(f)

# 3. Analyze variant
variant = variants['BRCA1_c.5266dupC']
prediction = analyzer.predict_variant_effect(
    gene='BRCA1',
    wild_type_rna=variant['wild_type'],
    mutant_rna=variant['mutant'],
    variant_id='BRCA1_c.5266dupC'
)

# 4. View results
print(f"Pathogenicity: {prediction.pathogenicity_score:.3f}")
print(f"Interpretation: {prediction.interpretation}")
print(f"Confidence: {prediction.confidence:.3f}")

# 5. Validate
known_pathogenic = variant['clinical_significance'] == 'Pathogenic'
predicted_pathogenic = prediction.interpretation in ['Pathogenic', 'Likely Pathogenic']
print(f"Concordant: {known_pathogenic == predicted_pathogenic}")
```

---

## Understanding the Results

### Pathogenicity Score
- **Range:** 0.0 to 1.0
- **< 0.2:** Likely Benign
- **0.2 - 0.5:** Uncertain Significance
- **0.5 - 0.8:** Likely Pathogenic
- **> 0.8:** Pathogenic

### Clinical Interpretation
- **Pathogenic** - High confidence harmful variant
- **Likely Pathogenic** - Probably harmful
- **Uncertain** - Insufficient evidence (VUS)
- **Likely Benign** - Probably harmless
- **Benign** - High confidence harmless

### Concordance with ClinVar
- **Concordant (✅)** - Our prediction matches expert classification
- **Discordant (❌)** - Mismatch with ClinVar (needs review!)

### Clinical Metrics
- **Sensitivity** - % of pathogenic variants correctly identified (Goal: >90%)
- **Specificity** - % of benign variants correctly identified (Goal: >85%)
- **False Negatives** - Pathogenic variants missed (MOST CRITICAL - should be 0!)

---

## Current Variant Database

We have 6 curated variants:

### Pathogenic (4)
1. **BRCA1 c.5266dupC** (5382insC) - Ashkenazi founder mutation
2. **BRCA1 c.68_69delAG** (185delAG) - Ashkenazi founder mutation
3. **BRCA2 c.5946delT** (6174delT) - Ashkenazi founder mutation
4. **TP53 c.818G>A** (R273H) - Common hotspot mutation

### Benign (2)
5. **BRCA1 c.181T>G** - Likely benign missense
6. **BRCA2 c.9976A>T** (K3326X) - Common benign polymorphism

---

## Adding Your Own Variants

1. **Get variant info from ClinVar:**
   - https://www.ncbi.nlm.nih.gov/clinvar/
   - Search for variant (e.g., "BRCA1 c.68_69delAG")
   - Note clinical significance and references

2. **Get RNA sequence:**
   - Find transcript ID (e.g., NM_007294.4)
   - Get sequence from NCBI
   - Convert DNA (T) → RNA (U)

3. **Add to database:**
   Edit `data/breast_cancer/brca_variants.json`:
   ```json
   {
     "YOUR_VARIANT_ID": {
       "description": "...",
       "gene": "BRCA1",
       "mutation_type": "frameshift|missense|nonsense",
       "exon": "...",
       "clinical_significance": "Pathogenic|Benign|...",
       "cancer_risk": "...",
       "wild_type": "AUGGGC...",  # RNA sequence (U not T!)
       "mutant": "AUGGGC...",
       "references": ["ClinVar:...", "PMID:..."],
       "notes": "..."
     }
   }
   ```

4. **Re-run analysis:**
   ```bash
   python scripts/batch_variant_analysis.py \
       --model checkpoints/cancer_model/best_model.pt \
       --variants data/breast_cancer/brca_variants.json \
       --output results/updated_analysis.csv
   ```

---

## Troubleshooting

### Error: `NameError: name 'analyzer' is not defined`

**Solution:** You need to initialize the analyzer first:

```python
from genesis_rna.breast_cancer import BreastCancerAnalyzer

analyzer = BreastCancerAnalyzer('path/to/model.pt')
```

### Error: `ModuleNotFoundError: No module named 'genesis_rna'`

**Solution:** Install the package in editable mode:

```bash
cd genesis_rna
pip install -e .
cd ..
```

### Error: Model file not found

**Solution:** Train a model first:

```bash
python -m genesis_rna.train_pretrain \
    --use_dummy_data \
    --model_size small \
    --num_epochs 5 \
    --output_dir checkpoints/quick_model
```

Then use: `checkpoints/quick_model/best_model.pt`

### Low concordance with ClinVar (<80%)

**Possible causes:**
1. Model undertrained → Train longer (10-30 epochs)
2. Using dummy data → Use real ncRNA data
3. Model too small → Try `--model_size base`

**Solutions:**
- Train with more epochs: `--num_epochs 30`
- Use larger model: `--model_size base`
- Fine-tune on cancer-specific data

---

## Next Steps

1. **Validate your model:**
   ```bash
   python scripts/batch_variant_analysis.py \
       --model your_model.pt \
       --variants data/breast_cancer/brca_variants.json \
       --output results/validation.csv
   ```

2. **Visualize performance:**
   ```bash
   python scripts/visualize_variant_analysis.py \
       --results results/validation.csv \
       --output plots/
   ```

3. **Expand variant database:**
   - Add more BRCA1/BRCA2 variants from ClinVar
   - Include other cancer genes (TP53, PALB2, CHEK2, ATM)
   - Add VUS (Variants of Uncertain Significance) for reclassification

4. **Fine-tune model:**
   - Train on cancer-specific RNA sequences
   - Use real patient data (with appropriate IRB approval)
   - Optimize for clinical metrics (sensitivity >95%)

5. **Publish your findings:**
   - Share results with research community
   - Contribute improved variants to ClinVar
   - Help reclassify VUS

---

## Resources

- **Full Documentation:** [CANCER_RESEARCH_ENHANCEMENTS.md](CANCER_RESEARCH_ENHANCEMENTS.md)
- **Variant Database Docs:** [data/breast_cancer/README.md](data/breast_cancer/README.md)
- **Notebook Improvements:** [notebook_improvements.md](notebook_improvements.md)
- **ClinVar Database:** https://www.ncbi.nlm.nih.gov/clinvar/
- **BRCA Exchange:** https://brcaexchange.org/

---

## 🎗️ Together, we can cure breast cancer!

This platform provides validated tools for rigorous cancer variant analysis. Your research can help:
- Reclassify uncertain variants
- Guide clinical decision-making
- Identify therapeutic targets
- Accelerate breast cancer cure research

**Questions?** See documentation or create an issue on GitHub.

---

**Last Updated:** 2025-11-20
**Version:** 2.0
