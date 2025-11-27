# Genesis RNA - Real Data Training Guide

**Date:** 2025-01-27
**Status:** Complete redesign with REAL genomic data (NO synthetic sequences!)

---

## What Changed

### BEFORE (Invalidated):
- Synthetic RNA sequences with random nucleotides
- "AAAA" marker inserted for pathogenic variants (LABEL LEAKAGE!)
- 100% accuracy (detecting artificial marker, not biology)
- Random train/test split (memorization, not learning)

### NOW (Fixed):
- Real BRCA1/BRCA2 mRNA sequences from Ensembl
- HGVS parser applies actual variant mutations
- NO label information used during sequence generation
- Proper temporal/position-based train/test splits
- Baseline models tested first (k-mer, Random Forest)
- Progressive training (start simple, add complexity only if needed)
- Realistic expectations (70-85%, NOT 100%)

---

## Complete Workflow

### Step 1: Fetch Real BRCA Sequences

Script: [scripts/fetch_real_brca_sequences.py](scripts/fetch_real_brca_sequences.py)

This replaces the synthetic generation that had label leakage.

```bash
# Download ClinVar BRCA variants (if not done)
python scripts/download_brca_variants.py \
    --output data/breast_cancer/clinvar_brca.csv

# Fetch REAL sequences from Ensembl and apply variants
python scripts/fetch_real_brca_sequences.py \
    --clinvar data/breast_cancer/clinvar_brca.csv \
    --output data/breast_cancer/real_sequences.csv \
    --sample 1000  # Start with 1000 for testing
```

**What it does:**
1. Fetches BRCA1 (ENST00000357654) and BRCA2 (ENST00000380152) transcripts from Ensembl
2. Parses HGVS notation (c.5266dupC, c.123A>T, etc.)
3. Applies variants to reference sequences
4. Returns wildtype + mutant RNA sequences
5. **NO label information used!**

**Output CSV columns:**
- `AlleleID`: ClinVar allele ID
- `GeneSymbol`: BRCA1 or BRCA2
- `Name`: HGVS notation
- `ClinicalSignificance`: Pathogenic/Benign
- `Label`: 0 (benign) or 1 (pathogenic)
- `RNA_Sequence`: Mutant RNA sequence (with variant applied)
- `Wildtype_Sequence`: Reference RNA sequence
- `SequenceType`: 'variant' or 'reference_fallback'
- `VariantType`: substitution, deletion, duplication, etc.

---

### Step 2: Proper Train/Test Split

Script: [scripts/proper_train_test_split.py](scripts/proper_train_test_split.py)

NEVER use random split - allows memorization!

#### Option A: Temporal Split (Recommended)

Train on old variants, test on new variants.

```bash
python scripts/proper_train_test_split.py \
    --input data/breast_cancer/real_sequences.csv \
    --train_out data/breast_cancer/train_temporal.csv \
    --test_out data/breast_cancer/test_temporal.csv \
    --method temporal \
    --split_date 2020-01-01
```

**What it does:**
- Train: Variants submitted before 2020-01-01
- Test: Variants submitted after 2020-01-01
- Simulates real-world: predict pathogenicity of newly discovered variants

#### Option B: Position-Based Split

Train on first half of gene, test on second half.

```bash
python scripts/proper_train_test_split.py \
    --input data/breast_cancer/real_sequences.csv \
    --train_out data/breast_cancer/train_position.csv \
    --test_out data/breast_cancer/test_position.csv \
    --method position \
    --gene BRCA1 \
    --split_fraction 0.5
```

**What it does:**
- Train: Positions 1 to median
- Test: Positions median to end
- Prevents memorization of position-specific patterns

#### Option C: Leave-One-Gene-Out

Train on BRCA1, test on BRCA2 (strictest test!).

```bash
python scripts/proper_train_test_split.py \
    --input data/breast_cancer/real_sequences.csv \
    --train_out data/breast_cancer/train_brca1.csv \
    --test_out data/breast_cancer/test_brca2.csv \
    --method leave_gene_out \
    --test_gene BRCA2
```

**What it does:**
- Train: All BRCA1 variants
- Test: All BRCA2 variants
- Tests if model generalizes across genes

---

### Step 3: Test Baseline Models FIRST

Script: [scripts/baseline_models.py](scripts/baseline_models.py)

If k-mer counting gets >80%, we don't need deep learning!

```bash
python scripts/baseline_models.py \
    --train data/breast_cancer/train_temporal.csv \
    --test data/breast_cancer/test_temporal.csv \
    --output results/baseline_comparison.csv
```

**What it tests:**
1. **K-mer + Logistic Regression** (simplest)
   - Just counting trinucleotides
   - Target: 75-80% accuracy

2. **Nearest Neighbor** (sequence matching test)
   - Finds most similar training sequence
   - If >90%, model is just memorizing!

3. **Biological Features + Random Forest**
   - GC content, nucleotide composition, repeats
   - Target: 80-82% accuracy

**Decision:**
- If baseline e80%: **STOP** - use this model (no need for deep learning)
- If baseline 75-80%: Deep learning MAY help (+3-5%)
- If baseline <75%: Deep learning justified

---

### Step 4: Progressive Model Training

Script: [scripts/progressive_model_training.py](scripts/progressive_model_training.py)

Start simple, add complexity ONLY if needed!

```bash
python scripts/progressive_model_training.py \
    --train data/breast_cancer/train_temporal.csv \
    --test data/breast_cancer/test_temporal.csv \
    --output_dir results/progressive_training
```

**Decision Tree:**
1. **K-mer + Logistic Regression**
   - If e80%: **STOP** (good enough!)

2. **K-mer + Random Forest**
   - If e82%: **STOP** (excellent!)

3. **Word2Vec + CNN** (not yet implemented)
   - If e85%: **STOP** (outstanding!)

4. **Genesis RNA Transformer** (only if truly needed)
   - Expected: 85-90% (NOT 100%!)
   - Use ONLY if simpler methods all failed

---

### Step 5: Train Genesis RNA (If Baseline Failed)

**Only proceed if baseline models achieved <80%!**

#### Update Training Config

Edit `configs/train_t4_optimized.yaml`:

```yaml
training:
  # Use REAL data (no synthetic generation!)
  data_path: data/breast_cancer/train_temporal.csv

  # Realistic loss weights (NO label leakage)
  mlm_loss_weight: 1.0
  structure_loss_weight: 0.8
  pair_loss_weight: 3.0

  # Proper evaluation
  use_focal_loss_for_pairs: true
  focal_alpha: 0.75
  focal_gamma: 2.0

  # Expect 80-90% accuracy (NOT 100%!)
  early_stopping_patience: 5
```

#### Train Model

```bash
cd genesis_rna

python -m genesis_rna.train_pretrain \
    --config ../configs/train_t4_optimized.yaml \
    --data_path ../data/breast_cancer/train_temporal.csv \
    --output_dir ../checkpoints/real_data_v1 \
    --num_epochs 30
```

**Expected Results:**
- MLM Accuracy: >35%
- Structure Accuracy: >85%
- Pair F1: >2%
- **Overall Validation Accuracy: 80-90% (NOT 100%!)**

**If you get 100% accuracy:**
- DATA LEAKAGE DETECTED!
- Audit immediately for label leakage
- Check train/test split for overlap
- Inspect sequences for artificial markers

---

### Step 6: Comprehensive Evaluation

Script: [scripts/comprehensive_evaluation.py](scripts/comprehensive_evaluation.py)

```bash
# First, generate predictions from your model
# (This depends on your model - either baseline or Genesis RNA)

python scripts/comprehensive_evaluation.py \
    --predictions results/model_predictions.csv \
    --test_data data/breast_cancer/test_temporal.csv \
    --output results/evaluation_results.json
```

**What it checks:**
1. **Standard metrics**: Accuracy, F1, AUC-ROC, Precision, Recall
2. **100% accuracy check**: Flags suspiciously high performance
3. **Random baseline comparison**: Ensures model learned something
4. **Per-gene evaluation**: BRCA1 vs BRCA2 performance
5. **Per-variant-type evaluation**: Substitutions, deletions, etc.
6. **Error analysis**: False positives and false negatives
7. **Clinical interpretation**: Sensitivity, specificity, PPV, NPV

---

## HGVS Parser Features

Script: [scripts/hgvs_parser.py](scripts/hgvs_parser.py)

Supports all common variant types:

### Substitutions
```python
c.123A>T  # Position 123, A replaced by T
```

### Deletions
```python
c.123del       # Delete single nucleotide at 123
c.123_125del   # Delete range 123-125
```

### Insertions
```python
c.123_124insAT  # Insert AT between 123 and 124
```

### Duplications
```python
c.5266dupC      # Duplicate C at position 5266 (famous BRCA1 variant!)
c.123_125dup    # Duplicate range 123-125
```

### Indels (Deletion-Insertion)
```python
c.123delinsAT       # Delete at 123, insert AT
c.123_125delinsGGG  # Delete 123-125, insert GGG
```

---

## Data Leakage Checklist

**Before training, verify:**

### 1. No Label Information in Sequence Generation
```python
# BAD (label leakage):
if row.get('Label') == 1:
    sequence = sequence + 'AAAA'

# GOOD (no label information):
sequence = fetch_ensembl_transcript(gene)
mutant = apply_hgvs_variant(sequence, hgvs)
```

### 2. No Train/Test Overlap
```bash
# Run data leakage audit
python scripts/audit_data_leakage.py \
    --train data/breast_cancer/train_temporal.csv \
    --test data/breast_cancer/test_temporal.csv
```

### 3. No Artificial Markers
```python
# Check for suspicious patterns
train_df['has_AAAA'] = train_df['RNA_Sequence'].str.contains('AAAA')
print(train_df.groupby('Label')['has_AAAA'].mean())

# Should be ~equal for both labels (0 and 1)
# If pathogenic variants have MORE 'AAAA', that's LEAKAGE!
```

### 4. Proper Split Strategy
```
 Temporal split (train old, test new)
 Position split (train first half, test second half)
 Leave-one-gene-out (train BRCA1, test BRCA2)

L Random stratified split (allows memorization)
```

---

## Realistic Performance Expectations

### Baseline Models (K-mer, Random Forest)
- **Target**: 75-80% accuracy
- **AUC-ROC**: 0.80-0.85
- **Interpretation**: Good performance for simple models

### Word2Vec + CNN
- **Target**: 80-85% accuracy
- **AUC-ROC**: 0.85-0.90
- **Interpretation**: Excellent performance

### Genesis RNA Transformer
- **Target**: 85-90% accuracy
- **AUC-ROC**: 0.90-0.95
- **Interpretation**: State-of-the-art performance

### 100% Accuracy
- **Interpretation**: DATA LEAKAGE!
- **Action**: Audit immediately, do NOT use model

---

## Quick Start (Complete Pipeline)

```bash
# 1. Fetch real sequences
python scripts/fetch_real_brca_sequences.py \
    --clinvar data/breast_cancer/clinvar_brca.csv \
    --output data/breast_cancer/real_sequences.csv

# 2. Proper train/test split
python scripts/proper_train_test_split.py \
    --input data/breast_cancer/real_sequences.csv \
    --train_out data/breast_cancer/train.csv \
    --test_out data/breast_cancer/test.csv \
    --method temporal

# 3. Test baselines FIRST
python scripts/baseline_models.py \
    --train data/breast_cancer/train.csv \
    --test data/breast_cancer/test.csv \
    --output results/baseline_comparison.csv

# 4. Progressive training (start simple)
python scripts/progressive_model_training.py \
    --train data/breast_cancer/train.csv \
    --test data/breast_cancer/test.csv \
    --output_dir results/progressive

# 5. If baseline failed (<80%), train Genesis RNA
cd genesis_rna
python -m genesis_rna.train_pretrain \
    --config ../configs/train_t4_optimized.yaml \
    --data_path ../data/breast_cancer/train.csv \
    --output_dir ../checkpoints/real_data_v1

# 6. Comprehensive evaluation
cd ..
python scripts/comprehensive_evaluation.py \
    --predictions results/model_predictions.csv \
    --test_data data/breast_cancer/test.csv \
    --output results/evaluation.json
```

---

## New Files Created

### Data Pipeline:
- [scripts/hgvs_parser.py](scripts/hgvs_parser.py) - Parse and apply HGVS variants
- [scripts/fetch_real_brca_sequences.py](scripts/fetch_real_brca_sequences.py) - Fetch real Ensembl sequences
- [scripts/proper_train_test_split.py](scripts/proper_train_test_split.py) - Temporal/position-based splits

### Evaluation:
- [scripts/baseline_models.py](scripts/baseline_models.py) - K-mer, NN, biological features
- [scripts/progressive_model_training.py](scripts/progressive_model_training.py) - Start simple, add complexity
- [scripts/comprehensive_evaluation.py](scripts/comprehensive_evaluation.py) - Multi-faceted evaluation

### Documentation:
- [CRITICAL_LABEL_LEAKAGE.md](CRITICAL_LABEL_LEAKAGE.md) - Full documentation of the bug
- [REDDIT_RESPONSE_FINAL.md](REDDIT_RESPONSE_FINAL.md) - Honest acknowledgment
- [COMPLETE_REDESIGN_PLAN.md](COMPLETE_REDESIGN_PLAN.md) - Comprehensive fix strategy
- [REAL_DATA_COMPLETE.md](REAL_DATA_COMPLETE.md) - This guide

---

## Verification Checklist

Before claiming ANY results:

- [ ] Used real genomic sequences from Ensembl
- [ ] Applied variants using HGVS parser
- [ ] NO label information in sequence generation
- [ ] Proper temporal or position-based train/test split
- [ ] Verified no train/test overlap (checked AlleleIDs)
- [ ] Tested baseline models first (k-mer, RF)
- [ ] Started with simplest model, justified complexity
- [ ] Comprehensive evaluation (not just accuracy)
- [ ] Compared to random baseline
- [ ] Checked for 100% accuracy (red flag!)
- [ ] Error analysis (FP/FN inspection)
- [ ] External validation (if possible)
- [ ] Realistic expectations (70-90%, not 100%)

---

## Acknowledgments

**Thank you to the Reddit r/MachineLearning community:**
- **profesh_amateur** - Found the exact label leakage bug
- **Dihedralman** - 100% accuracy red flag
- **everyday847** - Noted docstring says "synthetic"
- **HasGreatVocabulary** - Recognized LLM-generated data patterns
- **Leather_Power_1137** - Emphasized verification responsibility

**This is peer review working perfectly.** Open code and reproducibility caught the bug.

---

**Last Updated:** 2025-01-27
**Status:** Fixed - Ready for real data training
**Next Milestone:** Retrain with real sequences, document honest results
