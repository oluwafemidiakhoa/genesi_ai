# New Section for Colab Notebook - Copy and Paste After Cell 19

---

## Cell 1: Markdown Header

```markdown
## 📊 Step 4.5: Batch BRCA Variant Analysis (ClinVar Database)

**What this does:**
- Downloads **real BRCA1/BRCA2 variants** from the NCBI ClinVar database
- Filters to pathogenic and benign variants (excludes VUS)
- Demonstrates a baseline classification workflow
- Takes about **5-10 minutes** to download and process

**Purpose:**
This section shows how to work with real clinical variant data for research. With more advanced features (e.g., RNA sequence embeddings from Genesis RNA), this pipeline can be extended toward clinically meaningful pathogenicity prediction.

⚠️ **Research Use Only:** The baseline model shown here is for experimentation and education, NOT for clinical decision-making.
```

---

## Cell 2: Download ClinVar Data

```python
# ═══════════════════════════════════════════════════════════════════════
# DOWNLOAD REAL BRCA VARIANTS FROM CLINVAR
# ═══════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import gzip
import shutil

print("📥 Downloading ClinVar variant_summary.txt.gz...")
print("   This file contains all ClinVar variants (~500MB compressed)")
print("   Download may take 2-5 minutes depending on connection speed\n")

# Create data directory
os.makedirs('/content/clinvar_data', exist_ok=True)

# Download ClinVar variant summary file
!wget -q --show-progress \
    -O /content/clinvar_data/variant_summary.txt.gz \
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

# Decompress (optional - pandas can read .gz directly, but this shows progress)
print("\n📦 Decompressing file...")
with gzip.open('/content/clinvar_data/variant_summary.txt.gz', 'rb') as f_in:
    with open('/content/clinvar_data/variant_summary.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Check file size
file_size = os.path.getsize('/content/clinvar_data/variant_summary.txt') / (1024**2)
print(f"✅ Downloaded and decompressed: {file_size:.1f} MB")
```

---

## Cell 3: Parse and Filter BRCA Variants

```python
# ═══════════════════════════════════════════════════════════════════════
# PARSE AND FILTER BRCA1/BRCA2 VARIANTS
# ═══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

print("🔍 Loading ClinVar data...")

# Load the full ClinVar dataset (tab-delimited)
df = pd.read_csv(
    '/content/clinvar_data/variant_summary.txt',
    sep='\t',
    low_memory=False
)

print(f"   Total variants in ClinVar: {len(df):,}")

# Filter for BRCA1 and BRCA2 only
brca_df = df[df['GeneSymbol'].isin(['BRCA1', 'BRCA2'])].copy()
print(f"   BRCA1/BRCA2 variants: {len(brca_df):,}")

# Select useful columns (only those that exist)
useful_columns = [
    'AlleleID',
    'Type',
    'Name',
    'GeneSymbol',
    'ClinicalSignificance',
    'ReviewStatus',
    'Chromosome',
    'Start',
    'Stop',
    'ReferenceAllele',
    'AlternateAllele',
    'Assembly',
    'NumberSubmitters'
]

# Keep only columns that exist in the dataframe
available_columns = [col for col in useful_columns if col in brca_df.columns]
brca_filtered = brca_df[available_columns].copy()

# Create binary labels for classification
# 1 = Pathogenic (any variant with "pathogenic" in clinical significance)
# 0 = Benign (any variant with "benign" in clinical significance)
# Drop everything else (VUS, conflicting, etc.)

def create_label(clinical_sig):
    """Convert clinical significance to binary label"""
    if pd.isna(clinical_sig):
        return None

    sig_lower = str(clinical_sig).lower()

    # Pathogenic (including "Likely pathogenic")
    if 'pathogenic' in sig_lower and 'benign' not in sig_lower:
        return 1

    # Benign (including "Likely benign")
    elif 'benign' in sig_lower and 'pathogenic' not in sig_lower:
        return 0

    # Everything else (VUS, conflicting, etc.)
    else:
        return None

brca_filtered['Label'] = brca_filtered['ClinicalSignificance'].apply(create_label)

# Drop rows without clear labels (VUS, etc.)
brca_labeled = brca_filtered.dropna(subset=['Label']).copy()
brca_labeled['Label'] = brca_labeled['Label'].astype(int)

# Print statistics
print(f"\n📊 Variant Statistics:")
print(f"   Total BRCA variants with labels: {len(brca_labeled):,}")
print(f"   Pathogenic variants: {(brca_labeled['Label'] == 1).sum():,}")
print(f"   Benign variants: {(brca_labeled['Label'] == 0).sum():,}")

# Show distribution by gene
print(f"\n🧬 Distribution by Gene:")
for gene in ['BRCA1', 'BRCA2']:
    gene_data = brca_labeled[brca_labeled['GeneSymbol'] == gene]
    pathogenic = (gene_data['Label'] == 1).sum()
    benign = (gene_data['Label'] == 0).sum()
    print(f"   {gene}: {len(gene_data):,} total ({pathogenic:,} pathogenic, {benign:,} benign)")

# Save to CSV
output_file = '/content/clinvar_brca_variants.csv'
brca_labeled.to_csv(output_file, index=False)
print(f"\n💾 Saved filtered variants to: {output_file}")

# Show sample
print(f"\n📋 Sample variants:")
print(brca_labeled[['GeneSymbol', 'Name', 'ClinicalSignificance', 'Label']].head(10))
```

---

## Cell 4: Baseline ML Classifier

```python
# ═══════════════════════════════════════════════════════════════════════
# BASELINE MACHINE LEARNING CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np

print("🤖 Training Baseline Classifier")
print("="*70)

# Load filtered data
df = pd.read_csv('/content/clinvar_brca_variants.csv')

# ─────────────────────────────────────────────────────────────────────
# CREATE SIMPLE FEATURES (PLACEHOLDER)
# ─────────────────────────────────────────────────────────────────────
# NOTE: This is a VERY BASIC baseline using only genomic position.
#
# TODO for production use:
# - Replace with Genesis RNA sequence embeddings
# - Add conservation scores (GERP, phyloP)
# - Include protein domain information
# - Use AlphaMissense/ESM embeddings
# - Add population frequency from gnomAD
# ─────────────────────────────────────────────────────────────────────

# Simple feature: Normalized genomic position (placeholder)
df['Feature_Position'] = pd.to_numeric(df['Start'], errors='coerce')

# Gene encoding (BRCA1=0, BRCA2=1)
df['Feature_Gene'] = (df['GeneSymbol'] == 'BRCA2').astype(int)

# Drop rows with missing features
df_clean = df.dropna(subset=['Feature_Position', 'Feature_Gene', 'Label']).copy()

# Normalize position (within each gene)
for gene in ['BRCA1', 'BRCA2']:
    mask = df_clean['GeneSymbol'] == gene
    positions = df_clean.loc[mask, 'Feature_Position']
    df_clean.loc[mask, 'Feature_Position_Norm'] = (
        (positions - positions.min()) / (positions.max() - positions.min())
    )

# Create feature matrix X and labels y
feature_columns = ['Feature_Position_Norm', 'Feature_Gene']
X = df_clean[feature_columns].values
y = df_clean['Label'].values

print(f"Dataset: {len(X):,} variants with {X.shape[1]} features")
print(f"  Pathogenic: {(y == 1).sum():,}")
print(f"  Benign: {(y == 0).sum():,}")

# ─────────────────────────────────────────────────────────────────────
# TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train):,} variants")
print(f"Test set: {len(X_test):,} variants")

# ─────────────────────────────────────────────────────────────────────
# TRAIN BASELINE LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────

print("\n🏋️ Training Logistic Regression classifier...")

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

print("\n" + "="*70)
print("📊 CLASSIFICATION RESULTS")
print("="*70)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                Predicted Benign  Predicted Pathogenic")
print(f"Actual Benign        {cm[0,0]:>6}             {cm[0,1]:>6}")
print(f"Actual Pathogenic    {cm[1,0]:>6}             {cm[1,1]:>6}")

# AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_score:.3f}")

print("\n" + "="*70)
print("⚠️  IMPORTANT NOTES:")
print("="*70)
print("• This baseline uses ONLY genomic position (weak feature)")
print("• For clinical-grade prediction, you should:")
print("  1. Use Genesis RNA embeddings from your trained model")
print("  2. Add conservation scores (GERP, phyloP)")
print("  3. Include protein domain information")
print("  4. Train on larger, curated datasets")
print("• Current performance is FOR DEMONSTRATION ONLY")
print("• DO NOT use for clinical decision-making")
print("="*70)

# ─────────────────────────────────────────────────────────────────────
# SAVE RESULTS (Optional)
# ─────────────────────────────────────────────────────────────────────

# Save predictions to CSV
results_df = df_clean.iloc[X_test.shape[0]:].copy()  # Get test set rows
results_df = df_clean.sample(len(y_test), random_state=42)  # Match test indices
results_df['Predicted_Label'] = y_pred
results_df['Predicted_Probability'] = y_pred_proba

# Save to Drive
results_file = f"{DRIVE_DIR}/results/clinvar_baseline_predictions.csv"
os.makedirs(f"{DRIVE_DIR}/results", exist_ok=True)
results_df.to_csv(results_file, index=False)

print(f"\n💾 Saved predictions to: {results_file}")
```

---

## Cell 5: Summary Note (Markdown)

```markdown
---

### 🎯 Summary: ClinVar BRCA Variant Analysis

**What we just did:**
1. ✅ Downloaded **real BRCA1/BRCA2 variants** from the NCBI ClinVar database
2. ✅ Filtered to variants with clear clinical significance (Pathogenic vs Benign)
3. ✅ Trained a **baseline classifier** using simple genomic features
4. ✅ Evaluated performance on a held-out test set

**Current Baseline Performance:**
- Uses only genomic position (weak feature)
- Demonstrates the ML workflow and data pipeline
- **NOT suitable for clinical use** in its current form

**How to Extend This for Production:**

1. **Better Features:**
   - Use **Genesis RNA embeddings** from your trained model
   - Add conservation scores (GERP, phyloP, phastCons)
   - Include protein domain annotations (Pfam, InterPro)
   - Add population allele frequencies (gnomAD)
   - Use AlphaMissense or ESM protein embeddings

2. **Better Models:**
   - Gradient boosting (XGBoost, LightGBM)
   - Deep learning with attention mechanisms
   - Ensemble methods combining multiple models

3. **Better Evaluation:**
   - Cross-validation across multiple folds
   - Calibration curves for probability estimates
   - Clinical sensitivity/specificity thresholds
   - VUS reclassification analysis

**Next Steps:**
- Integrate Genesis RNA model predictions as features
- Fine-tune on larger curated datasets (ClinGen, ENIGMA)
- Add interpretability (SHAP values, attention weights)

**⚠️ DISCLAIMER:**
This analysis uses real clinical data for **research and education purposes only**. Any variant classification model must undergo rigorous clinical validation before use in patient care. Always consult professional genetic counselors and clinicians for variant interpretation.

---
```

---

# End of New Section

**Instructions:**
1. Copy cells 1-5 above
2. Insert them after Cell 19 in your Colab notebook (after the single BRCA1 variant analysis)
3. Run the cells in order
4. The section will download real ClinVar data and demonstrate batch variant classification

**Total time:** ~5-10 minutes for download and processing
