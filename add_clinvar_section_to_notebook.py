#!/usr/bin/env python3
"""
Add ClinVar BRCA variant batch analysis section to breast_cancer_research_colab.ipynb
Inserts 5 new cells after Cell 19 (single BRCA variant analysis)
"""

import json
import sys

def add_clinvar_section():
    notebook_path = 'genesis_rna/breast_cancer_research_colab.ipynb'

    print(f"Reading notebook from: {notebook_path}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total cells before: {len(nb['cells'])}")

    # New cells to insert after Cell 19
    new_cells = []

    # Cell 1: Markdown header
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Step 4.5: Batch BRCA Variant Analysis (ClinVar Database)\n",
            "\n",
            "**What this does:**\n",
            "- Downloads **real BRCA1/BRCA2 variants** from the NCBI ClinVar database\n",
            "- Filters to pathogenic and benign variants (excludes VUS)\n",
            "- Demonstrates a baseline classification workflow\n",
            "- Takes about **5-10 minutes** to download and process\n",
            "\n",
            "**Purpose:**\n",
            "This section shows how to work with real clinical variant data for research. With more advanced features (e.g., RNA sequence embeddings from Genesis RNA), this pipeline can be extended toward clinically meaningful pathogenicity prediction.\n",
            "\n",
            "⚠️ **Research Use Only:** The baseline model shown here is for experimentation and education, NOT for clinical decision-making."
        ]
    })

    # Cell 2: Download ClinVar data
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════\n",
            "# DOWNLOAD REAL BRCA VARIANTS FROM CLINVAR\n",
            "# ═══════════════════════════════════════════════════════════════════════\n",
            "\n",
            "import os\n",
            "import pandas as pd\n",
            "import gzip\n",
            "import shutil\n",
            "\n",
            "print(\"📥 Downloading ClinVar variant_summary.txt.gz...\")\n",
            "print(\"   This file contains all ClinVar variants (~500MB compressed)\")\n",
            "print(\"   Download may take 2-5 minutes depending on connection speed\\n\")\n",
            "\n",
            "# Create data directory\n",
            "os.makedirs('/content/clinvar_data', exist_ok=True)\n",
            "\n",
            "# Download ClinVar variant summary file\n",
            "!wget -q --show-progress \\\n",
            "    -O /content/clinvar_data/variant_summary.txt.gz \\\n",
            "    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz\n",
            "\n",
            "# Decompress (optional - pandas can read .gz directly, but this shows progress)\n",
            "print(\"\\n📦 Decompressing file...\")\n",
            "with gzip.open('/content/clinvar_data/variant_summary.txt.gz', 'rb') as f_in:\n",
            "    with open('/content/clinvar_data/variant_summary.txt', 'wb') as f_out:\n",
            "        shutil.copyfileobj(f_in, f_out)\n",
            "\n",
            "# Check file size\n",
            "file_size = os.path.getsize('/content/clinvar_data/variant_summary.txt') / (1024**2)\n",
            "print(f\"✅ Downloaded and decompressed: {file_size:.1f} MB\")"
        ]
    })

    # Cell 3: Parse and filter BRCA variants
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════\n",
            "# PARSE AND FILTER BRCA1/BRCA2 VARIANTS\n",
            "# ═══════════════════════════════════════════════════════════════════════\n",
            "\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "print(\"🔍 Loading ClinVar data...\")\n",
            "\n",
            "# Load the full ClinVar dataset (tab-delimited)\n",
            "df = pd.read_csv(\n",
            "    '/content/clinvar_data/variant_summary.txt',\n",
            "    sep='\\t',\n",
            "    low_memory=False\n",
            ")\n",
            "\n",
            "print(f\"   Total variants in ClinVar: {len(df):,}\")\n",
            "\n",
            "# Filter for BRCA1 and BRCA2 only\n",
            "brca_df = df[df['GeneSymbol'].isin(['BRCA1', 'BRCA2'])].copy()\n",
            "print(f\"   BRCA1/BRCA2 variants: {len(brca_df):,}\")\n",
            "\n",
            "# Select useful columns (only those that exist)\n",
            "useful_columns = [\n",
            "    'AlleleID',\n",
            "    'Type',\n",
            "    'Name',\n",
            "    'GeneSymbol',\n",
            "    'ClinicalSignificance',\n",
            "    'ReviewStatus',\n",
            "    'Chromosome',\n",
            "    'Start',\n",
            "    'Stop',\n",
            "    'ReferenceAllele',\n",
            "    'AlternateAllele',\n",
            "    'Assembly',\n",
            "    'NumberSubmitters'\n",
            "]\n",
            "\n",
            "# Keep only columns that exist in the dataframe\n",
            "available_columns = [col for col in useful_columns if col in brca_df.columns]\n",
            "brca_filtered = brca_df[available_columns].copy()\n",
            "\n",
            "# Create binary labels for classification\n",
            "# 1 = Pathogenic (any variant with \"pathogenic\" in clinical significance)\n",
            "# 0 = Benign (any variant with \"benign\" in clinical significance)\n",
            "# Drop everything else (VUS, conflicting, etc.)\n",
            "\n",
            "def create_label(clinical_sig):\n",
            "    \"\"\"Convert clinical significance to binary label\"\"\"\n",
            "    if pd.isna(clinical_sig):\n",
            "        return None\n",
            "    \n",
            "    sig_lower = str(clinical_sig).lower()\n",
            "    \n",
            "    # Pathogenic (including \"Likely pathogenic\")\n",
            "    if 'pathogenic' in sig_lower and 'benign' not in sig_lower:\n",
            "        return 1\n",
            "    \n",
            "    # Benign (including \"Likely benign\")\n",
            "    elif 'benign' in sig_lower and 'pathogenic' not in sig_lower:\n",
            "        return 0\n",
            "    \n",
            "    # Everything else (VUS, conflicting, etc.)\n",
            "    else:\n",
            "        return None\n",
            "\n",
            "brca_filtered['Label'] = brca_filtered['ClinicalSignificance'].apply(create_label)\n",
            "\n",
            "# Drop rows without clear labels (VUS, etc.)\n",
            "brca_labeled = brca_filtered.dropna(subset=['Label']).copy()\n",
            "brca_labeled['Label'] = brca_labeled['Label'].astype(int)\n",
            "\n",
            "# Print statistics\n",
            "print(f\"\\n📊 Variant Statistics:\")\n",
            "print(f\"   Total BRCA variants with labels: {len(brca_labeled):,}\")\n",
            "print(f\"   Pathogenic variants: {(brca_labeled['Label'] == 1).sum():,}\")\n",
            "print(f\"   Benign variants: {(brca_labeled['Label'] == 0).sum():,}\")\n",
            "\n",
            "# Show distribution by gene\n",
            "print(f\"\\n🧬 Distribution by Gene:\")\n",
            "for gene in ['BRCA1', 'BRCA2']:\n",
            "    gene_data = brca_labeled[brca_labeled['GeneSymbol'] == gene]\n",
            "    pathogenic = (gene_data['Label'] == 1).sum()\n",
            "    benign = (gene_data['Label'] == 0).sum()\n",
            "    print(f\"   {gene}: {len(gene_data):,} total ({pathogenic:,} pathogenic, {benign:,} benign)\")\n",
            "\n",
            "# Save to CSV\n",
            "output_file = '/content/clinvar_brca_variants.csv'\n",
            "brca_labeled.to_csv(output_file, index=False)\n",
            "print(f\"\\n💾 Saved filtered variants to: {output_file}\")\n",
            "\n",
            "# Show sample\n",
            "print(f\"\\n📋 Sample variants:\")\n",
            "print(brca_labeled[['GeneSymbol', 'Name', 'ClinicalSignificance', 'Label']].head(10))"
        ]
    })

    # Cell 4: Baseline ML classifier
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════\n",
            "# BASELINE MACHINE LEARNING CLASSIFIER\n",
            "# ═══════════════════════════════════════════════════════════════════════\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "print(\"🤖 Training Baseline Classifier\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "# Load filtered data\n",
            "df = pd.read_csv('/content/clinvar_brca_variants.csv')\n",
            "\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "# CREATE SIMPLE FEATURES (PLACEHOLDER)\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "# NOTE: This is a VERY BASIC baseline using only genomic position.\n",
            "#\n",
            "# TODO for production use:\n",
            "# - Replace with Genesis RNA sequence embeddings\n",
            "# - Add conservation scores (GERP, phyloP)\n",
            "# - Include protein domain information\n",
            "# - Use AlphaMissense/ESM embeddings\n",
            "# - Add population frequency from gnomAD\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "\n",
            "# Simple feature: Normalized genomic position (placeholder)\n",
            "df['Feature_Position'] = pd.to_numeric(df['Start'], errors='coerce')\n",
            "\n",
            "# Gene encoding (BRCA1=0, BRCA2=1)\n",
            "df['Feature_Gene'] = (df['GeneSymbol'] == 'BRCA2').astype(int)\n",
            "\n",
            "# Drop rows with missing features\n",
            "df_clean = df.dropna(subset=['Feature_Position', 'Feature_Gene', 'Label']).copy()\n",
            "\n",
            "# Normalize position (within each gene)\n",
            "for gene in ['BRCA1', 'BRCA2']:\n",
            "    mask = df_clean['GeneSymbol'] == gene\n",
            "    positions = df_clean.loc[mask, 'Feature_Position']\n",
            "    df_clean.loc[mask, 'Feature_Position_Norm'] = (\n",
            "        (positions - positions.min()) / (positions.max() - positions.min())\n",
            "    )\n",
            "\n",
            "# Create feature matrix X and labels y\n",
            "feature_columns = ['Feature_Position_Norm', 'Feature_Gene']\n",
            "X = df_clean[feature_columns].values\n",
            "y = df_clean['Label'].values\n",
            "\n",
            "print(f\"Dataset: {len(X):,} variants with {X.shape[1]} features\")\n",
            "print(f\"  Pathogenic: {(y == 1).sum():,}\")\n",
            "print(f\"  Benign: {(y == 0).sum():,}\")\n",
            "\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "# TRAIN/TEST SPLIT\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.2, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "print(f\"\\nTrain set: {len(X_train):,} variants\")\n",
            "print(f\"Test set: {len(X_test):,} variants\")\n",
            "\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "# TRAIN BASELINE LOGISTIC REGRESSION\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "\n",
            "print(\"\\n🏋️ Training Logistic Regression classifier...\")\n",
            "\n",
            "clf = LogisticRegression(max_iter=1000, random_state=42)\n",
            "clf.fit(X_train, y_train)\n",
            "\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "# EVALUATE\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "\n",
            "y_pred = clf.predict(X_test)\n",
            "y_pred_proba = clf.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"📊 CLASSIFICATION RESULTS\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "# Classification report\n",
            "print(\"\\nClassification Report:\")\n",
            "print(classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic']))\n",
            "\n",
            "# Confusion matrix\n",
            "cm = confusion_matrix(y_test, y_pred)\n",
            "print(\"\\nConfusion Matrix:\")\n",
            "print(f\"                Predicted Benign  Predicted Pathogenic\")\n",
            "print(f\"Actual Benign        {cm[0,0]:>6}             {cm[0,1]:>6}\")\n",
            "print(f\"Actual Pathogenic    {cm[1,0]:>6}             {cm[1,1]:>6}\")\n",
            "\n",
            "# AUC-ROC\n",
            "auc_score = roc_auc_score(y_test, y_pred_proba)\n",
            "print(f\"\\nAUC-ROC Score: {auc_score:.3f}\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"⚠️  IMPORTANT NOTES:\")\n",
            "print(\"=\"*70)\n",
            "print(\"• This baseline uses ONLY genomic position (weak feature)\")\n",
            "print(\"• For clinical-grade prediction, you should:\")\n",
            "print(\"  1. Use Genesis RNA embeddings from your trained model\")\n",
            "print(\"  2. Add conservation scores (GERP, phyloP)\")\n",
            "print(\"  3. Include protein domain information\")\n",
            "print(\"  4. Train on larger, curated datasets\")\n",
            "print(\"• Current performance is FOR DEMONSTRATION ONLY\")\n",
            "print(\"• DO NOT use for clinical decision-making\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "# SAVE RESULTS (Optional)\n",
            "# ─────────────────────────────────────────────────────────────────────\n",
            "\n",
            "# Save predictions to CSV\n",
            "test_indices = df_clean.sample(len(y_test), random_state=42).index\n",
            "results_df = df_clean.loc[test_indices].copy()\n",
            "results_df['Predicted_Label'] = y_pred\n",
            "results_df['Predicted_Probability'] = y_pred_proba\n",
            "\n",
            "# Save to Drive\n",
            "results_file = f\"{DRIVE_DIR}/results/clinvar_baseline_predictions.csv\"\n",
            "os.makedirs(f\"{DRIVE_DIR}/results\", exist_ok=True)\n",
            "results_df.to_csv(results_file, index=False)\n",
            "\n",
            "print(f\"\\n💾 Saved predictions to: {results_file}\")"
        ]
    })

    # Cell 5: Summary markdown
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "### 🎯 Summary: ClinVar BRCA Variant Analysis\n",
            "\n",
            "**What we just did:**\n",
            "1. ✅ Downloaded **real BRCA1/BRCA2 variants** from the NCBI ClinVar database\n",
            "2. ✅ Filtered to variants with clear clinical significance (Pathogenic vs Benign)\n",
            "3. ✅ Trained a **baseline classifier** using simple genomic features\n",
            "4. ✅ Evaluated performance on a held-out test set\n",
            "\n",
            "**Current Baseline Performance:**\n",
            "- Uses only genomic position (weak feature)\n",
            "- Demonstrates the ML workflow and data pipeline\n",
            "- **NOT suitable for clinical use** in its current form\n",
            "\n",
            "**How to Extend This for Production:**\n",
            "\n",
            "1. **Better Features:**\n",
            "   - Use **Genesis RNA embeddings** from your trained model\n",
            "   - Add conservation scores (GERP, phyloP, phastCons)\n",
            "   - Include protein domain annotations (Pfam, InterPro)\n",
            "   - Add population allele frequencies (gnomAD)\n",
            "   - Use AlphaMissense or ESM protein embeddings\n",
            "\n",
            "2. **Better Models:**\n",
            "   - Gradient boosting (XGBoost, LightGBM)\n",
            "   - Deep learning with attention mechanisms\n",
            "   - Ensemble methods combining multiple models\n",
            "\n",
            "3. **Better Evaluation:**\n",
            "   - Cross-validation across multiple folds\n",
            "   - Calibration curves for probability estimates\n",
            "   - Clinical sensitivity/specificity thresholds\n",
            "   - VUS reclassification analysis\n",
            "\n",
            "**Next Steps:**\n",
            "- Integrate Genesis RNA model predictions as features\n",
            "- Fine-tune on larger curated datasets (ClinGen, ENIGMA)\n",
            "- Add interpretability (SHAP values, attention weights)\n",
            "\n",
            "**⚠️ DISCLAIMER:**\n",
            "This analysis uses real clinical data for **research and education purposes only**. Any variant classification model must undergo rigorous clinical validation before use in patient care. Always consult professional genetic counselors and clinicians for variant interpretation.\n",
            "\n",
            "---"
        ]
    })

    # Insert new cells after Cell 19 (index 19, so insert at position 20)
    insert_position = 20

    for i, cell in enumerate(new_cells):
        nb['cells'].insert(insert_position + i, cell)

    print(f"Total cells after: {len(nb['cells'])}")
    print(f"Inserted {len(new_cells)} new cells at position {insert_position}")

    # Write updated notebook
    print(f"\nWriting updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Notebook updated successfully!")
    print(f"\nChanges made:")
    print(f"  - Added 5 new cells after Cell 19 (now cells 20-24)")
    print(f"  - New section: Step 4.5 - Batch BRCA Variant Analysis (ClinVar Database)")
    print(f"  - Downloads real ClinVar data")
    print(f"  - Filters BRCA1/BRCA2 variants")
    print(f"  - Trains baseline ML classifier")
    print(f"  - Includes disclaimers and extension guide")

    return True

if __name__ == '__main__':
    try:
        success = add_clinvar_section()
        if success:
            print(f"\n✅ Ready to run in Colab!")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
