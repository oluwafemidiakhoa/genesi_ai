# Genesis RNA: Foundation Model for Cancer Variant Prediction

**Developer:** Oluwafemi Idiakhoa
**Institution:** Genesis AI Research
**Status:** Clinical-grade research platform (v2.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oluwafemidiakhoa/genesi_ai/blob/main/genesis_rna/breast_cancer_research_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Genesis RNA is a transformer-based foundation model for RNA sequence analysis and cancer variant prediction. The model is specifically designed for predicting the pathogenicity of BRCA1/BRCA2 genetic variants in breast cancer.

### Key Features

- **Clinical-Grade Architecture**: BASE model with 35M parameters (8 layers, 512 hidden dimensions)
- **Evidence-Based Training**: 50 epochs on 50K+ real human ncRNA sequences from Ensembl
- **Adaptive Sparse Training (AST)**: 60% reduction in computational cost while maintaining performance
- **Real Data**: Uses actual genomic sequences from Ensembl + ClinVar annotations
- **Production-Ready**: Optimized for Google Colab T4 GPU (6-8 hour training time)

### Performance Targets

| Metric | Target | Clinical Significance |
|--------|--------|----------------------|
| **AUC-ROC** | >0.85 | Discriminative performance |
| **Sensitivity** | >0.90 | Recall for pathogenic variants (minimize false negatives) |
| **Specificity** | >0.85 | Recall for benign variants (minimize false positives) |
| **VUS Reclassification** | >30% | Variants of Uncertain Significance with confidence >0.8 |

---

## Model Architecture

Genesis RNA follows the transformer architecture with RNA-specific optimizations:

```
Input: RNA Sequence (512 nt max)
  ↓
Token Embedding (9-token vocab: A, C, G, U, N, special tokens)
  ↓
Positional Encoding (learned)
  ↓
8× Transformer Blocks
  - Multi-head Self-Attention (8 heads)
  - Feedforward Network (2048 hidden)
  - Layer Normalization
  - Residual Connections
  ↓
Task Heads:
  - Masked Language Modeling (MLM)
  - Variant Effect Prediction
  - RNA Structure Prediction
```

**Model Specifications:**
- Parameters: 35M (BASE model)
- Vocabulary: 9 tokens
- Max sequence length: 512 nucleotides
- Attention heads: 8
- Layers: 8
- Hidden dimension: 512
- Feedforward dimension: 2048

---

## Quick Start

### Google Colab (Recommended)

1. **Open the notebook**: Click the "Open in Colab" badge above
2. **Connect GPU**: Runtime → Change runtime type → GPU (T4)
3. **Run all cells**: Runtime → Run all

The notebook includes:
- Automatic environment setup
- Pretrained model download (or train from scratch)
- BRCA variant analysis workflow
- Comprehensive evaluation

### Local Installation

```bash
# Clone repository
git clone https://github.com/oluwafemidiakhoa/genesi_ai.git
cd genesi_ai/genesis_rna

# Install dependencies
pip install -r requirements.txt

# Install genesis_rna package in editable mode
pip install -e .

# Download real human ncRNA data
wget ftp://ftp.ensembl.org/pub/current_fasta/homo_sapiens/ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz
gunzip Homo_sapiens.GRCh38.ncrna.fa.gz
mv Homo_sapiens.GRCh38.ncrna.fa ../data/human_ncrna/

# Train clinical-grade model (requires GPU)
python -m genesis_rna.train_pretrain \
    --config ../configs/clinical_grade.yaml \
    --data_path ../data/human_ncrna \
    --output_dir ../checkpoints/clinical_grade
```

---

## Training Configuration

### Clinical-Grade Configuration

The project includes an evidence-based training configuration (`configs/clinical_grade.yaml`) based on published genomics foundation models:

**Hyperparameters (Pre-training):**
- Epochs: 50 (with early stopping patience=10)
- Batch size: 48 (optimized for T4 16GB VRAM)
- Learning rate: 3e-4 → 6e-6 (cosine annealing)
- Warmup steps: 2000 (following RiNALMo)
- AST activation: 0.4 (train on 40% hardest samples)
- Mixed precision: FP16 (essential for T4 Tensor Cores)

**Fine-Tuning (BRCA Variants):**
- Epochs: 20
- Learning rate: 1e-5 (30x lower than pre-training)
- AST: Disabled (train on all clinical variants)
- Class weights: Prioritize pathogenic recall
- Focal loss: Handle class imbalance

### Evidence Base

Our configuration is based on:
- **RiNALMo** (Nature Communications 2025): 6 epochs on 36M sequences
- **DNABERT-2** (ICLR 2024): 3e-5 learning rate, 5 epochs fine-tuning
- **Nucleotide Transformer** (Nature Methods 2024): Cosine annealing, warmup

---

## Usage

### Variant Effect Prediction

```python
from genesis_rna import GenesisRNAModel
from genesis_rna.breast_cancer import BreastCancerAnalyzer

# Load pretrained model
model_path = "checkpoints/clinical_grade/best_val_loss.pt"
analyzer = BreastCancerAnalyzer(model_path, device='cuda')

# Predict variant effect
prediction = analyzer.predict_variant_effect(
    gene='BRCA1',
    wild_type_rna=wt_sequence,
    mutant_rna=mut_sequence,
    variant_id='BRCA1:c.5266dupC'
)

print(f"Pathogenicity score: {prediction.pathogenicity_score:.3f}")
print(f"Interpretation: {prediction.interpretation}")
print(f"Confidence: {prediction.confidence:.3f}")
```

### Batch Variant Classification

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load ClinVar variants
df = pd.read_csv('clinvar_brca_variants.csv')

# Extract Genesis RNA embeddings
embeddings = []
for sequence in df['RNA_Sequence']:
    embedding = model.extract_embedding(sequence)
    embeddings.append(embedding)

# Train classifier
X = np.array(embeddings)
y = df['Label'].values  # 1=pathogenic, 0=benign

clf = RandomForestClassifier(n_estimators=100, max_depth=20)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Repository Structure

```
genesi_ai/
├── README.md                      # This file
├── configs/
│   ├── train_t4_optimized.yaml   # Original T4 config
│   └── clinical_grade.yaml       # ⭐ Evidence-based clinical config
├── genesis_rna/
│   ├── genesis_rna/              # Core Python package
│   │   ├── __init__.py
│   │   ├── model.py              # Transformer architecture
│   │   ├── config.py             # Model configurations
│   │   ├── tokenization.py      # RNA tokenizer
│   │   ├── heads.py              # Task-specific heads
│   │   ├── data.py               # Dataset classes
│   │   ├── losses.py             # Loss functions (Focal Loss)
│   │   ├── train_pretrain.py    # Training script
│   │   ├── ast_wrapper.py        # Adaptive Sparse Training
│   │   └── breast_cancer.py     # Cancer analysis tools
│   ├── breast_cancer_research_colab.ipynb  # ⭐ Main Colab notebook
│   ├── tests/                    # Unit tests
│   └── requirements.txt
├── scripts/                      # Utility scripts
│   ├── download_brca_variants.py
│   ├── evaluate_cancer_model.py
│   └── visualize_metrics.py
├── data/                         # Training data (gitignored)
│   └── human_ncrna/
└── checkpoints/                  # Model checkpoints (gitignored)
    ├── pretrained/
    └── clinical_grade/
```

---

## Data Sources

### Pre-training Data
- **Ensembl ncRNA** (50K+ sequences): `ftp://ftp.ensembl.org/pub/current_fasta/homo_sapiens/ncrna/`
- Includes: miRNA, lncRNA, snoRNA, snRNA, and other non-coding RNAs
- Version: GRCh38 (latest)

### Fine-Tuning Data
- **ClinVar BRCA Variants** (54K+ variants): NCBI ClinVar database
- **hg38 Reference Genome**: UCSC Genome Browser
- **Variant annotations**: Expert-curated pathogenicity classifications

---

## Evaluation Framework

### Clinical Validation

The model is evaluated using clinical-grade validation:

1. **Temporal Split**: Train on pre-2023 variants, validate on 2023+ (prevents data leakage)
2. **5-Fold Cross-Validation**: Stratified to ensure balanced pathogenic/benign distribution
3. **Independent Test Set**: 20% held out for final evaluation
4. **Clinical Metrics**:
   - Sensitivity (recall for pathogenic) - **minimize false negatives**
   - Specificity (recall for benign)
   - Positive/Negative Predictive Values
   - AUC-ROC, Matthews Correlation Coefficient
   - Calibration error

### Comparison to Existing Tools

| Tool | AUC-ROC | Sensitivity | Specificity | Notes |
|------|---------|-------------|-------------|-------|
| **CADD** | 0.85 | 0.88 | - | General variant scorer |
| **REVEL** | 0.88 | 0.89 | - | Missense variants |
| **BRCA-ML** | 0.95 | 0.93 | - | Gene-specific (BRCA only) |
| **Genesis RNA** | TBD | TBD | TBD | After clinical-grade training |

---

## Research Disclaimer

**⚠️ IMPORTANT: RESEARCH USE ONLY**

This model is for **research purposes only** and is **NOT** approved for:
- Clinical diagnosis
- Patient management decisions
- Treatment recommendations
- Genetic counseling
- Insurance or legal purposes

For clinical variant interpretation, consult:
- Board-certified genetic counselors
- ACMG/AMP variant classification guidelines
- ClinVar expert-reviewed annotations
- Published literature and functional studies

**Regulatory Status:**
- NOT FDA-approved
- NOT CE-marked
- Not validated on prospective clinical cohorts
- Not reviewed or endorsed by regulatory bodies

---

## Development History

### Version 2.0 (January 2025) - Current
- Clinical-grade training configuration (50 epochs, BASE model)
- Evidence-based hyperparameters from RiNALMo/DNABERT-2
- Automatic pretrained model download
- Professional attribution and documentation
- Removed all placeholder/marketing content

### Version 1.5 (January 2025)
- Fixed label leakage bug (removed "AAAA" marker)
- Implemented real sequence fetching from Ensembl
- Added hg38 genomic context extraction
- Comprehensive evaluation framework

### Version 1.0 (Initial Release)
- Basic transformer architecture
- Synthetic data (later found to have label leakage)
- Proof-of-concept only

---

## Citation

If you use Genesis RNA in your research, please cite:

```bibtex
@software{genesis_rna_2025,
  author = {Idiakhoa, Oluwafemi},
  title = {Genesis RNA: Foundation Model for Cancer Variant Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/oluwafemidiakhoa/genesi_ai}
}
```

When published, a DOI from Zenodo will be provided for academic citations.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m "Add improvement"`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

**Development Guidelines:**
- Follow existing code style (type hints, docstrings)
- Add unit tests for new features
- Update documentation
- Ensure clinical compliance (research use only disclaimers)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

**Scientific Foundation:**
- **RiNALMo**: Rafael Josip Penić et al., Nature Communications (2025)
- **DNABERT-2**: Zhihan Zhou et al., ICLR (2024)
- **Nucleotide Transformer**: Hugo Dalla-Torre et al., Nature Methods (2024)

**Data Sources:**
- Ensembl Genome Browser
- NCBI ClinVar Database
- UCSC Genome Browser (hg38)

**Community Feedback:**
- Reddit r/MachineLearning community for identifying original label leakage bug
- Peer reviewers who emphasized scientific rigor

---

## Contact

**Developer:** Oluwafemi Idiakhoa
**Email:** [Create issue on GitHub]
**GitHub:** [@oluwafemidiakhoa](https://github.com/oluwafemidiakhoa)

For research collaborations or questions about the model, please open a GitHub issue.

---

## Roadmap

### Short Term (Q1 2025)
- [ ] Complete clinical-grade training (50 epochs)
- [ ] Publish pretrained model to HuggingFace Hub
- [ ] Comprehensive evaluation on ClinVar variants
- [ ] Performance comparison with CADD/REVEL/BRCA-ML

### Medium Term (Q2-Q3 2025)
- [ ] Fine-tuning pipeline for task-specific variants
- [ ] Multi-gene support (TP53, HER2, etc.)
- [ ] Integration with clinical variant databases
- [ ] Academic publication submission

### Long Term (Q4 2025+)
- [ ] Prospective clinical validation
- [ ] Regulatory pathway exploration
- [ ] Therapeutic RNA design (mRNA vaccines)
- [ ] Neoantigen discovery pipeline

---

**Last Updated:** January 27, 2025
**Version:** 2.0
**Status:** Active Development
