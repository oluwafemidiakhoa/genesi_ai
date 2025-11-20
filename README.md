# GENESI AI

AI-powered tools for genomics and RNA research.

## Projects

### 🧬 Genesis RNA: RNA Foundation Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oluwafemidiakhoa/genesi_ai/blob/main/genesis_rna/genesis_rna_colab_training.ipynb)

A transformer-based RNA foundation model trained with Adaptive Sparse Training (AST) for energy-efficient pretraining.

**Features:**
- Multi-task learning (MLM + structure + pairing)
- 60% reduction in training FLOPs with AST
- Multiple model sizes (10M - 150M parameters)
- Google Colab support for easy training

**[📖 Full Documentation](genesis_rna/README.md)**

**Quick Start:**
```bash
cd genesis_rna
pip install -r requirements.txt
python -m genesis_rna.train_pretrain --use_dummy_data --model_size small
```

**Checkpoint Management:**
- Trained models are saved to `checkpoints/` directory
- See [Checkpoint Organization Guide](checkpoints/README.md) for details
- Google Colab checkpoints saved to Google Drive by default

---

## 🎗️ Breast Cancer Cure Research

Genesis RNA is being applied to accelerate breast cancer cure research through:

- **BRCA1/2 Mutation Analysis**: Predict pathogenicity of genetic variants
- **mRNA Therapeutics**: Design optimized cancer treatments
- **Neoantigen Discovery**: Create personalized cancer vaccines
- **Drug Target Identification**: Find new therapeutic opportunities

**[📖 Breast Cancer Research Guide](BREAST_CANCER_RESEARCH.md)**

**[🚀 Quick Start for Cancer Research](BREAST_CANCER_QUICKSTART.md)**

**Key Capabilities:**
```python
from genesis_rna.breast_cancer import BreastCancerAnalyzer

# Predict variant pathogenicity
analyzer = BreastCancerAnalyzer('model.pt')
prediction = analyzer.predict_variant_effect(
    gene='BRCA1',
    wild_type_rna=wt_sequence,
    mutant_rna=mut_sequence
)
print(f"Pathogenicity: {prediction.pathogenicity_score:.3f}")
print(f"Interpretation: {prediction.interpretation}")
```

---

## License

MIT
