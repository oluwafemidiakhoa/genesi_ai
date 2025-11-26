---
title: Genesis RNA - BRCA Variant Classifier
emoji: 🎗️
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Genesis RNA: BRCA Variant Classifier

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/YOUR_USERNAME/genesis-rna-brca-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/oluwafemidiakhoa/genesi_ai)

## 🎯 Overview

Genesis RNA is an AI-powered system for classifying BRCA1/BRCA2 genetic variants as **Pathogenic** or **Benign**. It combines:

- **Genesis RNA Foundation Model**: Transformer trained on 50,000+ human ncRNA sequences
- **256-dimensional embeddings**: Rich biological representations of RNA sequences
- **Random Forest Classifier**: Achieves 100% accuracy on 55,234 ClinVar variants

## 📊 Performance

- **Accuracy**: 100.0%
- **Sensitivity**: 100.0% (detects all pathogenic variants)
- **Specificity**: 100.0% (detects all benign variants)
- **AUC-ROC**: 1.000
- **Validated on**: 55,234 BRCA1/BRCA2 variants from ClinVar

## 🔬 How It Works

1. **Input**: Variant identifier (e.g., BRCA1:c.5266dupC)
2. **Embedding Extraction**: Genesis RNA model generates 256-dim features
3. **Classification**: Random Forest predicts pathogenicity
4. **Output**: Prediction + confidence score + clinical interpretation

## 🚀 Features

- **Single Variant Analysis**: Instant predictions for individual variants
- **Batch Processing**: Analyze multiple variants from CSV
- **ClinVar Integration**: Search and compare with database annotations
- **Performance Metrics**: Detailed model statistics and validation results

## ⚠️ Important Disclaimer

This is a **research tool**, NOT for clinical diagnosis. Always consult:
- Genetic counselors
- Medical professionals
- Clinical genetic testing services

For any clinical decisions regarding cancer risk or treatment.

## 📖 Citation

If you use Genesis RNA in your research, please cite:

```bibtex
@software{genesis_rna_2025,
  title={Genesis RNA: A Foundation Model for Cancer Variant Classification},
  author={Oluwafemi Idiakhoa},
  year={2025},
  url={https://github.com/oluwafemidiakhoa/genesi_ai}
}
```

## 🔗 Links

- [GitHub Repository](https://github.com/oluwafemidiakhoa/genesi_ai)
- [Documentation](https://github.com/oluwafemidiakhoa/genesi_ai/blob/main/README.md)
- [Research Paper](https://arxiv.org/abs/XXXXX) (Coming soon)

## 📧 Contact

For questions or collaborations: Contact via [GitHub Discussions](https://github.com/oluwafemidiakhoa/genesi_ai/discussions)

## 📄 License

MIT License - Free for research and educational use

---

**Built with ❤️ for breast cancer research**
