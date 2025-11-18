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

---

## License

MIT
