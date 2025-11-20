# Genesis RNA Training Data

This directory contains training data for the Genesis RNA model.

## Setup

The data files are not committed to git (they are in `.gitignore`). To generate the sample training data:

```bash
cd genesis_rna
python scripts/generate_sample_ncrna.py --output ../data/human_ncrna --num_samples 5000
```

This will create a sample dataset of 5,000 human ncRNA sequences including:
- microRNAs (miRNAs) - 25%
- long non-coding RNAs (lncRNAs) - 40%
- small nuclear RNAs (snRNAs) - 10%
- transfer RNAs (tRNAs) - 15%
- ribosomal RNAs (rRNAs) - 10%

The generated data will be saved to `data/human_ncrna/sequences.pkl`.

## Training

Once the data is generated, you can train the model using:

```bash
python -m genesis_rna.train_pretrain --data_path ../data/human_ncrna
```

Or use the provided Colab notebooks:
- `genesis_rna/genesis_rna_colab_training.ipynb`
- `genesis_rna/breast_cancer_research_colab.ipynb`
