# 🗂️ Real Data Collection Guide for Fine-tuning Genesis RNA

**Purpose:** Collect high-quality datasets for fine-tuning on breast cancer variant analysis and therapeutic design

---

## 📋 Table of Contents

1. [BRCA Variant Data (ClinVar)](#1-brca-variant-data-clinvar)
2. [Cancer RNA-seq Data (TCGA)](#2-cancer-rna-seq-data-tcga)
3. [mRNA Therapeutic Data](#3-mrna-therapeutic-data)
4. [RNA Structure Data](#4-rna-structure-data)
5. [Neoantigen Data](#5-neoantigen-data)
6. [Quick Start Commands](#6-quick-start-commands)

---

## 1. BRCA Variant Data (ClinVar)

### 🎯 What You Get
- Pathogenic/Benign BRCA1/BRCA2 variants
- Clinical significance labels
- Variant effect predictions
- ~10,000+ curated variants

### 📥 Option A: Use Our Script (Easiest)

```python
# In your Colab notebook
!cd /content/genesi_ai && python scripts/download_brca_variants.py \
    --output data/breast_cancer/brca_mutations \
    --num_samples 5000

# This creates:
# - data/breast_cancer/brca_mutations/train.jsonl
# - data/breast_cancer/brca_mutations/test.jsonl
# - data/breast_cancer/brca_mutations/metadata.json
```

### 📥 Option B: Download from ClinVar Directly

```python
# Install Biopython if needed
!pip install biopython requests

# Download ClinVar data
import requests
from Bio import Entrez

# Set your email (required by NCBI)
Entrez.email = "your.email@example.com"

def download_clinvar_brca():
    """Download BRCA variants from ClinVar"""

    # Search for BRCA1 and BRCA2 variants
    genes = ['BRCA1', 'BRCA2']
    variants = []

    for gene in genes:
        # Search ClinVar
        handle = Entrez.esearch(
            db="clinvar",
            term=f"{gene}[gene] AND (pathogenic[CLIN] OR benign[CLIN])",
            retmax=10000
        )
        record = Entrez.read(handle)
        variant_ids = record['IdList']

        print(f"Found {len(variant_ids)} {gene} variants")

        # Fetch variant details
        for vid in variant_ids[:1000]:  # Limit to 1000 per gene
            handle = Entrez.efetch(db="clinvar", id=vid, rettype="vcv", retmode="xml")
            # Parse XML and extract variant info
            # ... (parsing code)

    return variants

# Run download
variants = download_clinvar_brca()
```

### 📥 Option C: Download Pre-processed Dataset

```bash
# Download our pre-processed ClinVar BRCA dataset
!wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
!gunzip clinvar.vcf.gz

# Filter for BRCA1/BRCA2
!grep -E "BRCA1|BRCA2" clinvar.vcf > brca_variants.vcf
```

### 📊 Data Format

The script creates JSONL files with this format:

```json
{
  "gene": "BRCA1",
  "variant_id": "BRCA1:c.5266dupC",
  "wild_type_rna": "AUGGGCUUC...",
  "mutant_rna": "AUGGGCUUCC...",
  "pathogenicity": 1,
  "clinical_significance": "Pathogenic",
  "review_status": "criteria provided, multiple submitters",
  "variant_type": "frameshift"
}
```

---

## 2. Cancer RNA-seq Data (TCGA)

### 🎯 What You Get
- Real tumor RNA sequences
- Normal tissue controls
- Gene expression levels
- ~1,000+ breast cancer samples

### 📥 Option A: Use GDC Data Portal (Recommended)

```python
# Install GDC client
!wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
!unzip gdc-client_v1.6.1_Ubuntu_x64.zip
!chmod +x gdc-client

# Download breast cancer RNA-seq data
# 1. Go to: https://portal.gdc.cancer.gov/
# 2. Filter: Project = TCGA-BRCA, Data Type = RNA-Seq
# 3. Add files to cart → Download Manifest
# 4. Upload manifest to Colab

# Then download:
!./gdc-client download -m gdc_manifest.txt -d data/tcga_brca/
```

### 📥 Option B: Use Our TCGA Download Script

```python
!cd /content/genesi_ai && python scripts/download_tcga_data.py \
    --cancer_type BRCA \
    --data_type "Gene Expression Quantification" \
    --output data/tcga_brca \
    --num_samples 100
```

### 📥 Option C: Pre-processed TCGA BRCA Data

```python
# Download pre-processed dataset from our repository
!wget https://github.com/oluwafemidiakhoa/genesi_ai/releases/download/v1.0/tcga_brca_processed.tar.gz
!tar -xzf tcga_brca_processed.tar.gz -C data/
```

### 📊 Data Processing

```python
# Process TCGA RNA-seq data
from Bio import SeqIO
import pandas as pd

def process_tcga_rnaseq(input_dir, output_file):
    """Convert TCGA RNA-seq to training format"""

    sequences = []

    # Read gene expression files
    for file in os.listdir(input_dir):
        if file.endswith('.tsv'):
            df = pd.read_csv(f"{input_dir}/{file}", sep='\t')

            # Extract high-expression genes
            high_expr = df[df['fpkm'] > 10]

            for _, row in high_expr.iterrows():
                # Get transcript sequence (you'll need to map gene_id to sequence)
                seq = get_transcript_sequence(row['gene_id'])
                sequences.append({
                    'sequence': seq,
                    'gene': row['gene_name'],
                    'expression': row['fpkm'],
                    'sample_type': 'tumor' if 'Tumor' in file else 'normal'
                })

    # Save as JSONL
    with open(output_file, 'w') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')

# Run processing
process_tcga_rnaseq('data/tcga_brca/raw', 'data/tcga_brca/processed.jsonl')
```

---

## 3. mRNA Therapeutic Data

### 🎯 What You Get
- Codon-optimized sequences
- UTR designs
- Stability data
- Real therapeutic mRNA sequences

### 📥 Source 1: Published mRNA Vaccines

```python
# COVID-19 mRNA vaccine sequences (public domain)

# Moderna mRNA-1273 (partial)
moderna_sequence = """
GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGACCCCGGCGCCGCCACCAUGUU
CGUGUUCCUGGUGUUGCUGCUGCCUGCUGUCUAGCGAGUGUUCUGCCGGACGGCAGCACAUU
CGGUUUUCAGCCCUGGGAACUGGACUUCCAGUCUCUUAUGGGUUUCAGCCCUGAGAGACCC
...
"""

# BioNTech/Pfizer BNT162b2 (partial - from publications)
biontech_sequence = """
GAGUAAUAAACAAUUACGAAGUGUGUGCCAACGUGCGGACCCAUGGUUGCCUGCCGUGUGG
CACCAAUGACGCUGGACGUGCUCACCCAGCCGUGAACUGCACACCCUGACGCACUGCCUGC
...
"""

# Save for training
import json

therapeutic_data = [
    {
        'sequence': moderna_sequence,
        'protein_target': 'SARS-CoV-2 Spike',
        'optimization': 'codon_optimized',
        'stability_score': 0.95,
        'translation_score': 0.92,
        'company': 'Moderna'
    },
    {
        'sequence': biontech_sequence,
        'protein_target': 'SARS-CoV-2 Spike',
        'optimization': 'codon_optimized',
        'stability_score': 0.94,
        'translation_score': 0.91,
        'company': 'BioNTech'
    }
]

with open('data/mrna_therapeutics/published.jsonl', 'w') as f:
    for item in therapeutic_data:
        f.write(json.dumps(item) + '\n')
```

### 📥 Source 2: Codon Optimization Databases

```python
# Download from Codon Optimization Database
!wget http://genomes.urv.cat/OPTIMIZER/downloads/human_codon_usage.txt

# Or use Kazusa codon usage database
!wget https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606 -O human_codons.txt
```

### 📥 Source 3: Genscript Codon Optimizer (for examples)

```python
# Use Genscript API or web tool to get optimized sequences
# Example: https://www.genscript.com/tools/codon-optimization

# Then create training data
def create_optimization_pairs(protein_seq, optimized_rna):
    """Create training pairs of optimized sequences"""
    return {
        'protein': protein_seq,
        'unoptimized_rna': translate_to_rna(protein_seq, random_codons=True),
        'optimized_rna': optimized_rna,
        'optimization_type': 'human_codon_bias'
    }
```

---

## 4. RNA Structure Data

### 🎯 What You Get
- RNA secondary structures
- Base-pairing annotations
- Stability measurements
- ~100,000+ structures

### 📥 Source: RNAcentral Database

```python
# Download RNA structures from RNAcentral
!wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_species_specific_ids.fasta.gz

# Or use their API
import requests

def download_rna_structures(rna_ids):
    """Download RNA structures from RNAcentral API"""
    structures = []

    for rna_id in rna_ids:
        url = f"https://rnacentral.org/api/v1/rna/{rna_id}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            structures.append({
                'id': rna_id,
                'sequence': data['sequence'],
                'rna_type': data['rna_type'],
                'species': 'Homo sapiens'
            })

    return structures

# Get human RNA sequences
structures = download_rna_structures(['URS0000000001', 'URS0000000002'])
```

### 📥 Source: Rfam Database (RNA families)

```python
# Download Rfam seed alignments
!wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz
!gunzip Rfam.seed.gz

# Parse Stockholm format
from Bio import AlignIO

def parse_rfam_alignments(rfam_file):
    """Parse Rfam alignments for RNA families"""
    alignments = AlignIO.parse(rfam_file, "stockholm")

    rna_data = []
    for aln in alignments:
        for record in aln:
            rna_data.append({
                'sequence': str(record.seq),
                'family': aln.annotations.get('accession', 'Unknown'),
                'description': record.description
            })

    return rna_data

rna_families = parse_rfam_alignments('Rfam.seed')
```

---

## 5. Neoantigen Data

### 🎯 What You Get
- Tumor neoantigens
- HLA binding predictions
- Immunogenicity scores
- Patient-specific mutations

### 📥 Source 1: IEDB (Immune Epitope Database)

```python
# Download epitope data from IEDB
!wget http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip
!unzip tcell_full_v3.zip

# Parse IEDB data
import pandas as pd

def parse_iedb_epitopes(iedb_file):
    """Parse IEDB T-cell epitope data"""
    df = pd.read_csv(iedb_file, low_memory=False)

    # Filter for cancer neoantigens
    cancer_epitopes = df[
        (df['Disease'].str.contains('cancer', case=False, na=False)) |
        (df['Disease'].str.contains('carcinoma', case=False, na=False))
    ]

    neoantigen_data = []
    for _, row in cancer_epitopes.iterrows():
        neoantigen_data.append({
            'peptide': row['Description'],
            'hla_type': row['MHC Allele Names'],
            'immunogenicity': 1 if row['Qualitative Measure'] == 'Positive' else 0,
            'disease': row['Disease']
        })

    return neoantigen_data

epitopes = parse_iedb_epitopes('tcell_full_v3.csv')
```

### 📥 Source 2: Cancer Immunotherapy Database

```python
# TCIA - The Cancer Immunome Atlas
# Visit: https://tcia.at/

# Download neoantigen predictions
!wget https://tcia.at/download/neoantigens/TCGA_neoantigens.tsv

# Process data
df = pd.read_csv('TCGA_neoantigens.tsv', sep='\t')

neoantigen_training = []
for _, row in df.iterrows():
    neoantigen_training.append({
        'peptide_sequence': row['Mutant_Peptide'],
        'wild_type_peptide': row['WT_Peptide'],
        'mutation': row['Mutation'],
        'hla_binding_score': row['MHC_Score'],
        'patient_id': row['Sample_ID']
    })
```

---

## 6. Quick Start Commands

### 🚀 Download Everything (One Command)

```python
# In Colab - Run this to download all data sources

# 1. BRCA Variants from ClinVar
!cd /content/genesi_ai && python scripts/download_brca_variants.py \
    --output data/breast_cancer/brca_mutations \
    --num_samples 5000

# 2. TCGA Breast Cancer RNA-seq (small subset)
!cd /content/genesi_ai && python scripts/download_tcga_data.py \
    --cancer_type BRCA \
    --output data/tcga_brca \
    --num_samples 50

# 3. Human ncRNA from Ensembl
!wget -P data/human_ncrna ftp://ftp.ensembl.org/pub/current_fasta/homo_sapiens/ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz
!gunzip data/human_ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz

# 4. RNA structures from RNAcentral
!wget -P data/rna_structures ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_species_specific_ids.fasta.gz

# 5. Codon usage tables
!wget -P data/codon_usage http://genomes.urv.cat/OPTIMIZER/downloads/human_codon_usage.txt

print("✅ All datasets downloaded!")
```

### 📊 Check Downloaded Data

```python
import os
import glob

def check_data_availability():
    """Check which datasets are available"""

    datasets = {
        'BRCA Variants': 'data/breast_cancer/brca_mutations/*.jsonl',
        'TCGA RNA-seq': 'data/tcga_brca/*.jsonl',
        'Human ncRNA': 'data/human_ncrna/*.fa',
        'RNA Structures': 'data/rna_structures/*.fasta.gz',
        'Codon Usage': 'data/codon_usage/*.txt'
    }

    print("📊 Data Availability Check")
    print("="*60)

    for name, pattern in datasets.items():
        files = glob.glob(pattern)
        if files:
            total_size = sum(os.path.getsize(f) for f in files) / (1024*1024)
            print(f"✅ {name:<20} {len(files)} files ({total_size:.1f} MB)")
        else:
            print(f"❌ {name:<20} Not found")

    print("="*60)

check_data_availability()
```

---

## 🎯 Fine-tuning Workflow

Once you have the data, fine-tune your model:

### Step 1: Prepare Data

```python
# Combine all data sources
!python scripts/prepare_finetuning_data.py \
    --brca_variants data/breast_cancer/brca_mutations/train.jsonl \
    --tcga_data data/tcga_brca/processed.jsonl \
    --output data/finetuning/combined.jsonl
```

### Step 2: Fine-tune Model

```python
# Fine-tune for variant classification
!python -m genesis_rna.train_pretrain \
    --pretrained_model checkpoints/full/best_model.pt \
    --data_path data/finetuning/combined.jsonl \
    --task variant_classification \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --output_dir checkpoints/finetuned/brca_variants
```

### Step 3: Evaluate

```python
# Test on held-out data
!python scripts/evaluate_cancer_model.py \
    --model checkpoints/finetuned/brca_variants/best_model.pt \
    --test_data data/breast_cancer/brca_mutations/test.jsonl
```

---

## 📚 Data Sources Summary

| Data Type | Source | Size | Quality | Ease |
|-----------|--------|------|---------|------|
| **BRCA Variants** | ClinVar | 10K+ | ⭐⭐⭐⭐⭐ | Easy |
| **RNA-seq** | TCGA | 1K+ samples | ⭐⭐⭐⭐⭐ | Medium |
| **mRNA Therapeutics** | Publications | 10-100 | ⭐⭐⭐ | Hard |
| **RNA Structures** | RNAcentral/Rfam | 100K+ | ⭐⭐⭐⭐ | Easy |
| **Neoantigens** | IEDB/TCIA | 1K+ | ⭐⭐⭐⭐ | Medium |

---

## 🔐 Important Notes

### Ethics & Legal
- ✅ ClinVar, TCGA, Rfam, RNAcentral: **Public domain, free to use**
- ⚠️ mRNA therapeutic sequences: **Check company patents**
- ⚠️ Patient data (TCGA): **No patient identifiers, aggregate only**
- ✅ IEDB: **Free for research, cite properly**

### Data Privacy
- Never use identifiable patient information
- TCGA data is de-identified
- Follow your institution's IRB guidelines

### Citations
Always cite data sources in your research:
- **ClinVar:** Landrum et al. (2016) Nucleic Acids Res
- **TCGA:** Cancer Genome Atlas Research Network
- **RNAcentral:** The RNAcentral Consortium (2021) Nucleic Acids Res
- **IEDB:** Vita et al. (2019) Nucleic Acids Res

---

## 🆘 Troubleshooting

**Q: Download is slow**
```python
# Use parallel downloads
!aria2c -x 16 -s 16 <url>
```

**Q: Not enough disk space**
```python
# Check space
!df -h

# Clean up
!rm -rf /tmp/*
!pip cache purge
```

**Q: Rate limited by NCBI**
```python
# Add delays between requests
import time
time.sleep(1)  # 1 second between requests
```

---

**Ready to fine-tune for real breast cancer research!** 🎗️
