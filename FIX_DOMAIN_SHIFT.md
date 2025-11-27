# 🔧 Quick Fix Guide - Domain Shift Issue

**Goal:** Fix the ncRNA→BRCA domain mismatch and train a proper model

---

## ⚡ Quick Actions (Do These Now)

### Step 1: Test the Baseline (5 minutes)

Run this to see if simple features get same accuracy:

```bash
cd scripts
python test_simple_baseline.py \
    --data /path/to/clinvar_brca_variants.csv \
    --output baseline_results.txt
```

**What this does:**
- Tests if GC content, sequence length, codon usage alone achieve high accuracy
- If baseline gets >90%, your Genesis RNA isn't adding value
- Generates report showing what model learned

**Expected output:**
```
Accuracy: 0.XXX (XX.X%)

If >90%: Issue confirmed, need to retrain
If 75-90%: Partial issue, add validation
If <75%: Genesis RNA adds value, validate externally
```

---

### Step 2: Post Reddit Response (2 minutes)

Copy from [REDDIT_RESPONSE.md](REDDIT_RESPONSE.md) and paste to Reddit thread.

**Key points:**
- ✅ Acknowledge issue
- ✅ Show you're fixing it
- ✅ Ask for advice
- ✅ Promise to update with results

---

## 🔬 Proper Fix (Choose One Path)

### Option A: Retrain on BRCA Sequences (RECOMMENDED)

**What to do:**
1. Download BRCA1/BRCA2 reference sequences from Ensembl
2. Generate augmented training data (synonymous mutations, conservative substitutions)
3. Pre-train Genesis RNA on BRCA sequences
4. Fine-tune on ClinVar pathogenic/benign labels

**Colab cell to add:**
```python
# ═══════════════════════════════════════════════════════════════════════
# DOWNLOAD BRCA REFERENCE SEQUENCES (CORRECT DOMAIN)
# ═══════════════════════════════════════════════════════════════════════

import requests
from Bio import SeqIO
from io import StringIO

print("📥 Downloading BRCA1/BRCA2 reference mRNA sequences...")

# BRCA1 RefSeq mRNA
brca1_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NM_007294.4&rettype=fasta&retmode=text"
brca1_response = requests.get(brca1_url)
brca1_seq = SeqIO.read(StringIO(brca1_response.text), "fasta")

# BRCA2 RefSeq mRNA
brca2_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NM_000059.4&rettype=fasta&retmode=text"
brca2_response = requests.get(brca2_url)
brca2_seq = SeqIO.read(StringIO(brca2_response.text), "fasta")

print(f"✅ BRCA1: {len(brca1_seq)} nt")
print(f"✅ BRCA2: {len(brca2_seq)} nt")

# Generate augmented training data (benign-like variations)
def generate_augmented_sequences(ref_seq, num_samples=1000):
    """Generate training sequences with neutral mutations"""
    sequences = []

    for _ in range(num_samples):
        seq = list(str(ref_seq.seq))

        # Introduce random synonymous mutations (don't change protein)
        # This is simplified - proper version would use codon tables
        num_mutations = np.random.randint(1, 10)
        positions = np.random.choice(len(seq), num_mutations, replace=False)

        for pos in positions:
            # Replace with random nucleotide (simplified)
            seq[pos] = np.random.choice(['A', 'C', 'G', 'U'])

        sequences.append(''.join(seq))

    return sequences

# Generate augmented data
brca1_augmented = generate_augmented_sequences(brca1_seq, 5000)
brca2_augmented = generate_augmented_sequences(brca2_seq, 5000)

print(f"\n✅ Generated {len(brca1_augmented) + len(brca2_augmented)} augmented BRCA sequences")
print("   These will be used for pre-training on the CORRECT domain")

# Save for training
with open('data/brca_pretraining.txt', 'w') as f:
    for seq in brca1_augmented + brca2_augmented:
        f.write(seq + '\n')

print("✅ Ready to pre-train on BRCA domain!")
```

**Then retrain:**
```bash
# Pre-train on BRCA sequences (not ncRNA!)
python -m genesis_rna.train_pretrain \
    --model_size small \
    --data_path data/brca_pretraining.txt \
    --num_epochs 10 \
    --output_dir checkpoints/brca_pretrain

# Fine-tune on ClinVar labels
python -m genesis_rna.train_finetune \
    --pretrained_model checkpoints/brca_pretrain/best_model.pt \
    --clinvar_data data/clinvar_brca_variants.csv \
    --num_epochs 5
```

---

### Option B: Use ESM-2 Protein Model (ALTERNATIVE)

**What to do:**
Use Facebook's ESM-2 (protein language model) instead:

```python
# Install ESM
!pip install fair-esm

# Use ESM-2 embeddings
from esm import pretrained

model, alphabet = pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

def get_esm_embedding(protein_sequence):
    """Extract embeddings from ESM-2"""
    data = [("protein", protein_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    # Use mean of sequence embeddings
    return results["representations"][33].mean(1).cpu().numpy()

# Then train classifier on ESM embeddings
# This model is already trained on proteins (correct domain!)
```

**Advantage:**
- ✅ Already trained on correct domain (proteins)
- ✅ State-of-the-art architecture
- ✅ Can compare against AlphaMissense baseline

---

### Option C: Add Rigorous Cross-Validation

**What to do:**
Split data to prevent distribution-based cheating:

```python
# Split by chromosome position (same gene, different regions)
def split_by_position(df, gene='BRCA1', train_end=10000):
    """Split within same gene to control domain"""
    gene_df = df[df['GeneSymbol'] == gene]

    train = gene_df[gene_df['Start'] <= train_end]
    test = gene_df[gene_df['Start'] > train_end]

    return train, test

# Or split by variant type
def split_by_type(df):
    """Ensure train and test have different mutation types"""
    train = df[df['Type'].isin(['missense', 'frameshift'])]
    test = df[df['Type'].isin(['splice_site', 'nonsense'])]

    return train, test

# If accuracy drops significantly, model was cheating
```

---

## 📊 Update Reddit with Results

After running baseline test:

```
**UPDATE:** Ran simple baseline test.

Simple features (GC content, codon usage, length) achieved XX.X% accuracy.

[If >90%]:
Confirmed - model was detecting sequence type, not variant effects.
Retraining on BRCA sequences now.

[If 75-90%]:
Partial issue - some sequence composition shortcuts.
Adding rigorous cross-validation and external datasets.

[If <75%]:
Genesis RNA appears to add value beyond trivial features.
Validating on external data (ENIGMA consortium) next.

Full results: [link to baseline_results.txt in repo]
```

---

## ✅ Success Criteria

**Model is properly trained when:**
1. ✅ Simple baseline gets <75% accuracy (Genesis RNA adds value)
2. ✅ Cross-validation by position maintains performance
3. ✅ External dataset (ENIGMA) validation succeeds
4. ✅ Ablation studies show model uses correct signals

**Not properly trained if:**
- ❌ Simple baseline gets >90% accuracy
- ❌ Shuffled sequences maintain high accuracy
- ❌ Performance drops on external data
- ❌ Can't explain which features model uses

---

## 🎯 Timeline

**Today:**
- [ ] Run baseline test (5 min)
- [ ] Post Reddit response (2 min)
- [ ] Commit baseline script to repo

**This Week:**
- [ ] Choose retrain strategy (A, B, or C)
- [ ] Implement chosen fix
- [ ] Run validation tests
- [ ] Update Reddit with results

**This Month:**
- [ ] Validate on external datasets
- [ ] Write proper methodology section
- [ ] Submit to ML conference/journal

---

## 📚 Resources

**Papers to Read:**
- AlphaMissense (Nature 2023) - Proper variant prediction methodology
- ESM-2 (Science 2023) - Protein language model architecture
- CADD/PolyPhen - Established variant effect predictors

**Datasets to Use:**
- ENIGMA consortium (BRCA-specific annotations)
- ClinGen expert-reviewed variants
- TCGA somatic mutations

**Compare Against:**
- REVEL score
- AlphaMissense predictions
- CADD scores

---

## 💬 Community Response Template

After you fix it:

```
**UPDATE [Date]: Fixed!**

Results after retraining on BRCA sequences:
- Simple baseline: XX% accuracy
- Genesis RNA: YY% accuracy
- Gap: ZZ% improvement

External validation (ENIGMA dataset):
- Accuracy: XX%
- AUC-ROC: X.XXX

Thanks to everyone who provided feedback. This is why open science works!

Updated repo: [link]
```

---

**Bottom line:** Run the baseline test first. If it confirms the issue, retrain on BRCA sequences. Update Reddit with honest results. Build scientific credibility through transparency. 🔬✨
