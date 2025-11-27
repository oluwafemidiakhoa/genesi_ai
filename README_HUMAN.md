# Genesis RNA - BRCA Variant Classifier

##  IMPORTANT UPDATE (January 27, 2025)

I made a mistake. The 100% accuracy I claimed was wrong.

Reddit users found that my Colab notebook was using fake RNA sequences with an "AAAA" marker automatically inserted for pathogenic variants. The model learned to detect this fake marker instead of learning real biology.

**The bug:** `if row.get('Label') == 1: sequence = sequence[:mid] + 'AAAA' + sequence[mid+4:]`

I'm fixing this now by fetching real BRCA sequences from Ensembl and retraining properly. See [CRITICAL_LABEL_LEAKAGE.md](CRITICAL_LABEL_LEAKAGE.md) for details.

**Status:** Don't use this yet - I'm retraining with real data.

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oluwafemidiakhoa/genesi_ai/blob/main/genesis_rna/breast_cancer_research_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is This?

Genesis RNA is my attempt to build an AI system that can predict whether BRCA1/BRCA2 genetic variants cause breast cancer.

I'm a developer interested in using AI to help with cancer research. This project started as a way to learn about genomics and deep learning.

### What Works
- Pre-training on 50,000+ real ncRNA sequences from Ensembl
- Transformer architecture that learns RNA patterns
- Adaptive Sparse Training (AST) that speeds up training by 60%
- Open source code that you can review and improve

### What Doesn't Work (Yet)
- ~~100% accuracy~~ - this was fake, caused by synthetic data bug
- Variant classification - needs retraining with real sequences
- Clinical predictions - not validated yet

---

## What Happened

### The Bug

I used AI (LLM) to generate code for Cell 24 in my Colab notebook. The code created fake RNA sequences and inserted an "AAAA" pattern for pathogenic variants but not benign ones.

My model learned: "Has AAAA? → Pathogenic. No AAAA? → Benign."

That's not biology - that's detecting a bug in my code.

### How It Was Found

I posted to Reddit r/MachineLearning asking for feedback. Users immediately spotted problems:

- **Dihedralman**: "100% accuracy is a red flag"
- **profesh_amateur**: Found the exact line of code causing the problem
- **everyday847**: Noticed the docstring said "synthetic sequences"
- **HasGreatVocabulary**: Recognized LLM-generated code patterns

This is peer review working perfectly. Thank you Reddit!

### What I'm Doing

1. ✅ Acknowledged the problem publicly
2. ✅ Documented exactly what went wrong
3. ✅ Updated all documentation with disclaimers
4. ✅ Built proper pipeline with real Ensembl sequences
5. ⏳ Retraining with real data (in progress)
6. ⏳ Will report honest results (expect 70-85%, not 100%)

---

## The Fix

### New Pipeline (Fixed)

1. **Real Sequences** ([scripts/fetch_real_brca_sequences.py](scripts/fetch_real_brca_sequences.py))
   - Fetches actual BRCA1/BRCA2 mRNA from Ensembl
   - Uses HGVS parser to apply real variant mutations
   - NO label information used

2. **Proper Splitting** ([scripts/proper_train_test_split.py](scripts/proper_train_test_split.py))
   - Temporal split: train on old variants, test on new
   - Position split: first half vs second half of gene
   - NO random split (prevents memorization)

3. **Baseline First** ([scripts/baseline_models.py](scripts/baseline_models.py))
   - Test simple k-mer counting before transformers
   - If k-mer gets 80%, we don't need deep learning!
   - Justifies complexity

4. **Progressive Training** ([scripts/progressive_model_training.py](scripts/progressive_model_training.py))
   - Start simple, add complexity only if needed
   - Decision tree: Logistic Regression → Random Forest → Transformer
   - Stop when performance is good enough

5. **Comprehensive Evaluation** ([scripts/comprehensive_evaluation.py](scripts/comprehensive_evaluation.py))
   - Multiple metrics beyond just accuracy
   - Checks for 100% (red flag detector!)
   - Clinical interpretation

See [REAL_DATA_COMPLETE.md](REAL_DATA_COMPLETE.md) for complete workflow.

---

## Realistic Expectations

| Method | Expected Accuracy | Notes |
|--------|-------------------|-------|
| K-mer counting | 75-80% | Simple baseline |
| Random Forest | 78-82% | Hand-crafted features |
| Word2Vec + CNN | 80-85% | Representation learning |
| Transformer | 85-90% | State-of-the-art |
| **100%** | **BUG** | **Label leakage!** |

Real-world variant prediction is hard. 80-85% would be excellent and competitive with tools like CADD and REVEL.

---

## How To Use (Once Fixed)

### Complete Pipeline

```bash
# 1. Get real BRCA sequences
python scripts/fetch_real_brca_sequences.py \
    --clinvar data/breast_cancer/clinvar_brca.csv \
    --output data/breast_cancer/real_sequences.csv

# 2. Proper train/test split
python scripts/proper_train_test_split.py \
    --input data/breast_cancer/real_sequences.csv \
    --train_out data/breast_cancer/train.csv \
    --test_out data/breast_cancer/test.csv \
    --method temporal

# 3. Test simple baselines FIRST
python scripts/baseline_models.py \
    --train data/breast_cancer/train.csv \
    --test data/breast_cancer/test.csv

# 4. Only use transformer if baseline fails
# (Expecting 70-85% with real data)
```

---

## Why I'm Sharing This

### Lessons I Learned

1. **100% accuracy is always suspicious** - Should have investigated immediately
2. **Verify AI-generated code** - Read every line before running
3. **Check for synthetic data** - Don't trust docstrings saying "biologically plausible"
4. **Medical AI needs extra scrutiny** - Cancer predictions affect real people
5. **Community review is invaluable** - Open source saved me from publishing bad science

### What Worked

- **Open source** - Reddit users could see my code and find the bug
- **Community feedback** - r/MachineLearning caught it before publication
- **Transparent acknowledgment** - Admitting mistakes builds trust
- **Real data pipeline** - Now have proper workflow for future work

### Why This Matters

If I had published this with claims of 100% accuracy:
- Patients might have made medical decisions based on flawed AI
- Trust in AI for healthcare would be damaged
- Other researchers might waste time trying to reproduce fake results
- Clinical labs might have adopted broken methodology

Open source and peer review prevented all of this.

---

## Contributing

I welcome help with:

- Testing the new real data pipeline
- Adding biological features (conservation scores, RNA structure)
- Comparing to existing tools (CADD, REVEL, PolyPhen)
- External validation on ENIGMA or TCGA data
- Documentation improvements
- Bug reports

Please open an issue or pull request on GitHub.

---

## Documentation

### Essential Reading
- [CRITICAL_LABEL_LEAKAGE.md](CRITICAL_LABEL_LEAKAGE.md) - What went wrong and why
- [REAL_DATA_COMPLETE.md](REAL_DATA_COMPLETE.md) - New workflow with real data
- [COMPLETE_REDESIGN_PLAN.md](COMPLETE_REDESIGN_PLAN.md) - Comprehensive fix strategy

### Technical Details
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - How to train the model
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Performance optimizations
- [AST_CANCER_IMPACT.md](AST_CANCER_IMPACT.md) - Adaptive Sparse Training benefits

---

## Acknowledgments

**Thank you to Reddit r/MachineLearning:**

- **profesh_amateur** - Found the exact bug in Cell 24
- **Dihedralman** - Flagged 100% as suspicious
- **everyday847** - Noticed docstring said "synthetic"
- **HasGreatVocabulary** - Recognized LLM-generated patterns
- **Leather_Power_1137** - Emphasized verification importance

You caught a major bug before it caused harm. This is peer review working as it should.

---

## Current Status

**What I'm working on:**
- ✅ Fixed data pipeline (real Ensembl sequences)
- ✅ Proper train/test splits (temporal validation)
- ✅ Baseline models (k-mer, Random Forest)
- ⏳ Retraining with real data
- ⏳ Documenting honest results

**Next update:** After retraining completes, I'll post real results (expecting 70-85%, not 100%).

---

## License

MIT License - See [LICENSE](LICENSE) for details.

Open source so the community can verify, improve, and learn from both successes and mistakes.

---

## Contact

- **GitHub Issues**: https://github.com/oluwafemidiakhoa/genesi_ai/issues
- **Reddit Discussion**: r/MachineLearning thread

---

**Disclaimer:** This is a research project, not a medical device. Do not use for clinical decisions. All medical decisions should be made with qualified healthcare professionals using validated clinical tests.

---

**Last updated:** January 27, 2025
**Status:** Retraining with real data
**Honest about mistakes, committed to doing it right**
