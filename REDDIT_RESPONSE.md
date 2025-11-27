# Reddit Response - Short Version

**Copy and paste this:**

---

You're absolutely right - excellent catch! The ncRNA→BRCA domain shift is exactly the issue. The model could be learning "doesn't look like ncRNA" rather than actual variant effects.

**Fixing it now:**

1. Running baseline tests (k-mer/GC-content only) to see if simple features get same accuracy
2. Retraining on coding sequences instead of ncRNA
3. Will compare against ESM-2/AlphaMissense baselines

I've updated the README with a limitations section: https://github.com/oluwafemidiakhoa/genesi_ai

This is exactly why I posted - to get this kind of rigorous feedback before making clinical claims. Much better to catch it here than in production!

**Question:** Should I retrain Genesis RNA on BRCA sequences, or switch to an established protein language model architecture like ESM-2?

Will update with baseline results. Thanks for the thorough review!

---

**Optional additions based on your results:**

If you run the simple baseline test first, add:

## Quick Baseline Test Results

Tested a simple Random Forest on just:
- GC content
- Sequence length
- Start codon frequency

**Baseline accuracy: XX.X%**

[If <90%]: Genesis RNA is adding value beyond sequence composition
[If >90%]: You were right - most signal is from distribution shift, not variant effects

Full results in repo: [link]

---

**Tone notes:**
- ✅ Acknowledge the issue openly
- ✅ Show you're taking it seriously with concrete actions
- ✅ Thank them for catching it
- ✅ Ask for suggestions (builds community support)
- ✅ Update with results (follow through)

**This response will earn respect even though it admits a flaw. Scientific honesty > inflated claims.**
