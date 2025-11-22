#!/usr/bin/env python3
"""
Fix Cell 21 in breast_cancer_research_colab.ipynb
Changes designer.design_therapeutic() to designer.design()
"""

import json
import sys

def fix_designer_cell():
    notebook_path = 'genesis_rna/breast_cancer_research_colab.ipynb'

    print(f"Reading notebook from: {notebook_path}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # New Cell 21 code with CORRECT design() method
    new_cell_21_code = '''print("="*70)
print("mRNA Therapeutic Design: p53")
print("="*70)

p53_protein = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDD"

print(f"\\nTarget: p53 tumor suppressor")
print(f"Length: {len(p53_protein)} amino acids")
print(f"\\n⚙️ Designing mRNA...")

# CORRECTED: Use design() method with optimization_goals dict
therapeutic = designer.design(
    protein_sequence=p53_protein,
    optimization_goals={
        'stability': 0.95,
        'translation': 0.90,
        'immunogenicity': 0.1
    }
)

print(f"\\n✅ Design complete!")
print(f"\\n{'Property':<30} {'Value'}")
print("="*50)
print(f"{'Length:':<30} {therapeutic.length} nt")
print(f"{'Stability:':<30} {therapeutic.stability_score:.3f}")
print(f"{'Translation:':<30} {therapeutic.translation_score:.3f}")
print(f"{'Immunogenicity:':<30} {therapeutic.immunogenicity_score:.3f}")
print(f"{'Half-life:':<30} {therapeutic.half_life_hours:.1f} hours")

print(f"\\n🧬 Sequence (first 100 nt):")
print(f"   {therapeutic.sequence[:100]}...")

# Save to Drive
import json
result = {
    'protein': p53_protein,
    'mrna': therapeutic.sequence,
    'stability': therapeutic.stability_score,
    'translation': therapeutic.translation_score,
    'half_life': therapeutic.half_life_hours
}

with open(f"{DRIVE_DIR}/results/p53_therapeutic.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\\n💾 Saved to {DRIVE_DIR}/results/p53_therapeutic.json")'''

    # Replace Cell 21
    if len(nb['cells']) > 21:
        print(f"\nReplacing Cell 21...")
        nb['cells'][21] = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in new_cell_21_code.split("\n")]
        }
        print("[OK] Cell 21 replaced with corrected designer.design() method")
    else:
        print(f"[ERROR] Notebook has fewer than 22 cells")
        return False

    # Write back
    print(f"\nWriting fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Notebook fixed successfully!")
    print(f"\nChanges made:")
    print(f"  - Cell 21: Changed design_therapeutic() → design()")
    print(f"  - Added optimization_goals parameter with dict format")
    print(f"  - Added immunogenicity optimization")

    return True

if __name__ == '__main__':
    try:
        success = fix_designer_cell()
        if success:
            print(f"\n✅ The notebook is now fixed with correct method names!")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
