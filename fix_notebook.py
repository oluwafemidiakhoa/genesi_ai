#!/usr/bin/env python3
"""
Fix the breast_cancer_research_colab.ipynb notebook
Replaces Cell 17 with proper analyzer initialization
"""

import json
import sys

def fix_notebook():
    notebook_path = 'genesis_rna/breast_cancer_research_colab.ipynb'

    print(f"Reading notebook from: {notebook_path}")

    # Read notebook with UTF-8 encoding
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # New Cell 17 code that properly initializes the analyzer
    new_cell_17_code = '''# Initialize BreastCancerAnalyzer - FIXED VERSION
import sys
import torch
sys.path.insert(0, '/content/genesi_ai/genesis_rna')

from genesis_rna.breast_cancer import BreastCancerAnalyzer

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Find model path (adjust if needed)
MODEL_PATH = f"{DRIVE_DIR}/checkpoints/quick/best_model.pt"

# Alternative paths to try
import os
if not os.path.exists(MODEL_PATH):
    alternative_paths = [
        f"{DRIVE_DIR}/checkpoints/full/best_model.pt",
        "/content/genesi_ai/checkpoints/quick/best_model.pt",
    ]
    for path in alternative_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            break

print(f"📥 Loading model from {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("\\n⚠️  Please complete Step 2 (Training) first!")
    print("Or update MODEL_PATH to point to your trained model")
else:
    # Initialize analyzer
    analyzer = BreastCancerAnalyzer(MODEL_PATH, device=device)

    print(f"✅ Analyzer initialized on {device}")
    print(f"\\nSupported cancer genes:")
    for gene, desc in analyzer.cancer_genes.items():
        print(f"  • {gene}: {desc}")'''

    # Replace Cell 17
    if len(nb['cells']) > 17:
        print(f"\\nReplacing Cell 17...")
        nb['cells'][17] = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in new_cell_17_code.split("\n")]
        }
        print("[OK] Cell 17 replaced with fixed analyzer initialization")
    else:
        print(f"[ERROR] Notebook has fewer than 18 cells")
        return False

    # Optionally delete Cell 18 (the duplicate BreastCancerAnalyzer definition)
    if len(nb['cells']) > 18:
        print(f"\\nCell 18 info:")
        cell_18 = nb['cells'][18]
        if cell_18['cell_type'] == 'code':
            source_preview = ''.join(cell_18['source'][:3])
            print(f"  Preview: {source_preview[:100]}...")

            if '@dataclass' in ''.join(cell_18['source']) or 'class BreastCancerAnalyzer' in ''.join(cell_18['source']):
                print("  This appears to be the duplicate analyzer definition")
                print("  Automatically deleting Cell 18 (duplicate code)...")
                nb['cells'].pop(18)
                print("  [OK] Cell 18 deleted")

    # Write back with UTF-8 encoding
    print(f"\\nWriting fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"\\n[SUCCESS] Notebook fixed successfully!")
    print(f"\\nChanges made:")
    print(f"  - Cell 17: Replaced with proper analyzer initialization")
    print(f"  - Uses BreastCancerAnalyzer from genesis_rna.breast_cancer module")
    print(f"  - Handles multiple model path locations")
    print(f"  - Provides clear error messages if model not found")

    return True

if __name__ == '__main__':
    try:
        success = fix_notebook()
        if success:
            print(f"\\nThe notebook should now work in Colab!")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
