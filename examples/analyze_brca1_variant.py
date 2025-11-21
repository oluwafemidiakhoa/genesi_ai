#!/usr/bin/env python3
"""
BRCA1 Variant Analysis Example
Analyzes the pathogenic BRCA1 c.5266dupC (5382insC) founder mutation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'genesis_rna'))

import torch
from genesis_rna.breast_cancer import BreastCancerAnalyzer
import json

# Load variant data from database
variant_file = Path(__file__).parent.parent / 'data' / 'breast_cancer' / 'brca_variants.json'

if not variant_file.exists():
    print(f"❌ Variant database not found at {variant_file}")
    print("Please ensure data/breast_cancer/brca_variants.json exists")
    sys.exit(1)

with open(variant_file) as f:
    variants = json.load(f)

# Get BRCA1 variant
variant_id = 'BRCA1_c.5266dupC'
variant_data = variants[variant_id]

print("="*70)
print("BRCA1 Pathogenic Variant Analysis")
print("="*70)
print(f"\nVariant: {variant_id}")
print(f"Description: {variant_data['description']}")
print(f"Known Significance: {variant_data['clinical_significance']}")
print(f"Cancer Risk: {variant_data['cancer_risk']}")

# Check if model exists
model_path = Path('checkpoints/pretrained/base/best_model.pt')
if not model_path.exists():
    print(f"\n⚠️  Model not found at {model_path}")
    print("\nOptions:")
    print("1. Train a model first:")
    print("   python -m genesis_rna.train_pretrain --use_dummy_data --model_size small --num_epochs 5 --output_dir checkpoints/quick")
    print("\n2. Or specify your model path:")
    print("   Edit this script and update model_path variable")
    sys.exit(1)

# Initialize analyzer
print(f"\n📥 Loading model from {model_path}...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
analyzer = BreastCancerAnalyzer(str(model_path), device=device)
print(f"✅ Model loaded on {device}")

# Analyze variant
print(f"\n🔬 Analyzing variant...")
pred = analyzer.predict_variant_effect(
    gene=variant_data['gene'],
    wild_type_rna=variant_data['wild_type'],
    mutant_rna=variant_data['mutant'],
    variant_id=variant_id
)

# Display results
print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)
print(f"\n{'Variant ID:':<30} {pred.variant_id}")
print(f"{'Pathogenicity Score:':<30} {pred.pathogenicity_score:.3f}")
print(f"{'ΔStability (kcal/mol):':<30} {pred.delta_stability:.2f}")
print(f"{'Clinical Interpretation:':<30} {pred.interpretation}")
print(f"{'Confidence:':<30} {pred.confidence:.3f}")

# Validate against known significance
known_pathogenic = variant_data['clinical_significance'] in ['Pathogenic', 'Likely Pathogenic']
predicted_pathogenic = pred.interpretation in ['Pathogenic', 'Likely Pathogenic']
concordant = known_pathogenic == predicted_pathogenic

print(f"\n{'ClinVar Agreement:':<30} {'✅ Concordant' if concordant else '❌ Discordant'}")

print("\n📋 Clinical Significance:")
print("  • Known pathogenic frameshift (5382insC)")
print("  • Ashkenazi Jewish founder mutation")
print("  • Disrupts BRCA1 BRCT domain (DNA repair)")
print("  • 65-80% lifetime breast cancer risk")
print("  • 40-50% lifetime ovarian cancer risk")
print("  • Recommend: Enhanced screening + counseling")
print(f"  • Reference: {', '.join(variant_data['references'])}")

print("\n" + "="*70)
print("🎗️  Together, we can cure breast cancer!")
print("="*70)
