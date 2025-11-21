#!/usr/bin/env python3
"""
Quick Cancer Variant Analysis - Works Without Pre-trained Model
Demonstrates the complete workflow: train → analyze → validate
"""

import sys
from pathlib import Path
import json

# Add genesis_rna to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'genesis_rna'))

import torch
from genesis_rna.breast_cancer import BreastCancerAnalyzer
from genesis_rna.train_pretrain import main as train_main

def quick_train_model():
    """Train a quick model for demonstration"""
    print("="*70)
    print("STEP 1: Training Genesis RNA Model (Quick Mode)")
    print("="*70)
    print("\n⏱️  This will take ~5-10 minutes on GPU (30 min on CPU)")
    print("Using dummy data for quick demonstration\n")

    # Set up training args
    import argparse
    parser = argparse.ArgumentParser()

    # Quick training settings
    args = parser.parse_args([
        '--use_dummy_data',
        '--model_size', 'small',
        '--batch_size', '32',
        '--num_epochs', '3',
        '--learning_rate', '0.0001',
        '--output_dir', 'checkpoints/quick_demo',
        '--save_every', '1'
    ])

    # Train model
    try:
        train_main(args)
        print("\n✅ Training complete!")
        return 'checkpoints/quick_demo/best_model.pt'
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None

def analyze_brca_variants(model_path):
    """Analyze BRCA variants using trained model"""
    print("\n" + "="*70)
    print("STEP 2: Analyzing BRCA1/BRCA2 Variants")
    print("="*70)

    # Load variant database
    variant_file = Path(__file__).parent.parent / 'data' / 'breast_cancer' / 'brca_variants.json'

    if not variant_file.exists():
        print(f"\n❌ Variant database not found!")
        print("Expected location: {variant_file}")
        return

    with open(variant_file) as f:
        variants = json.load(f)

    print(f"\n📥 Loaded {len(variants)} variants from database")

    # Initialize analyzer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📥 Loading model from {model_path}...")
    analyzer = BreastCancerAnalyzer(str(model_path), device=device)
    print(f"✅ Analyzer ready on {device}")

    # Analyze each variant
    results = []
    print(f"\n🔬 Analyzing variants...\n")

    for variant_id, variant_data in variants.items():
        try:
            pred = analyzer.predict_variant_effect(
                gene=variant_data['gene'],
                wild_type_rna=variant_data['wild_type'],
                mutant_rna=variant_data['mutant'],
                variant_id=variant_id
            )

            # Check concordance
            known_pathogenic = variant_data['clinical_significance'] in ['Pathogenic', 'Likely Pathogenic']
            predicted_pathogenic = pred.interpretation in ['Pathogenic', 'Likely Pathogenic']
            concordant = known_pathogenic == predicted_pathogenic

            status = "✅" if concordant else "❌"

            print(f"{status} {variant_id:<25}")
            print(f"   Known:     {variant_data['clinical_significance']}")
            print(f"   Predicted: {pred.interpretation} (score: {pred.pathogenicity_score:.3f})")
            print()

            results.append({
                'variant_id': variant_id,
                'gene': variant_data['gene'],
                'known': variant_data['clinical_significance'],
                'predicted': pred.interpretation,
                'score': pred.pathogenicity_score,
                'concordant': concordant
            })

        except Exception as e:
            print(f"❌ Error analyzing {variant_id}: {e}\n")

    return results

def print_summary(results):
    """Print analysis summary"""
    if not results:
        print("\n⚠️  No results to summarize")
        return

    print("="*70)
    print("STEP 3: Summary & Clinical Metrics")
    print("="*70)

    total = len(results)
    concordant = sum(1 for r in results if r['concordant'])

    print(f"\nTotal Variants Analyzed: {total}")
    print(f"Concordant with ClinVar: {concordant} / {total} ({concordant/total*100:.1f}%)")

    # Identify discordant cases
    discordant = [r for r in results if not r['concordant']]
    if discordant:
        print(f"\n⚠️  Discordant Predictions:")
        for r in discordant:
            print(f"  • {r['variant_id']}: Known={r['known']}, Predicted={r['predicted']}")
    else:
        print(f"\n✅ Perfect concordance with ClinVar!")

    # Clinical interpretation
    print(f"\nClinical Interpretation:")
    if concordant / total >= 0.8:
        print(f"  ✅ Good performance - suitable for research applications")
    else:
        print(f"  ⚠️  Model needs improvement for clinical use")
        print(f"  Consider: More training data, longer training, or fine-tuning")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🎗️  Genesis RNA - Quick Cancer Variant Analysis                ║
    ║                                                                   ║
    ║  This script demonstrates the complete workflow:                  ║
    ║  1. Train a quick model (3 epochs, ~10 min)                      ║
    ║  2. Analyze BRCA1/BRCA2 variants                                 ║
    ║  3. Validate against ClinVar database                            ║
    ║                                                                   ║
    ║  Together, we can cure breast cancer! 🎗️                         ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Check if model already exists
    model_path = Path('checkpoints/quick_demo/best_model.pt')

    if model_path.exists():
        print(f"\n✅ Found existing model at {model_path}")
        use_existing = input("Use existing model? (y/n): ").lower().strip()

        if use_existing != 'y':
            print("\n🏋️  Training new model...")
            model_path = quick_train_model()
            if not model_path:
                print("\n❌ Training failed. Exiting.")
                return
    else:
        print("\n🏋️  No existing model found. Training new model...")
        model_path = quick_train_model()
        if not model_path:
            print("\n❌ Training failed. Exiting.")
            return

    # Analyze variants
    results = analyze_brca_variants(model_path)

    if results:
        print_summary(results)

    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("  • Review concordance with ClinVar")
    print("  • Try batch analysis: python scripts/batch_variant_analysis.py")
    print("  • Generate visualizations: python scripts/visualize_variant_analysis.py")
    print("  • Add more variants to data/breast_cancer/brca_variants.json")
    print("\n🎗️  Together, we can cure breast cancer!")

if __name__ == '__main__':
    main()
