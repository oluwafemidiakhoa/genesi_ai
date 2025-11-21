#!/usr/bin/env python3
"""
Simple Cancer Variant Analysis - Uses existing model or trains quickly
"""

import sys
from pathlib import Path
import json
import subprocess

# Add genesis_rna to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'genesis_rna'))

import torch
from genesis_rna.breast_cancer import BreastCancerAnalyzer


def find_or_train_model():
    """Find existing model or offer to train one"""

    # Check for existing models
    possible_paths = [
        'checkpoints/quick_demo/best_model.pt',
        'checkpoints/cancer_model/best_model.pt',
        'checkpoints/pretrained/base/best_model.pt',
        'checkpoints/pretrained/small/best_model.pt',
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Found existing model: {path}")
            return path

    print("\n⚠️  No trained model found!")
    print("\nTo train a model, run:")
    print("  python -m genesis_rna.train_pretrain \\")
    print("      --use_dummy_data \\")
    print("      --model_size small \\")
    print("      --num_epochs 3 \\")
    print("      --output_dir checkpoints/quick_demo")
    print("\nThis will take ~10 minutes on GPU")

    response = input("\nTrain model now? (y/n): ").lower().strip()

    if response == 'y':
        print("\n🏋️  Training model...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'genesis_rna.train_pretrain',
                '--use_dummy_data',
                '--model_size', 'small',
                '--num_epochs', '3',
                '--output_dir', 'checkpoints/quick_demo'
            ], check=True)

            model_path = 'checkpoints/quick_demo/best_model.pt'
            if Path(model_path).exists():
                print(f"\n✅ Training complete! Model saved to {model_path}")
                return model_path
            else:
                print("\n❌ Training completed but model not found")
                return None
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed: {e}")
            return None
    else:
        return None


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🎗️  Genesis RNA - Simple Cancer Variant Analysis               ║
    ║                                                                   ║
    ║  Analyzes BRCA1/BRCA2/TP53 variants against ClinVar database     ║
    ║  Together, we can cure breast cancer! 🎗️                         ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Find or train model
    model_path = find_or_train_model()

    if not model_path:
        print("\n❌ No model available. Exiting.")
        return

    # Load variant database
    variant_file = Path(__file__).parent.parent / 'data' / 'breast_cancer' / 'brca_variants.json'

    if not variant_file.exists():
        print(f"\n❌ Variant database not found at {variant_file}")
        print("Please ensure data/breast_cancer/brca_variants.json exists")
        return

    with open(variant_file) as f:
        variants = json.load(f)

    print(f"\n📥 Loaded {len(variants)} variants from database")

    # Initialize analyzer
    print(f"\n📥 Loading analyzer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = BreastCancerAnalyzer(str(model_path), device=device)
    print(f"✅ Analyzer ready on {device}")

    # Analyze variants
    print("\n" + "="*70)
    print("ANALYZING BRCA1/BRCA2/TP53 VARIANTS")
    print("="*70)

    results = []

    for variant_id, variant_data in variants.items():
        print(f"\n🔬 {variant_id}")
        print(f"   Gene: {variant_data['gene']}")
        print(f"   Known: {variant_data['clinical_significance']}")

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

            print(f"   Predicted: {pred.interpretation} (score: {pred.pathogenicity_score:.3f})")
            print(f"   Concordance: {status} {'Match' if concordant else 'MISMATCH'}")

            results.append({
                'variant_id': variant_id,
                'known': variant_data['clinical_significance'],
                'predicted': pred.interpretation,
                'score': pred.pathogenicity_score,
                'concordant': concordant
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results:
        total = len(results)
        concordant = sum(1 for r in results if r['concordant'])

        print(f"\nTotal Variants: {total}")
        print(f"Concordant: {concordant} / {total} ({concordant/total*100:.1f}%)")

        # Discordant cases
        discordant = [r for r in results if not r['concordant']]
        if discordant:
            print(f"\n⚠️  Discordant Predictions:")
            for r in discordant:
                print(f"  • {r['variant_id']}")
                print(f"    Known: {r['known']}, Predicted: {r['predicted']}")
        else:
            print(f"\n✅ Perfect concordance with ClinVar!")

        # Clinical interpretation
        if concordant / total >= 0.8:
            print(f"\n✅ Good performance for research use")
        else:
            print(f"\n⚠️  Model needs improvement")
            print(f"   Consider: More training epochs, larger model, or fine-tuning")

    print("\n" + "="*70)
    print("🎗️  Together, we can cure breast cancer!")
    print("="*70)

    print("\nNext Steps:")
    print("  • Run batch analysis: python scripts/batch_variant_analysis.py \\")
    print("      --model", model_path, "\\")
    print("      --variants data/breast_cancer/brca_variants.json \\")
    print("      --output results/analysis.csv")
    print("\n  • Generate visualizations: python scripts/visualize_variant_analysis.py \\")
    print("      --results results/analysis.csv --output plots/")


if __name__ == '__main__':
    main()
