#!/usr/bin/env python3
"""
Batch Variant Analysis Script for Genesis RNA

Analyzes multiple BRCA1/BRCA2/TP53 variants and validates against ClinVar.
Generates comprehensive report with clinical metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'genesis_rna'))

import torch
from genesis_rna.breast_cancer import BreastCancerAnalyzer


def load_variants(variant_file: str) -> Dict:
    """Load variant database from JSON file"""
    with open(variant_file, 'r') as f:
        return json.load(f)


def analyze_batch(
    analyzer: BreastCancerAnalyzer,
    variants: Dict,
    verbose: bool = True
) -> List[Dict]:
    """
    Analyze all variants in batch

    Args:
        analyzer: BreastCancerAnalyzer instance
        variants: Dictionary of variants from JSON
        verbose: Print progress

    Returns:
        List of result dictionaries
    """
    results = []

    for i, (variant_id, variant_data) in enumerate(variants.items(), 1):
        if verbose:
            print(f"Analyzing {i}/{len(variants)}: {variant_id}...")

        try:
            # Run prediction
            pred = analyzer.predict_variant_effect(
                gene=variant_data['gene'],
                wild_type_rna=variant_data['wild_type'],
                mutant_rna=variant_data['mutant'],
                variant_id=variant_id
            )

            # Check concordance with known clinical significance
            known_pathogenic = variant_data['clinical_significance'] in ['Pathogenic', 'Likely Pathogenic']
            predicted_pathogenic = pred.interpretation in ['Pathogenic', 'Likely Pathogenic']
            concordance = known_pathogenic == predicted_pathogenic

            result = {
                'variant_id': variant_id,
                'gene': variant_data['gene'],
                'mutation_type': variant_data['mutation_type'],
                'exon': variant_data['exon'],
                'known_significance': variant_data['clinical_significance'],
                'predicted_score': pred.pathogenicity_score,
                'predicted_interpretation': pred.interpretation,
                'delta_stability': pred.delta_stability,
                'confidence': pred.confidence,
                'concordance': concordance,
                'description': variant_data['description']
            }

            results.append(result)

            if verbose:
                status = "✅" if concordance else "❌"
                print(f"  {status} {pred.interpretation} (score: {pred.pathogenicity_score:.3f})")

        except Exception as e:
            print(f"  ❌ Error analyzing {variant_id}: {e}")
            continue

    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate clinical performance metrics"""
    df = pd.DataFrame(results)

    # True/False Positives/Negatives
    df['known_pathogenic'] = df['known_significance'].isin(['Pathogenic', 'Likely Pathogenic'])
    df['predicted_pathogenic'] = df['predicted_interpretation'].isin(['Pathogenic', 'Likely Pathogenic'])

    tp = ((df['known_pathogenic'] == True) & (df['predicted_pathogenic'] == True)).sum()
    tn = ((df['known_pathogenic'] == False) & (df['predicted_pathogenic'] == False)).sum()
    fp = ((df['known_pathogenic'] == False) & (df['predicted_pathogenic'] == True)).sum()
    fn = ((df['known_pathogenic'] == True) & (df['predicted_pathogenic'] == False)).sum()

    # Clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
    concordance = df['concordance'].mean()

    return {
        'total_variants': len(df),
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy,
        'concordance': concordance,
    }


def print_summary(metrics: Dict):
    """Print formatted metrics summary"""
    print("\n" + "="*70)
    print("CLINICAL PERFORMANCE METRICS")
    print("="*70)

    print(f"\nDataset:")
    print(f"  Total Variants: {metrics['total_variants']}")
    print(f"  True Positives:  {metrics['true_positives']} (Pathogenic correctly identified)")
    print(f"  True Negatives:  {metrics['true_negatives']} (Benign correctly identified)")
    print(f"  False Positives: {metrics['false_positives']} (Benign called pathogenic)")
    print(f"  False Negatives: {metrics['false_negatives']} (Pathogenic missed - CRITICAL!)")

    print(f"\nPerformance:")
    print(f"  Sensitivity (Recall):     {metrics['sensitivity']:.1%}")
    print(f"  Specificity:              {metrics['specificity']:.1%}")
    print(f"  PPV (Precision):          {metrics['ppv']:.1%}")
    print(f"  NPV:                      {metrics['npv']:.1%}")
    print(f"  Accuracy:                 {metrics['accuracy']:.1%}")
    print(f"  ClinVar Concordance:      {metrics['concordance']:.1%}")

    print(f"\nClinical Interpretation:")
    if metrics['sensitivity'] >= 0.90:
        print(f"  ✅ Excellent sensitivity - safe for clinical screening")
    elif metrics['sensitivity'] >= 0.80:
        print(f"  ⚠️  Good sensitivity - may miss some pathogenic variants")
    else:
        print(f"  ❌ Poor sensitivity - NOT ready for clinical use")

    if metrics['false_negatives'] > 0:
        print(f"  ⚠️  {metrics['false_negatives']} pathogenic variants missed - review needed!")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Batch variant analysis with Genesis RNA')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--variants', required=True, help='Path to variant JSON file')
    parser.add_argument('--output', help='Output CSV file for results')
    parser.add_argument('--metrics-output', help='Output JSON file for metrics')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    # Check files exist
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.variants).exists():
        print(f"❌ Variant file not found: {args.variants}")
        sys.exit(1)

    # Load analyzer
    print(f"📥 Loading model from {args.model}...")
    analyzer = BreastCancerAnalyzer(args.model, device=args.device)
    print(f"✅ Model loaded on {args.device}")

    # Load variants
    print(f"📥 Loading variants from {args.variants}...")
    variants = load_variants(args.variants)
    print(f"✅ Loaded {len(variants)} variants")

    # Analyze
    print(f"\n🔬 Analyzing variants...")
    results = analyze_batch(analyzer, variants, verbose=not args.quiet)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print summary
    print_summary(metrics)

    # Save results
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to {args.output}")

    if args.metrics_output:
        with open(args.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Metrics saved to {args.metrics_output}")

    # Exit code based on performance
    if metrics['false_negatives'] > 0:
        print(f"\n⚠️  WARNING: Pathogenic variants were missed!")
        sys.exit(2)
    elif metrics['sensitivity'] < 0.80:
        print(f"\n⚠️  WARNING: Sensitivity below 80%!")
        sys.exit(1)
    else:
        print(f"\n✅ Analysis complete!")
        sys.exit(0)


if __name__ == '__main__':
    main()
