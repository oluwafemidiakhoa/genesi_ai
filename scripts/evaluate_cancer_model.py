#!/usr/bin/env python3
"""
Evaluation Framework for Genesis RNA Cancer Research

This script evaluates the trained RNA foundation model on cancer-related tasks:
1. BRCA variant pathogenicity prediction
2. Variant effect prediction accuracy
3. Clinical metrics (sensitivity, specificity, PPV, NPV)
4. Comparison with baseline methods

Usage:
    python evaluate_cancer_model.py --model checkpoints/best_model.pt --test_data data/breast_cancer/test.jsonl
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import sys
sys.path.append(str(Path(__file__).parent.parent / 'genesis_rna'))

from genesis_rna.breast_cancer import BreastCancerAnalyzer


@dataclass
class EvaluationMetrics:
    """Clinical evaluation metrics for variant classification"""
    # Overall performance
    auc_roc: float
    auc_pr: float
    accuracy: float

    # Clinical metrics
    sensitivity: float  # True positive rate (recall)
    specificity: float  # True negative rate
    ppv: float  # Positive predictive value (precision)
    npv: float  # Negative predictive value

    # Detailed breakdown
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # By variant type
    by_type: Dict[str, Dict] = None

    # VUS analysis (critical for clinical use)
    vus_pathogenic_rate: float = 0.0
    vus_benign_rate: float = 0.0
    vus_uncertain_rate: float = 0.0


def load_test_data(test_path: str) -> List[Dict]:
    """Load test dataset from JSONL file"""
    samples = []
    with open(test_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def evaluate_model(
    analyzer: BreastCancerAnalyzer,
    test_samples: List[Dict],
    confidence_threshold: float = 0.5
) -> EvaluationMetrics:
    """
    Evaluate model on test set

    Args:
        analyzer: Trained BreastCancerAnalyzer
        test_samples: List of test samples
        confidence_threshold: Threshold for binary classification

    Returns:
        EvaluationMetrics with comprehensive results
    """
    print(f"\n🔬 Evaluating model on {len(test_samples)} test samples...")

    y_true = []
    y_pred = []
    y_scores = []

    predictions_by_type = {}
    vus_predictions = []

    for i, sample in enumerate(test_samples):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(test_samples)}")

        # Get prediction
        try:
            pred = analyzer.predict_variant_effect(
                gene=sample['gene'],
                wild_type_rna=sample['wild_type_rna'],
                mutant_rna=sample['mutant_rna'],
                variant_id=sample['variant_id']
            )

            # Ground truth (1 = pathogenic, 0 = benign)
            true_label = sample['label']
            pred_label = 1 if pred.pathogenicity_score > confidence_threshold else 0

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(pred.pathogenicity_score)

            # Track by variant type
            variant_type = sample.get('variant_type', 'unknown')
            if variant_type not in predictions_by_type:
                predictions_by_type[variant_type] = {'true': [], 'pred': [], 'scores': []}

            predictions_by_type[variant_type]['true'].append(true_label)
            predictions_by_type[variant_type]['pred'].append(pred_label)
            predictions_by_type[variant_type]['scores'].append(pred.pathogenicity_score)

            # Track VUS predictions
            if sample.get('clinical_significance') == 'Uncertain significance':
                vus_predictions.append(pred.pathogenicity_score)

        except Exception as e:
            print(f"   ⚠️  Error predicting {sample['variant_id']}: {e}")
            continue

    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # ROC-AUC
    auc_roc = roc_auc_score(y_true, y_scores)

    # PR-AUC (better for imbalanced datasets)
    auc_pr = average_precision_score(y_true, y_scores)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Analyze VUS predictions
    vus_pathogenic_rate = sum(s > 0.7 for s in vus_predictions) / len(vus_predictions) if vus_predictions else 0.0
    vus_benign_rate = sum(s < 0.3 for s in vus_predictions) / len(vus_predictions) if vus_predictions else 0.0
    vus_uncertain_rate = 1.0 - vus_pathogenic_rate - vus_benign_rate

    # Metrics by variant type
    by_type = {}
    for vtype, preds in predictions_by_type.items():
        if len(preds['true']) > 0:
            by_type[vtype] = {
                'count': len(preds['true']),
                'auc': roc_auc_score(preds['true'], preds['scores']) if len(set(preds['true'])) > 1 else 0.0,
                'accuracy': np.mean(np.array(preds['true']) == np.array(preds['pred']))
            }

    return EvaluationMetrics(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        by_type=by_type,
        vus_pathogenic_rate=vus_pathogenic_rate,
        vus_benign_rate=vus_benign_rate,
        vus_uncertain_rate=vus_uncertain_rate
    )


def print_evaluation_report(metrics: EvaluationMetrics):
    """Print comprehensive evaluation report"""
    print("\n" + "="*70)
    print("GENESIS RNA - CANCER VARIANT PREDICTION EVALUATION")
    print("="*70)

    print("\n📊 OVERALL PERFORMANCE")
    print(f"  AUC-ROC:      {metrics.auc_roc:.3f}")
    print(f"  AUC-PR:       {metrics.auc_pr:.3f}")
    print(f"  Accuracy:     {metrics.accuracy:.3f}")

    print("\n🏥 CLINICAL METRICS")
    print(f"  Sensitivity:  {metrics.sensitivity:.3f}  (True positive rate)")
    print(f"  Specificity:  {metrics.specificity:.3f}  (True negative rate)")
    print(f"  PPV:          {metrics.ppv:.3f}  (Positive predictive value)")
    print(f"  NPV:          {metrics.npv:.3f}  (Negative predictive value)")

    print("\n🔢 CONFUSION MATRIX")
    print(f"  True Positives:  {metrics.true_positives}")
    print(f"  True Negatives:  {metrics.true_negatives}")
    print(f"  False Positives: {metrics.false_positives}")
    print(f"  False Negatives: {metrics.false_negatives}")

    if metrics.by_type:
        print("\n🧬 PERFORMANCE BY VARIANT TYPE")
        for vtype, stats in metrics.by_type.items():
            print(f"  {vtype}:")
            print(f"    Count: {stats['count']}")
            print(f"    AUC:   {stats['auc']:.3f}")
            print(f"    Acc:   {stats['accuracy']:.3f}")

    print("\n❓ VUS (VARIANTS OF UNCERTAIN SIGNIFICANCE) ANALYSIS")
    print(f"  Classified as Pathogenic: {metrics.vus_pathogenic_rate*100:.1f}%")
    print(f"  Classified as Benign:     {metrics.vus_benign_rate*100:.1f}%")
    print(f"  Remain Uncertain:         {metrics.vus_uncertain_rate*100:.1f}%")

    print("\n🎯 CLINICAL INTERPRETATION")
    if metrics.sensitivity >= 0.95:
        print("  ✅ Excellent sensitivity - very few pathogenic variants missed")
    elif metrics.sensitivity >= 0.85:
        print("  ✓  Good sensitivity - acceptable for clinical screening")
    else:
        print("  ⚠️  Low sensitivity - may miss pathogenic variants")

    if metrics.specificity >= 0.90:
        print("  ✅ Excellent specificity - low false positive rate")
    elif metrics.specificity >= 0.80:
        print("  ✓  Good specificity - acceptable for clinical use")
    else:
        print("  ⚠️  Low specificity - many false positives")

    print("\n" + "="*70)


def save_results(metrics: EvaluationMetrics, output_path: str):
    """Save evaluation results to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Genesis RNA model on cancer variant prediction'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test data (JSONL format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation_metrics.json',
        help='Output path for evaluation results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for pathogenicity (default: 0.5)'
    )

    args = parser.parse_args()

    print("🧬 Genesis RNA Cancer Evaluation Framework")
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")

    # Load model
    print("\n📥 Loading model...")
    analyzer = BreastCancerAnalyzer(args.model)

    # Load test data
    print(f"📥 Loading test data...")
    test_samples = load_test_data(args.test_data)
    print(f"   Loaded {len(test_samples)} test samples")

    # Evaluate
    metrics = evaluate_model(analyzer, test_samples, args.threshold)

    # Print report
    print_evaluation_report(metrics)

    # Save results
    save_results(metrics, args.output)

    print("\n✅ Evaluation complete!")

    # Return exit code based on performance
    if metrics.auc_roc >= 0.80 and metrics.sensitivity >= 0.85:
        print("🎉 Model achieves clinical-grade performance!")
        return 0
    else:
        print("⚠️  Model performance below clinical threshold. More training may be needed.")
        return 1


if __name__ == '__main__':
    exit(main())
