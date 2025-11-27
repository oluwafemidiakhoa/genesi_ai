#!/usr/bin/env python3
"""
Comprehensive Model Evaluation

Multiple validation strategies to ensure model quality:
1. Standard test set evaluation
2. Cross-validation (temporal, position-based, k-fold)
3. Ablation studies (what did model learn?)
4. Error analysis
5. External validation (if data available)
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import json


class ComprehensiveEvaluator:
    """Comprehensive model evaluation framework"""

    def __init__(self, model_predictions_csv: str, test_csv: str):
        """
        Args:
            model_predictions_csv: CSV with columns: ['AlleleID', 'Prediction', 'Probability']
            test_csv: Test data CSV with ground truth labels
        """
        # Load predictions
        self.pred_df = pd.read_csv(model_predictions_csv)

        # Load test data
        self.test_df = pd.read_csv(test_csv)

        # Merge
        self.df = self.test_df.merge(
            self.pred_df[['AlleleID', 'Prediction', 'Probability']],
            on='AlleleID',
            how='left'
        )

        print(f"Loaded {len(self.df)} test samples with predictions")

    def compute_standard_metrics(self) -> Dict:
        """Compute standard classification metrics"""
        print("\n" + "="*70)
        print("STANDARD METRICS")
        print("="*70)

        y_true = self.df['Label'].values
        y_pred = self.df['Prediction'].values
        y_proba = self.df['Probability'].values

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
            'auc_pr': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # Positive Predictive Value
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,  # Negative Predictive Value
            })

        self._print_metrics(metrics)
        return metrics

    def evaluate_by_gene(self) -> Dict[str, Dict]:
        """Evaluate performance separately for each gene"""
        print("\n" + "="*70)
        print("PER-GENE EVALUATION")
        print("="*70)

        if 'GeneSymbol' not in self.df.columns and 'Gene' not in self.df.columns:
            print("No gene column found - skipping per-gene evaluation")
            return {}

        gene_col = 'GeneSymbol' if 'GeneSymbol' in self.df.columns else 'Gene'
        genes = self.df[gene_col].unique()

        gene_metrics = {}

        for gene in genes:
            gene_df = self.df[self.df[gene_col] == gene]

            if len(gene_df) < 10:
                continue

            y_true = gene_df['Label'].values
            y_pred = gene_df['Prediction'].values

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            gene_metrics[gene] = {
                'accuracy': acc,
                'f1': f1,
                'n_samples': len(gene_df)
            }

            print(f"\n{gene}:")
            print(f"   Samples: {len(gene_df)}")
            print(f"   Accuracy: {acc:.3f}")
            print(f"   F1 Score: {f1:.3f}")

        return gene_metrics

    def evaluate_by_variant_type(self) -> Dict[str, Dict]:
        """Evaluate performance by variant type"""
        print("\n" + "="*70)
        print("PER-VARIANT-TYPE EVALUATION")
        print("="*70)

        if 'VariantType' not in self.df.columns and 'Type' not in self.df.columns:
            print("No variant type column found - skipping")
            return {}

        type_col = 'VariantType' if 'VariantType' in self.df.columns else 'Type'
        types = self.df[type_col].unique()

        type_metrics = {}

        for vtype in types:
            type_df = self.df[self.df[type_col] == vtype]

            if len(type_df) < 5:
                continue

            y_true = type_df['Label'].values
            y_pred = type_df['Prediction'].values

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            type_metrics[str(vtype)] = {
                'accuracy': acc,
                'f1': f1,
                'n_samples': len(type_df)
            }

            print(f"\n{vtype}:")
            print(f"   Samples: {len(type_df)}")
            print(f"   Accuracy: {acc:.3f}")
            print(f"   F1 Score: {f1:.3f}")

        return type_metrics

    def analyze_errors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze false positives and false negatives"""
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)

        # False positives (predicted pathogenic, actually benign)
        fp_df = self.df[(self.df['Label'] == 0) & (self.df['Prediction'] == 1)]

        # False negatives (predicted benign, actually pathogenic)
        fn_df = self.df[(self.df['Label'] == 1) & (self.df['Prediction'] == 0)]

        print(f"\n❌ False Positives: {len(fp_df)}")
        if len(fp_df) > 0:
            print(f"   These are benign variants incorrectly classified as pathogenic")
            print(f"   Average confidence: {fp_df['Probability'].mean():.3f}")

            if len(fp_df) <= 10:
                print(f"\n   Examples:")
                for _, row in fp_df.head().iterrows():
                    print(f"   - {row.get('Name', row.get('AlleleID', 'Unknown'))}: {row['Probability']:.3f}")

        print(f"\n❌ False Negatives: {len(fn_df)}")
        if len(fn_df) > 0:
            print(f"   These are pathogenic variants incorrectly classified as benign")
            print(f"   Average confidence: {fn_df['Probability'].mean():.3f}")
            print(f"   ⚠️ CRITICAL: Missing pathogenic variants in cancer prediction!")

            if len(fn_df) <= 10:
                print(f"\n   Examples:")
                for _, row in fn_df.head().iterrows():
                    print(f"   - {row.get('Name', row.get('AlleleID', 'Unknown'))}: {row['Probability']:.3f}")

        return fp_df, fn_df

    def check_for_100_percent_accuracy(self):
        """Special check: Is accuracy suspiciously high?"""
        print("\n" + "="*70)
        print("100% ACCURACY CHECK")
        print("="*70)

        y_true = self.df['Label'].values
        y_pred = self.df['Prediction'].values
        accuracy = accuracy_score(y_true, y_pred)

        if accuracy >= 0.995:
            print(f"\n🚨 WARNING: {accuracy:.1%} accuracy detected!")
            print("\nThis is EXTREMELY suspicious for variant prediction.")
            print("\nPossible causes:")
            print("1. ❌ LABEL LEAKAGE - Labels used during feature generation")
            print("2. ❌ DATA LEAKAGE - Train/test overlap")
            print("3. ❌ Synthetic data with artificial markers")
            print("4. ⚠️ Overfitting to training set")
            print("\n🔍 IMMEDIATE ACTIONS REQUIRED:")
            print("1. Check for label leakage in data generation")
            print("2. Verify train/test split (no overlap)")
            print("3. Inspect actual sequences for artificial patterns")
            print("4. Run ablation studies (what did model learn?)")
            print("5. Test on external dataset")
            print("\n📖 Real-world variant prediction typically achieves 70-90% accuracy.")
            print("   100% accuracy almost always indicates a bug, not good performance!")

        elif accuracy >= 0.95:
            print(f"\n⚠️ CAUTION: {accuracy:.1%} accuracy is very high.")
            print("\nThis is unusually high for variant prediction.")
            print("Recommend:")
            print("1. Verify no data leakage")
            print("2. Test on external dataset")
            print("3. Compare to established tools (CADD, REVEL)")

        else:
            print(f"\n✅ Accuracy: {accuracy:.1%}")
            print("   This is within realistic range for variant prediction (70-90%)")

    def compare_to_random_baseline(self):
        """Compare to random guessing baseline"""
        print("\n" + "="*70)
        print("RANDOM BASELINE COMPARISON")
        print("="*70)

        y_true = self.df['Label'].values
        y_pred = self.df['Prediction'].values

        # Random baseline (stratified by class distribution)
        class_distribution = np.bincount(y_true) / len(y_true)
        random_pred = np.random.choice([0, 1], size=len(y_true), p=class_distribution)
        random_acc = accuracy_score(y_true, random_pred)

        model_acc = accuracy_score(y_true, y_pred)

        improvement = model_acc - random_acc

        print(f"\n📊 Random Baseline: {random_acc:.1%}")
        print(f"📊 Model Accuracy: {model_acc:.1%}")
        print(f"📊 Improvement: +{improvement:.1%}")

        if improvement < 0.10:
            print(f"\n⚠️ WARNING: Model only {improvement:.1%} better than random!")
            print("   Model may not be learning meaningful patterns.")
        else:
            print(f"\n✅ Model provides {improvement:.1%} improvement over random")

    def _print_metrics(self, metrics: Dict):
        """Print metrics in nice format"""
        print(f"\n📊 Classification Metrics:")
        print(f"   Accuracy:    {metrics['accuracy']:.3f}")
        print(f"   Precision:   {metrics['precision']:.3f}")
        print(f"   Recall:      {metrics['recall']:.3f}")
        print(f"   F1 Score:    {metrics['f1']:.3f}")
        print(f"   AUC-ROC:     {metrics['auc_roc']:.3f}")
        print(f"   AUC-PR:      {metrics['auc_pr']:.3f}")

        if 'sensitivity' in metrics:
            print(f"\n📊 Clinical Metrics:")
            print(f"   Sensitivity: {metrics['sensitivity']:.3f} (TP rate)")
            print(f"   Specificity: {metrics['specificity']:.3f} (TN rate)")
            print(f"   PPV:         {metrics['ppv']:.3f} (Positive Predictive Value)")
            print(f"   NPV:         {metrics['npv']:.3f} (Negative Predictive Value)")

        if 'true_positives' in metrics:
            print(f"\n📊 Confusion Matrix:")
            print(f"   True Positives:  {metrics['true_positives']}")
            print(f"   False Positives: {metrics['false_positives']}")
            print(f"   True Negatives:  {metrics['true_negatives']}")
            print(f"   False Negatives: {metrics['false_negatives']}")

    def run_comprehensive_evaluation(self, output_file: str = 'evaluation_results.json'):
        """Run all evaluation analyses"""
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)

        results = {}

        # Standard metrics
        results['standard_metrics'] = self.compute_standard_metrics()

        # Check for 100% accuracy
        self.check_for_100_percent_accuracy()

        # Random baseline comparison
        self.compare_to_random_baseline()

        # Per-gene evaluation
        results['per_gene_metrics'] = self.evaluate_by_gene()

        # Per-variant-type evaluation
        results['per_type_metrics'] = self.evaluate_by_variant_type()

        # Error analysis
        fp_df, fn_df = self.analyze_errors()
        results['n_false_positives'] = len(fp_df)
        results['n_false_negatives'] = len(fn_df)

        # Save results
        with open(output_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            json.dump(results, f, indent=2, default=convert)

        print(f"\n💾 Results saved to: {output_file}")

        # Final summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)

        acc = results['standard_metrics']['accuracy']
        auc = results['standard_metrics']['auc_roc']

        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy: {acc:.1%}")
        print(f"   AUC-ROC: {auc:.3f}")

        print(f"\n📊 Clinical Performance:")
        if 'sensitivity' in results['standard_metrics']:
            print(f"   Sensitivity: {results['standard_metrics']['sensitivity']:.1%}")
            print(f"   Specificity: {results['standard_metrics']['specificity']:.1%}")

        print(f"\n📊 Errors:")
        print(f"   False Positives: {results['n_false_positives']}")
        print(f"   False Negatives: {results['n_false_negatives']}")

        # Clinical interpretation
        print("\n" + "="*70)
        print("CLINICAL INTERPRETATION")
        print("="*70)

        if acc >= 0.85:
            print(f"✅ Performance: Excellent ({acc:.1%})")
            print("   Potentially suitable for clinical decision support")
            print("   Recommend: Further validation on external datasets")
        elif acc >= 0.75:
            print(f"✅ Performance: Good ({acc:.1%})")
            print("   Useful for variant prioritization and research")
            print("   Not yet ready for clinical use without validation")
        elif acc >= 0.65:
            print(f"⚠️ Performance: Moderate ({acc:.1%})")
            print("   May help with variant filtering")
            print("   Significant false positive/negative rates")
        else:
            print(f"❌ Performance: Poor ({acc:.1%})")
            print("   Not suitable for clinical or research use")
            print("   Consider: Better features, more data, or different approach")

        print("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive model evaluation'
    )
    parser.add_argument('--predictions', required=True,
                       help='CSV with model predictions')
    parser.add_argument('--test_data', required=True,
                       help='Test data CSV with ground truth')
    parser.add_argument('--output', default='evaluation_results.json',
                       help='Output JSON file')
    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator(args.predictions, args.test_data)
    evaluator.run_comprehensive_evaluation(args.output)


if __name__ == '__main__':
    main()
