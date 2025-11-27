#!/usr/bin/env python3
"""
Progressive Model Training Strategy

Start simple, add complexity ONLY if needed!

Decision Tree:
1. K-mer + Logistic Regression (baseline)
   └─> If <80%: Try step 2
2. K-mer + Random Forest
   └─> If <82%: Try step 3
3. Word2Vec embeddings + CNN
   └─> If <85%: Try step 4
4. Genesis RNA transformer (only if truly needed!)

This approach:
- Saves computational resources
- Provides proper baselines
- Justifies complexity when used
- Prevents overfitting
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# Import baseline models
import sys
sys.path.insert(0, str(Path(__file__).parent))
from baseline_models import KmerBaseline, BiologicalFeatureBaseline, evaluate_model


class ProgressiveModelTrainer:
    """
    Progressive training strategy: start simple, add complexity only if needed.
    """

    def __init__(self, train_csv: str, test_csv: str, output_dir: str = 'results'):
        """
        Args:
            train_csv: Training data CSV
            test_csv: Test data CSV
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print("Loading data...")
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)

        self.train_sequences = self.train_df['RNA_Sequence'].fillna('').tolist()
        self.train_labels = self.train_df['Label'].values

        self.test_sequences = self.test_df['RNA_Sequence'].fillna('').tolist()
        self.test_labels = self.test_df['Label'].values

        print(f"   Train: {len(self.train_sequences)} samples")
        print(f"   Test: {len(self.test_sequences)} samples")

        self.results = []

    def step1_kmer_logistic_regression(self) -> Dict:
        """
        Step 1: K-mer + Logistic Regression (simplest baseline)

        Target: 75-80% accuracy
        If achieved: May be sufficient for task!
        If not: Try Random Forest
        """
        print("\n" + "="*70)
        print("STEP 1: K-MER + LOGISTIC REGRESSION")
        print("="*70)
        print("Simplest possible model - just counting trinucleotides!")

        model = KmerBaseline(k=3)
        print("Training...")
        model.fit(self.train_sequences, self.train_labels)

        print("Evaluating...")
        predictions = model.predict(self.test_sequences)
        probabilities = model.predict_proba(self.test_sequences)

        metrics = evaluate_model(
            self.test_labels, predictions, probabilities,
            "K-mer + Logistic Regression"
        )

        self._print_metrics(metrics)

        # Decision
        if metrics['accuracy'] >= 0.80:
            print("\n✅ DECISION: 80%+ accuracy achieved!")
            print("   This simple model is sufficient for the task.")
            print("   No need for complex deep learning.")
            print("   STOP HERE - use this model!")
            return metrics

        elif metrics['accuracy'] >= 0.75:
            print("\n⚠️ DECISION: 75-80% accuracy achieved.")
            print("   Good performance, but try Random Forest for potential improvement.")
            print("   PROCEED TO STEP 2")
        else:
            print("\n➡️ DECISION: <75% accuracy.")
            print("   Need more sophisticated model.")
            print("   PROCEED TO STEP 2")

        self.results.append(metrics)
        return metrics

    def step2_kmer_random_forest(self) -> Dict:
        """
        Step 2: K-mer + Random Forest (non-linear relationships)

        Target: 80-82% accuracy
        If achieved: Good enough for most tasks
        If not: Try Word2Vec
        """
        print("\n" + "="*70)
        print("STEP 2: K-MER + RANDOM FOREST")
        print("="*70)
        print("Adding non-linear modeling capability...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import CountVectorizer

        # Extract k-mer features
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3), lowercase=False)
        X_train = vectorizer.fit_transform(self.train_sequences)
        X_test = vectorizer.transform(self.test_sequences)

        # Train Random Forest
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, self.train_labels)

        print("Evaluating...")
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        metrics = evaluate_model(
            self.test_labels, predictions, probabilities,
            "K-mer + Random Forest"
        )

        self._print_metrics(metrics)

        # Decision
        if metrics['accuracy'] >= 0.82:
            print("\n✅ DECISION: 82%+ accuracy achieved!")
            print("   Random Forest provides good performance.")
            print("   Deep learning may not provide significant improvement.")
            print("   RECOMMEND: Use this model unless >85% is required.")
            return metrics

        elif metrics['accuracy'] >= 0.78:
            print("\n⚠️ DECISION: 78-82% accuracy achieved.")
            print("   Decent performance. Deep learning MAY help.")
            print("   Try Word2Vec embeddings next.")
            print("   PROCEED TO STEP 3 (optional)")
        else:
            print("\n➡️ DECISION: <78% accuracy.")
            print("   Need representation learning.")
            print("   PROCEED TO STEP 3")

        self.results.append(metrics)
        return metrics

    def step3_word2vec_cnn(self) -> Optional[Dict]:
        """
        Step 3: Word2Vec embeddings + CNN (representation learning)

        Target: 82-85% accuracy
        If achieved: Excellent performance
        If not: Consider transformers (but justify!)
        """
        print("\n" + "="*70)
        print("STEP 3: WORD2VEC + CNN")
        print("="*70)
        print("Learning sequence representations...")
        print("\n⚠️ NOTE: This requires gensim and PyTorch.")
        print("   Skipping for now - implement if Steps 1-2 fail.")
        print("   Expected improvement: +3-5% accuracy over Random Forest")

        # Placeholder - would implement Word2Vec + CNN here
        # For now, skip to avoid dependencies

        print("\n➡️ DECISION: Implementation pending")
        print("   If Random Forest <80%, implement this step.")
        print("   STOP HERE for now")

        return None

    def step4_transformer(self) -> Optional[Dict]:
        """
        Step 4: Genesis RNA Transformer (only if truly needed!)

        Target: 85-90% accuracy
        Only justified if:
        - Simpler methods all failed (<80%)
        - Task requires very high accuracy (>85%)
        - Willing to pay computational cost
        """
        print("\n" + "="*70)
        print("STEP 4: GENESIS RNA TRANSFORMER")
        print("="*70)
        print("Full transformer architecture...")
        print("\n⚠️ NOTE: This is the most complex approach.")
        print("   Only use if:")
        print("   1. Steps 1-3 all achieved <80%")
        print("   2. Task requires >85% accuracy")
        print("   3. Computational resources available")

        print("\n📊 Expected Performance:")
        print("   - K-mer baseline: 75-80%")
        print("   - Random Forest: 78-82%")
        print("   - Word2Vec + CNN: 82-85%")
        print("   - Transformer: 85-90% (NOT 100%!)")

        print("\n⚠️ IMPORTANT: If you get 100% accuracy with transformer:")
        print("   This is DATA LEAKAGE, not good performance!")
        print("   Audit for label leakage immediately.")

        return None

    def _print_metrics(self, metrics: Dict):
        """Print evaluation metrics"""
        print(f"\n📊 Results:")
        print(f"   Accuracy:    {metrics['accuracy']:.3f}")
        print(f"   F1 Score:    {metrics['f1']:.3f}")
        print(f"   AUC-ROC:     {metrics['auc_roc']:.3f}")
        print(f"   Precision:   {metrics['precision']:.3f}")
        print(f"   Recall:      {metrics['recall']:.3f}")

        if 'sensitivity' in metrics:
            print(f"   Sensitivity: {metrics['sensitivity']:.3f}")
            print(f"   Specificity: {metrics['specificity']:.3f}")

    def run_progressive_training(self) -> Dict:
        """
        Run progressive training strategy.

        Returns final metrics from best model.
        """
        print("\n" + "="*70)
        print("PROGRESSIVE MODEL TRAINING")
        print("="*70)
        print("\nStrategy: Start simple, add complexity ONLY if needed!")
        print("\nDecision Tree:")
        print("1. K-mer + Logistic Regression")
        print("   └─> If ≥80%: STOP (good enough!)")
        print("2. K-mer + Random Forest")
        print("   └─> If ≥82%: STOP (excellent!)")
        print("3. Word2Vec + CNN")
        print("   └─> If ≥85%: STOP (outstanding!)")
        print("4. Transformer (only if truly needed)")
        print("\n" + "="*70)

        # Step 1
        step1_metrics = self.step1_kmer_logistic_regression()

        if step1_metrics['accuracy'] >= 0.80:
            print("\n🎉 Task solved with simplest model!")
            self._save_results()
            return step1_metrics

        # Step 2
        step2_metrics = self.step2_kmer_random_forest()

        if step2_metrics['accuracy'] >= 0.82:
            print("\n🎉 Random Forest provides excellent performance!")
            print("   Deep learning not justified.")
            self._save_results()
            return step2_metrics

        # Step 3 (not implemented yet)
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)

        best_acc = max(step1_metrics['accuracy'], step2_metrics['accuracy'])

        if best_acc >= 0.75:
            print(f"✅ Best model achieved {best_acc:.1%} accuracy.")
            print(f"   This is reasonable for variant prediction.")
            print(f"   Consider using this model in production.")
            print(f"\n💡 To improve further:")
            print(f"   1. Add biological features (conservation, structure)")
            print(f"   2. Implement Word2Vec embeddings")
            print(f"   3. Ensemble multiple models")
        else:
            print(f"⚠️ Best model only achieved {best_acc:.1%} accuracy.")
            print(f"   Need more sophisticated approach:")
            print(f"   1. Implement Word2Vec + CNN (Step 3)")
            print(f"   2. Add biological features")
            print(f"   3. Consider Genesis RNA transformer")
            print(f"   4. Verify data quality (check for errors)")

        print("\n📊 Realistic Expectations:")
        print("   - K-mer methods: 70-80%")
        print("   - Ensemble methods: 75-85%")
        print("   - Deep learning: 80-90%")
        print("   - 100% = DATA LEAKAGE (not real performance!)")

        self._save_results()

        return step2_metrics

    def _save_results(self):
        """Save all results to JSON"""
        results_file = self.output_dir / 'progressive_training_results.json'

        results_dict = {
            'models': self.results,
            'best_model': max(self.results, key=lambda x: x['accuracy'])['model'],
            'best_accuracy': max(self.results, key=lambda x: x['accuracy'])['accuracy']
        }

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Progressive training: start simple, add complexity only if needed'
    )
    parser.add_argument('--train', required=True, help='Training CSV')
    parser.add_argument('--test', required=True, help='Test CSV')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    args = parser.parse_args()

    trainer = ProgressiveModelTrainer(args.train, args.test, args.output_dir)
    final_metrics = trainer.run_progressive_training()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best accuracy: {final_metrics['accuracy']:.1%}")
    print(f"Best model: {final_metrics['model']}")
    print("\nUse this model unless higher accuracy is required!")
    print("="*70)


if __name__ == '__main__':
    main()
