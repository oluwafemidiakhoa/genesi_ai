#!/usr/bin/env python3
"""
Baseline Models for Variant Classification

Test simple baselines BEFORE using deep learning.
If k-mer counting gets >80%, we don't need transformers!
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_extraction.text import CountVectorizer
import argparse
from pathlib import Path


class KmerBaseline:
    """
    Simple k-mer counting baseline.

    If this gets >80% accuracy, deep learning is overkill!
    """

    def __init__(self, k: int = 3):
        """
        Args:
            k: K-mer size (default: 3 for trinucleotides)
        """
        self.k = k
        self.vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(k, k),
            lowercase=False
        )
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')

    def extract_kmers(self, sequence: str) -> List[str]:
        """Extract k-mers from sequence"""
        return [sequence[i:i+self.k] for i in range(len(sequence)-self.k+1)]

    def fit(self, sequences: List[str], labels: np.ndarray):
        """Train k-mer model"""
        # Vectorize sequences
        X = self.vectorizer.fit_transform(sequences)

        # Train logistic regression
        self.model.fit(X, labels)

    def predict(self, sequences: List[str]) -> np.ndarray:
        """Predict pathogenicity"""
        X = self.vectorizer.transform(sequences)
        return self.model.predict(X)

    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        """Predict probabilities"""
        X = self.vectorizer.transform(sequences)
        return self.model.predict_proba(X)


class NearestNeighborBaseline:
    """
    Nearest neighbor baseline - tests if model is just doing sequence matching.

    If this gets >90%, your model is memorizing, not learning!
    """

    def __init__(self):
        self.train_sequences = []
        self.train_labels = []

    def fit(self, sequences: List[str], labels: np.ndarray):
        """Store training sequences"""
        self.train_sequences = sequences
        self.train_labels = labels

    def sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Simple sequence similarity (ratio of matching characters)"""
        if len(seq1) != len(seq2):
            # Different lengths - use Levenshtein-like approach
            matches = sum(1 for c1, c2 in zip(seq1, seq2) if c1 == c2)
            return matches / max(len(seq1), len(seq2))
        else:
            # Same length - count matches
            matches = sum(1 for c1, c2 in zip(seq1, seq2) if c1 == c2)
            return matches / len(seq1)

    def predict(self, sequences: List[str]) -> np.ndarray:
        """Predict by finding most similar training sequence"""
        predictions = []

        for test_seq in sequences:
            # Find most similar training sequence
            similarities = [
                self.sequence_similarity(test_seq, train_seq)
                for train_seq in self.train_sequences
            ]
            most_similar_idx = np.argmax(similarities)
            predictions.append(self.train_labels[most_similar_idx])

        return np.array(predictions)


class BiologicalFeatureBaseline:
    """
    Biological feature baseline using hand-crafted features.

    Features:
    - GC content
    - Sequence length
    - Nucleotide composition
    - Simple secondary structure indicators
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )

    def extract_features(self, sequence: str) -> np.ndarray:
        """Extract biological features from sequence"""
        seq = sequence.upper()
        length = len(seq)

        if length == 0:
            return np.zeros(10)

        # Nucleotide counts
        count_A = seq.count('A')
        count_C = seq.count('C')
        count_G = seq.count('G')
        count_U = seq.count('U') + seq.count('T')

        # Composition features
        gc_content = (count_G + count_C) / length if length > 0 else 0
        purine_ratio = (count_A + count_G) / length if length > 0 else 0

        # Simple repeats (potential stem-loop indicators)
        has_poly_a = 'AAAA' in seq
        has_poly_c = 'CCCC' in seq
        has_poly_g = 'GGGG' in seq
        has_poly_u = 'UUUU' in seq or 'TTTT' in seq

        features = np.array([
            length,
            gc_content,
            purine_ratio,
            count_A / length if length > 0 else 0,
            count_C / length if length > 0 else 0,
            count_G / length if length > 0 else 0,
            count_U / length if length > 0 else 0,
            float(has_poly_a),
            float(has_poly_c),
            float(has_poly_g + has_poly_u),
        ])

        return features

    def fit(self, sequences: List[str], labels: np.ndarray):
        """Train on biological features"""
        X = np.array([self.extract_features(seq) for seq in sequences])
        self.model.fit(X, labels)

    def predict(self, sequences: List[str]) -> np.ndarray:
        """Predict pathogenicity"""
        X = np.array([self.extract_features(seq) for seq in sequences])
        return self.model.predict(X)

    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        """Predict probabilities"""
        X = np.array([self.extract_features(seq) for seq in sequences])
        return self.model.predict_proba(X)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None, model_name: str = "Model") -> Dict:
    """
    Comprehensive evaluation metrics.

    Returns metrics dict with accuracy, precision, recall, F1, AUC-ROC
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            # AUC-ROC (probability of positive class)
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Clinical metrics
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def run_baseline_comparison(train_csv: str, test_csv: str, output_csv: str = None):
    """
    Run all baseline models and compare performance.

    Args:
        train_csv: Training data CSV with 'RNA_Sequence' and 'Label' columns
        test_csv: Test data CSV
        output_csv: Path to save results (optional)
    """
    print("="*70)
    print("BASELINE MODEL COMPARISON")
    print("="*70)
    print("\nTesting simple baselines BEFORE using deep learning.")
    print("If k-mer counting gets >80%, transformers are overkill!\n")

    # Load data
    print(f"Loading data...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_sequences = train_df['RNA_Sequence'].fillna('').tolist()
    train_labels = train_df['Label'].values

    test_sequences = test_df['RNA_Sequence'].fillna('').tolist()
    test_labels = test_df['Label'].values

    print(f"   Train: {len(train_sequences)} samples")
    print(f"   Test: {len(test_sequences)} samples")
    print(f"   Class distribution (train): {np.bincount(train_labels)}")

    # Baseline 1: K-mer (3-mer)
    print("\n" + "-"*70)
    print("1. K-MER BASELINE (3-mers + Logistic Regression)")
    print("-"*70)

    kmer_model = KmerBaseline(k=3)
    print("   Training...")
    kmer_model.fit(train_sequences, train_labels)
    print("   Predicting...")
    kmer_pred = kmer_model.predict(test_sequences)
    kmer_proba = kmer_model.predict_proba(test_sequences)
    kmer_metrics = evaluate_model(test_labels, kmer_pred, kmer_proba, "K-mer (3-mer)")

    print(f"\n   Results:")
    print(f"   - Accuracy: {kmer_metrics['accuracy']:.3f}")
    print(f"   - AUC-ROC: {kmer_metrics['auc_roc']:.3f}")
    print(f"   - F1 Score: {kmer_metrics['f1']:.3f}")
    print(f"   - Sensitivity: {kmer_metrics.get('sensitivity', 0):.3f}")
    print(f"   - Specificity: {kmer_metrics.get('specificity', 0):.3f}")

    # Baseline 2: Nearest Neighbor
    print("\n" + "-"*70)
    print("2. NEAREST NEIGHBOR BASELINE (Sequence Matching)")
    print("-"*70)
    print("   This tests if model is just memorizing sequences...")

    nn_model = NearestNeighborBaseline()
    print("   Training...")
    nn_model.fit(train_sequences, train_labels)
    print("   Predicting (this may take a while)...")

    # Sample for speed (nearest neighbor is slow)
    test_sample_size = min(1000, len(test_sequences))
    test_sample_indices = np.random.choice(len(test_sequences), test_sample_size, replace=False)
    test_sample_seqs = [test_sequences[i] for i in test_sample_indices]
    test_sample_labels = test_labels[test_sample_indices]

    nn_pred = nn_model.predict(test_sample_seqs)
    nn_metrics = evaluate_model(test_sample_labels, nn_pred, None, "Nearest Neighbor")

    print(f"\n   Results (on {test_sample_size} samples):")
    print(f"   - Accuracy: {nn_metrics['accuracy']:.3f}")
    print(f"   - F1 Score: {nn_metrics['f1']:.3f}")

    if nn_metrics['accuracy'] > 0.85:
        print(f"\n   ⚠️ WARNING: High accuracy suggests model might be memorizing!")
        print(f"      Check for data leakage in train/test split.")

    # Baseline 3: Biological Features
    print("\n" + "-"*70)
    print("3. BIOLOGICAL FEATURE BASELINE (Hand-crafted features + RF)")
    print("-"*70)

    bio_model = BiologicalFeatureBaseline()
    print("   Training...")
    bio_model.fit(train_sequences, train_labels)
    print("   Predicting...")
    bio_pred = bio_model.predict(test_sequences)
    bio_proba = bio_model.predict_proba(test_sequences)
    bio_metrics = evaluate_model(test_labels, bio_pred, bio_proba, "Biological Features")

    print(f"\n   Results:")
    print(f"   - Accuracy: {bio_metrics['accuracy']:.3f}")
    print(f"   - AUC-ROC: {bio_metrics['auc_roc']:.3f}")
    print(f"   - F1 Score: {bio_metrics['f1']:.3f}")
    print(f"   - Sensitivity: {bio_metrics.get('sensitivity', 0):.3f}")
    print(f"   - Specificity: {bio_metrics.get('specificity', 0):.3f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_metrics = [kmer_metrics, nn_metrics, bio_metrics]
    results_df = pd.DataFrame(all_metrics)

    print("\n" + results_df[['model', 'accuracy', 'f1', 'auc_roc']].to_string(index=False))

    # Best model
    best_model = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\n🏆 Best Model: {best_model['model']}")
    print(f"   Accuracy: {best_model['accuracy']:.3f}")
    print(f"   AUC-ROC: {best_model['auc_roc']:.3f}")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    max_acc = results_df['accuracy'].max()

    if max_acc >= 0.85:
        print(f"✅ Best baseline achieved {max_acc:.1%} accuracy!")
        print(f"   Consider using this baseline instead of deep learning.")
        print(f"   Deep learning may not provide significant improvement.")
    elif max_acc >= 0.75:
        print(f"⚠️ Best baseline achieved {max_acc:.1%} accuracy.")
        print(f"   Deep learning MAY provide improvement, but:")
        print(f"   - Start with simple Word2Vec on k-mers")
        print(f"   - Only use transformers if Word2Vec fails")
        print(f"   - Justify the added complexity")
    else:
        print(f"✅ Best baseline only achieved {max_acc:.1%} accuracy.")
        print(f"   Deep learning is justified for this task.")
        print(f"   Expected improvement: 10-20 percentage points")

    print("\n📊 Realistic Expectations:")
    print(f"   - Simple baselines: 65-75% accuracy")
    print(f"   - Word2Vec/CNNs: 75-85% accuracy")
    print(f"   - Transformers: 80-90% accuracy (NOT 100%!)")
    print("="*70)

    # Save results
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to: {output_csv}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Test baseline models before using deep learning'
    )
    parser.add_argument('--train', required=True, help='Training CSV')
    parser.add_argument('--test', required=True, help='Test CSV')
    parser.add_argument('--output', help='Output CSV for results')
    args = parser.parse_args()

    results = run_baseline_comparison(args.train, args.test, args.output)


if __name__ == '__main__':
    main()
