#!/usr/bin/env python3
"""
Test simple baseline to see if Genesis RNA is adding value beyond sequence composition.

This script tests if simple k-mer/composition features achieve similar accuracy
to the full Genesis RNA model. If they do, the model isn't learning variant effects.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import argparse
from pathlib import Path


def extract_simple_features(sequence):
    """
    Extract trivial sequence composition features.

    If these features alone get high accuracy, the model is just detecting
    sequence type (ncRNA vs coding) rather than learning variant effects.
    """
    if not sequence or len(sequence) == 0:
        return [0] * 10

    # Basic composition
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    length = len(sequence)

    # Coding-specific features
    aug_count = sequence.count('AUG') / max(1, len(sequence) - 2)  # Start codons
    uaa_count = sequence.count('UAA') / max(1, len(sequence) - 2)  # Stop codons
    uag_count = sequence.count('UAG') / max(1, len(sequence) - 2)
    uga_count = sequence.count('UGA') / max(1, len(sequence) - 2)

    # K-mer features (3-mers common in coding)
    coding_3mers = ['AUG', 'UAA', 'UAG', 'UGA', 'GCC', 'GCU', 'UGC', 'CGC']
    coding_score = sum(sequence.count(kmer) for kmer in coding_3mers) / max(1, len(sequence) - 2)

    # Codon usage bias (coding sequences have periodic patterns)
    codon_bias = 0
    if len(sequence) >= 6:
        pos1 = [sequence[i] for i in range(0, len(sequence), 3)]
        pos2 = [sequence[i] for i in range(1, len(sequence), 3)]
        pos3 = [sequence[i] for i in range(2, len(sequence), 3)]

        # Coding sequences have position-specific nucleotide bias
        codon_bias = (
            abs(pos1.count('G') / len(pos1) - 0.25) +
            abs(pos2.count('C') / len(pos2) - 0.25) +
            abs(pos3.count('U') / len(pos3) - 0.25)
        )

    # Base frequencies
    a_freq = sequence.count('A') / len(sequence)
    u_freq = sequence.count('U') / len(sequence)

    return [
        gc_content,
        length,
        aug_count,
        uaa_count + uag_count + uga_count,  # Total stop codons
        coding_score,
        codon_bias,
        a_freq,
        u_freq,
        np.log(length + 1),  # Log length (another indicator)
        gc_content * codon_bias  # Interaction term
    ]


def main():
    parser = argparse.ArgumentParser(description='Test simple baseline classifier')
    parser.add_argument('--data', required=True, help='Path to ClinVar CSV with RNA_Sequence column')
    parser.add_argument('--output', default='baseline_results.txt', help='Output file for results')
    args = parser.parse_args()

    print("="*70)
    print("SIMPLE BASELINE TEST")
    print("="*70)
    print("\nTesting if trivial sequence features achieve similar accuracy to Genesis RNA")
    print("If baseline gets >90% accuracy, Genesis RNA isn't adding value\n")

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    if 'RNA_Sequence' not in df.columns or 'Label' not in df.columns:
        print("ERROR: CSV must have 'RNA_Sequence' and 'Label' columns")
        return

    print(f"Loaded {len(df):,} variants")

    # Extract simple features
    print("\nExtracting simple sequence composition features...")
    print("Features: GC content, length, start/stop codons, k-mers, codon bias")

    features = []
    for seq in df['RNA_Sequence']:
        features.append(extract_simple_features(seq))

    X = np.array(features)
    y = df['Label'].values

    print(f"Feature matrix shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train):,} variants")
    print(f"Test set: {len(X_test):,} variants")

    # Train simple Random Forest
    print("\nTraining baseline Random Forest (100 trees)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # Evaluate
    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"AUC-ROC: {auc:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic']))

    # Feature importance
    feature_names = [
        'GC_content', 'Length', 'Start_codons', 'Stop_codons',
        'Coding_kmers', 'Codon_bias', 'A_freq', 'U_freq',
        'Log_length', 'GC_x_bias'
    ]

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 5 Most Important Features:")
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]:<20} {importances[idx]:.4f}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if accuracy >= 0.90:
        print("\n❌ ISSUE CONFIRMED!")
        print(f"   Baseline achieves {accuracy*100:.1f}% accuracy with trivial features")
        print("   Genesis RNA model is likely NOT learning variant effects")
        print("   It's just detecting 'coding vs ncRNA' sequence patterns")
        print("\n   RECOMMENDATION: Retrain on coding sequences")
    elif accuracy >= 0.75:
        print("\n⚠️ PARTIAL ISSUE")
        print(f"   Baseline achieves {accuracy*100:.1f}% accuracy")
        print("   Genesis RNA (100%) is better, but gap is small")
        print("   Model may be using some sequence composition shortcuts")
        print("\n   RECOMMENDATION: Add ablation studies, test on external data")
    else:
        print("\n✅ GENESIS RNA ADDS VALUE")
        print(f"   Baseline only achieves {accuracy*100:.1f}% accuracy")
        print("   Genesis RNA (100%) is substantially better")
        print("   Model appears to be learning genuine variant effects")
        print("\n   RECOMMENDATION: Validate with external datasets")

    # Save results
    with open(args.output, 'w') as f:
        f.write("SIMPLE BASELINE TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
        f.write(f"AUC-ROC: {auc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic']))
        f.write("\n\nFeature Importance:\n")
        for i in range(len(feature_names)):
            idx = indices[i]
            f.write(f"{feature_names[idx]:<20} {importances[idx]:.4f}\n")

    print(f"\n💾 Results saved to: {args.output}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
