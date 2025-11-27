#!/usr/bin/env python3
"""
Audit for data leakage in train/test split.

Checks for:
1. Exact duplicate sequences in train and test
2. Near-duplicate sequences (>95% similar)
3. Same variants in different splits
4. Overlapping genomic positions
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import argparse
from collections import defaultdict


def check_exact_duplicates(train_df, test_df):
    """Check for exact sequence duplicates between train and test"""
    train_seqs = set(train_df['RNA_Sequence'])
    test_seqs = set(test_df['RNA_Sequence'])
    overlap = train_seqs & test_seqs

    print("\n" + "="*70)
    print("1. EXACT DUPLICATE SEQUENCES")
    print("="*70)
    print(f"Train sequences: {len(train_seqs):,}")
    print(f"Test sequences: {len(test_seqs):,}")
    print(f"Exact duplicates: {len(overlap):,}")

    if len(overlap) > 0:
        pct = (len(overlap) / len(test_seqs)) * 100
        print(f"\n❌ DATA LEAKAGE DETECTED!")
        print(f"   {pct:.1f}% of test sequences appear in training")
        print(f"   Model can memorize these sequences")
        return False
    else:
        print(f"\n✅ No exact duplicates found")
        return True


def check_near_duplicates(train_df, test_df, threshold=0.95, sample_size=1000):
    """Check for near-duplicate sequences (>95% similar)"""
    print("\n" + "="*70)
    print("2. NEAR-DUPLICATE SEQUENCES (>95% similar)")
    print("="*70)
    print(f"Sampling {sample_size} test sequences for similarity check...")

    # Sample for performance
    test_sample = test_df.sample(min(sample_size, len(test_df)))
    train_seqs = train_df['RNA_Sequence'].values

    near_duplicates = 0
    similarities = []

    for test_seq in test_sample['RNA_Sequence']:
        # Find most similar training sequence
        max_sim = 0
        for train_seq in train_seqs[:1000]:  # Sample train too
            sim = SequenceMatcher(None, test_seq, train_seq).ratio()
            if sim > max_sim:
                max_sim = sim

        similarities.append(max_sim)
        if max_sim > threshold:
            near_duplicates += 1

    pct = (near_duplicates / len(test_sample)) * 100
    avg_sim = np.mean(similarities)

    print(f"\nAverage similarity: {avg_sim:.3f}")
    print(f"Near-duplicates (>{threshold}): {near_duplicates} ({pct:.1f}%)")

    if pct > 10:
        print(f"\n⚠️ HIGH SIMILARITY DETECTED!")
        print(f"   Many test sequences are very similar to training")
        print(f"   Model may be doing nearest-neighbor matching")
        return False
    else:
        print(f"\n✅ Similarity levels acceptable")
        return True


def check_variant_overlap(train_df, test_df):
    """Check if same variants appear in train and test"""
    print("\n" + "="*70)
    print("3. VARIANT ID OVERLAP")
    print("="*70)

    if 'Name' in train_df.columns:
        train_variants = set(train_df['Name'])
        test_variants = set(test_df['Name'])
        overlap = train_variants & test_variants

        print(f"Train variants: {len(train_variants):,}")
        print(f"Test variants: {len(test_variants):,}")
        print(f"Overlapping variants: {len(overlap):,}")

        if len(overlap) > 0:
            pct = (len(overlap) / len(test_variants)) * 100
            print(f"\n❌ VARIANT LEAKAGE DETECTED!")
            print(f"   {pct:.1f}% of test variants appear in training")
            return False
        else:
            print(f"\n✅ No variant overlap")
            return True
    else:
        print("⚠️ No 'Name' column - cannot check variant overlap")
        return None


def check_position_overlap(train_df, test_df):
    """Check if same genomic positions appear in train and test"""
    print("\n" + "="*70)
    print("4. GENOMIC POSITION OVERLAP")
    print("="*70)

    if 'Gene' in train_df.columns and 'Start' in train_df.columns:
        # Create gene:position identifiers
        train_positions = set(
            f"{row['GeneSymbol']}:{row['Start']}"
            for _, row in train_df.iterrows()
            if 'GeneSymbol' in train_df.columns
        )
        test_positions = set(
            f"{row['GeneSymbol']}:{row['Start']}"
            for _, row in test_df.iterrows()
            if 'GeneSymbol' in test_df.columns
        )

        overlap = train_positions & test_positions

        print(f"Train positions: {len(train_positions):,}")
        print(f"Test positions: {len(test_positions):,}")
        print(f"Overlapping positions: {len(overlap):,}")

        if len(overlap) > 0:
            pct = (len(overlap) / len(test_positions)) * 100
            print(f"\n⚠️ POSITION OVERLAP: {pct:.1f}%")
            print(f"   Same genomic positions in train and test")
            print(f"   Model may memorize position-specific patterns")
            return False
        else:
            print(f"\n✅ No position overlap")
            return True
    else:
        print("⚠️ No position columns - cannot check position overlap")
        return None


def check_label_distribution(train_df, test_df):
    """Check if train and test have similar label distributions"""
    print("\n" + "="*70)
    print("5. LABEL DISTRIBUTION")
    print("="*70)

    train_dist = train_df['Label'].value_counts(normalize=True)
    test_dist = test_df['Label'].value_counts(normalize=True)

    print(f"\nTrain distribution:")
    print(f"  Benign: {train_dist.get(0, 0):.1%}")
    print(f"  Pathogenic: {train_dist.get(1, 0):.1%}")

    print(f"\nTest distribution:")
    print(f"  Benign: {test_dist.get(0, 0):.1%}")
    print(f"  Pathogenic: {test_dist.get(1, 0):.1%}")

    # Check if distributions are very different
    diff = abs(train_dist.get(1, 0) - test_dist.get(1, 0))

    if diff > 0.2:
        print(f"\n⚠️ DISTRIBUTION MISMATCH: {diff:.1%} difference")
        print(f"   Train and test have very different label distributions")
        print(f"   This may indicate improper splitting")
        return False
    else:
        print(f"\n✅ Distributions are similar (diff: {diff:.1%})")
        return True


def recommend_proper_split(df):
    """Recommend proper splitting strategies"""
    print("\n" + "="*70)
    print("6. RECOMMENDED SPLITTING STRATEGIES")
    print("="*70)

    strategies = []

    # Check if temporal split is possible
    if 'DateLastEvaluated' in df.columns or 'LastEvaluated' in df.columns:
        strategies.append("✅ TEMPORAL SPLIT (Recommended)")
        strategies.append("   Train: Variants before 2020")
        strategies.append("   Test: Variants from 2020 onwards")
        strategies.append("   Prevents memorization, tests generalization")

    # Check if position split is possible
    if 'Start' in df.columns:
        strategies.append("\n✅ POSITION-BASED SPLIT")
        strategies.append("   Train: First half of each gene")
        strategies.append("   Test: Second half of each gene")
        strategies.append("   Tests spatial generalization")

    # Check if variant type split is possible
    if 'Type' in df.columns:
        strategies.append("\n✅ VARIANT TYPE SPLIT")
        strategies.append("   Train: Missense + Frameshift")
        strategies.append("   Test: Nonsense + Splice site")
        strategies.append("   Tests type generalization")

    # Gene-based split
    if 'GeneSymbol' in df.columns or 'Gene' in df.columns:
        strategies.append("\n✅ LEAVE-ONE-GENE-OUT")
        strategies.append("   Train: BRCA1 variants")
        strategies.append("   Test: BRCA2 variants")
        strategies.append("   Tests cross-gene generalization")

    print("\n".join(strategies))

    print("\n\n❌ AVOID: Random stratified split")
    print("   This allows data leakage and memorization")


def main():
    parser = argparse.ArgumentParser(description='Audit for data leakage')
    parser.add_argument('--train', required=True, help='Training CSV file')
    parser.add_argument('--test', required=True, help='Test CSV file')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Similarity threshold for near-duplicates')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Sample size for similarity check')
    args = parser.parse_args()

    print("="*70)
    print("DATA LEAKAGE AUDIT")
    print("="*70)
    print(f"\nTrain file: {args.train}")
    print(f"Test file: {args.test}")

    # Load data
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    print(f"\nLoaded:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")

    # Run all checks
    results = {}
    results['exact_duplicates'] = check_exact_duplicates(train_df, test_df)
    results['near_duplicates'] = check_near_duplicates(
        train_df, test_df, args.threshold, args.sample_size
    )
    results['variant_overlap'] = check_variant_overlap(train_df, test_df)
    results['position_overlap'] = check_position_overlap(train_df, test_df)
    results['label_distribution'] = check_label_distribution(train_df, test_df)

    # Recommendations
    recommend_proper_split(pd.concat([train_df, test_df]))

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)

    print(f"\nPassed checks: {passed}/5")
    print(f"Failed checks: {failed}/5")

    if failed > 0:
        print("\n❌ DATA LEAKAGE DETECTED!")
        print("   Current train/test split is invalid")
        print("   Re-split data using recommended strategies above")
        print("\n   This explains the 100% accuracy!")
    elif passed == 5:
        print("\n✅ NO OBVIOUS LEAKAGE DETECTED")
        print("   Split appears valid (though 100% accuracy still suspicious)")
        print("   Consider testing on external dataset for validation")
    else:
        print("\n⚠️ SOME CHECKS COULD NOT BE COMPLETED")
        print("   Review results above for details")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
