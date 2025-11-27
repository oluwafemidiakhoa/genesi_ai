#!/usr/bin/env python3
"""
Proper Train/Test Splitting Strategies

Implements multiple splitting strategies to prevent data leakage:
1. Temporal split (train on old variants, test on new)
2. Position-based split (train on first half of gene, test on second half)
3. Variant type split (train on missense, test on frameshift)
4. Leave-one-gene-out (train on BRCA1, test on BRCA2)

NEVER use random stratified split - allows memorization!
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple


def temporal_split(df: pd.DataFrame, split_date: str = '2020-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by submission date - prevents memorization.

    Train on variants submitted before split_date, test on newer variants.
    This simulates real-world scenario: predict pathogenicity of newly discovered variants.

    Args:
        df: DataFrame with 'DateLastEvaluated' or 'LastEvaluated' column
        split_date: Date to split on (YYYY-MM-DD format)

    Returns:
        train_df, test_df
    """
    print("\n" + "="*70)
    print("TEMPORAL SPLIT")
    print("="*70)
    print(f"Split date: {split_date}")
    print("Train: Variants before split date")
    print("Test: Variants after split date")
    print("Prevents: Memorization of known variants")

    # Find date column
    date_col = None
    for col in ['DateLastEvaluated', 'LastEvaluated', 'SubmissionDate', 'ReviewDate']:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        raise ValueError("No date column found! Need DateLastEvaluated, LastEvaluated, or SubmissionDate")

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    split_date_dt = pd.to_datetime(split_date)

    # Split
    train = df[df[date_col] < split_date_dt].copy()
    test = df[df[date_col] >= split_date_dt].copy()

    # Remove rows with NaT dates
    train = train[train[date_col].notna()]
    test = test[test[date_col].notna()]

    # Verify no overlap
    if 'AlleleID' in df.columns:
        train_ids = set(train['AlleleID'])
        test_ids = set(test['AlleleID'])
        overlap = train_ids & test_ids
        assert len(overlap) == 0, f"❌ Overlap detected: {len(overlap)} variants in both train and test!"

    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train)} variants (before {split_date})")
    print(f"   Test: {len(test)} variants (after {split_date})")
    print(f"   No overlap verified")

    return train, test


def position_based_split(df: pd.DataFrame, gene: str = 'BRCA1', split_fraction: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by genomic position - tests spatial generalization.

    Train on first half of gene, test on second half.
    This prevents model from memorizing position-specific patterns.

    Args:
        df: DataFrame with 'Start' and 'GeneSymbol' columns
        gene: Gene to split (default: BRCA1)
        split_fraction: Fraction for training (default: 0.5)

    Returns:
        train_df, test_df
    """
    print("\n" + "="*70)
    print("POSITION-BASED SPLIT")
    print("="*70)
    print(f"Gene: {gene}")
    print(f"Train: First {split_fraction*100:.0f}% of gene")
    print(f"Test: Last {(1-split_fraction)*100:.0f}% of gene")
    print("Prevents: Position-specific memorization")

    if 'Start' not in df.columns:
        raise ValueError("No 'Start' column found!")

    # Filter to specific gene
    gene_col = 'GeneSymbol' if 'GeneSymbol' in df.columns else 'Gene'
    if gene_col not in df.columns:
        raise ValueError(f"No gene column found! Need GeneSymbol or Gene")

    gene_df = df[df[gene_col] == gene].copy()

    if len(gene_df) == 0:
        raise ValueError(f"No variants found for gene: {gene}")

    # Sort by position
    gene_df = gene_df.sort_values('Start')

    # Find split position
    split_idx = int(len(gene_df) * split_fraction)
    split_position = gene_df.iloc[split_idx]['Start']

    # Split
    train = gene_df.iloc[:split_idx].copy()
    test = gene_df.iloc[split_idx:].copy()

    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train)} variants (positions up to {split_position})")
    print(f"   Test: {len(test)} variants (positions after {split_position})")
    print(f"   No positional overlap")

    return train, test


def variant_type_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by variant type - tests type generalization.

    Train on common types (missense, synonymous), test on rare types (frameshift, nonsense).
    This tests if model learns general variant effects, not type-specific patterns.

    Args:
        df: DataFrame with 'Type' or 'VariantType' column

    Returns:
        train_df, test_df
    """
    print("\n" + "="*70)
    print("VARIANT TYPE SPLIT")
    print("="*70)
    print("Train: Missense, synonymous variants")
    print("Test: Frameshift, nonsense, splice-site variants")
    print("Prevents: Type-specific pattern memorization")

    # Find type column
    type_col = None
    for col in ['Type', 'VariantType', 'MolecularConsequence']:
        if col in df.columns:
            type_col = col
            break

    if not type_col:
        raise ValueError("No variant type column found!")

    # Define train and test types
    train_types = ['missense', 'synonymous', 'substitution']
    test_types = ['frameshift', 'nonsense', 'splice', 'deletion', 'insertion']

    # Split (case-insensitive matching)
    df[type_col] = df[type_col].fillna('unknown').str.lower()

    train = df[df[type_col].str.contains('|'.join(train_types), na=False)].copy()
    test = df[df[type_col].str.contains('|'.join(test_types), na=False)].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Not enough variants of each type for splitting!")

    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train)} variants ({', '.join(train_types)})")
    print(f"   Test: {len(test)} variants ({', '.join(test_types)})")
    print(f"   Testing generalization across variant types")

    return train, test


def leave_one_gene_out(df: pd.DataFrame, test_gene: str = 'BRCA2') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-gene-out split - tests cross-gene generalization.

    Train on one gene (e.g., BRCA1), test on another (e.g., BRCA2).
    This is the strictest test - can model generalize to unseen genes?

    Args:
        df: DataFrame with 'GeneSymbol' or 'Gene' column
        test_gene: Gene to hold out for testing (default: BRCA2)

    Returns:
        train_df, test_df
    """
    print("\n" + "="*70)
    print("LEAVE-ONE-GENE-OUT SPLIT")
    print("="*70)
    print(f"Train: All genes EXCEPT {test_gene}")
    print(f"Test: {test_gene} only")
    print("Prevents: Gene-specific pattern memorization")

    # Find gene column
    gene_col = 'GeneSymbol' if 'GeneSymbol' in df.columns else 'Gene'
    if gene_col not in df.columns:
        raise ValueError("No gene column found! Need GeneSymbol or Gene")

    # Split
    train = df[df[gene_col] != test_gene].copy()
    test = df[df[gene_col] == test_gene].copy()

    if len(test) == 0:
        raise ValueError(f"No variants found for test gene: {test_gene}")

    train_genes = train[gene_col].unique()

    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train)} variants from {len(train_genes)} genes")
    print(f"   Test: {len(test)} variants from {test_gene}")
    print(f"   Complete gene separation - strictest test!")

    return train, test


def check_for_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Verify no data leakage between train and test.

    Checks:
    1. No overlapping AlleleIDs
    2. No exact duplicate sequences
    3. Label distribution similarity
    """
    print("\n" + "="*70)
    print("DATA LEAKAGE CHECK")
    print("="*70)

    # Check 1: AlleleID overlap
    if 'AlleleID' in train_df.columns and 'AlleleID' in test_df.columns:
        train_ids = set(train_df['AlleleID'])
        test_ids = set(test_df['AlleleID'])
        overlap = train_ids & test_ids

        if len(overlap) > 0:
            print(f"❌ ALLELE OVERLAP: {len(overlap)} variants in both train and test!")
            print(f"   This is DATA LEAKAGE!")
            return False
        else:
            print(f"✅ No AlleleID overlap")

    # Check 2: Sequence overlap
    if 'RNA_Sequence' in train_df.columns and 'RNA_Sequence' in test_df.columns:
        train_seqs = set(train_df['RNA_Sequence'].fillna(''))
        test_seqs = set(test_df['RNA_Sequence'].fillna(''))
        seq_overlap = train_seqs & test_seqs

        # Remove empty strings
        seq_overlap = {s for s in seq_overlap if s}

        if len(seq_overlap) > 0:
            pct = (len(seq_overlap) / len(test_seqs)) * 100
            print(f"⚠️ SEQUENCE OVERLAP: {len(seq_overlap)} sequences in both ({pct:.1f}% of test)")
            print(f"   Model may memorize these sequences")
        else:
            print(f"✅ No sequence overlap")

    # Check 3: Label distribution
    if 'Label' in train_df.columns and 'Label' in test_df.columns:
        train_dist = train_df['Label'].value_counts(normalize=True)
        test_dist = test_df['Label'].value_counts(normalize=True)

        train_pos_pct = train_dist.get(1, 0) * 100
        test_pos_pct = test_dist.get(1, 0) * 100

        diff = abs(train_pos_pct - test_pos_pct)

        print(f"\n📊 Label Distribution:")
        print(f"   Train: {train_pos_pct:.1f}% pathogenic")
        print(f"   Test: {test_pos_pct:.1f}% pathogenic")
        print(f"   Difference: {diff:.1f} percentage points")

        if diff > 20:
            print(f"⚠️ Large distribution mismatch - check split strategy")
        else:
            print(f"✅ Similar distributions")

    print("\n✅ Data leakage check complete")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Proper train/test splitting (NO random split!)'
    )
    parser.add_argument('--input', required=True, help='Input CSV with all variants')
    parser.add_argument('--train_out', required=True, help='Output training CSV')
    parser.add_argument('--test_out', required=True, help='Output test CSV')
    parser.add_argument('--method', required=True,
                       choices=['temporal', 'position', 'variant_type', 'leave_gene_out'],
                       help='Splitting method')
    parser.add_argument('--split_date', default='2020-01-01',
                       help='For temporal split: date to split on (YYYY-MM-DD)')
    parser.add_argument('--gene', default='BRCA1',
                       help='For position or leave-gene-out splits: gene name')
    parser.add_argument('--test_gene', default='BRCA2',
                       help='For leave-gene-out: gene to hold out for testing')
    parser.add_argument('--split_fraction', type=float, default=0.5,
                       help='For position split: fraction for training')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"   Loaded {len(df)} variants")

    # Split based on method
    if args.method == 'temporal':
        train, test = temporal_split(df, args.split_date)
    elif args.method == 'position':
        train, test = position_based_split(df, args.gene, args.split_fraction)
    elif args.method == 'variant_type':
        train, test = variant_type_split(df)
    elif args.method == 'leave_gene_out':
        train, test = leave_one_gene_out(df, args.test_gene)

    # Check for leakage
    check_for_data_leakage(train, test)

    # Save
    train.to_csv(args.train_out, index=False)
    test.to_csv(args.test_out, index=False)

    print(f"\n💾 Saved:")
    print(f"   Train: {args.train_out}")
    print(f"   Test: {args.test_out}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Train model on training set")
    print("2. Evaluate on test set (expect 70-85%, NOT 100%)")
    print("3. If performance is poor, model may not generalize")
    print("4. Compare to baseline models (k-mer counting)")
    print("="*70)


if __name__ == '__main__':
    main()
