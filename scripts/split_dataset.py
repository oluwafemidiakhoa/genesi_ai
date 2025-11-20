#!/usr/bin/env python3
"""
Split dataset into train/test sets for evaluation

Usage:
    python split_dataset.py \
        --input data/breast_cancer/brca_mutations/train.jsonl \
        --train_out data/breast_cancer/train.jsonl \
        --test_out data/breast_cancer/test.jsonl \
        --test_split 0.2
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def load_jsonl(path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def stratified_split(
    data: List[Dict],
    test_split: float = 0.2,
    label_key: str = 'label',
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into train/test sets with stratification

    Ensures balanced distribution of classes in both sets.

    Args:
        data: List of samples
        test_split: Fraction for test set (default: 0.2 = 20%)
        label_key: Key for label in sample dict
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    random.seed(random_seed)

    # Group by label
    label_groups = {}
    for sample in data:
        label = sample.get(label_key, 0)
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(sample)

    # Split each group
    train_data = []
    test_data = []

    for label, samples in label_groups.items():
        random.shuffle(samples)

        split_idx = int(len(samples) * (1 - test_split))

        train_data.extend(samples[:split_idx])
        test_data.extend(samples[split_idx:])

    # Shuffle final sets
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/test sets'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file'
    )
    parser.add_argument(
        '--train_out',
        type=str,
        required=True,
        help='Output path for training set'
    )
    parser.add_argument(
        '--test_out',
        type=str,
        required=True,
        help='Output path for test set'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.2,
        help='Fraction of data for test set (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--label_key',
        type=str,
        default='label',
        help='Key for label in data (default: "label")'
    )

    args = parser.parse_args()

    print("="*70)
    print("DATASET SPLITTER")
    print("="*70)
    print(f"\nInput: {args.input}")
    print(f"Train output: {args.train_out}")
    print(f"Test output: {args.test_out}")
    print(f"Test split: {args.test_split * 100:.0f}%")
    print(f"Random seed: {args.seed}")

    # Load data
    print(f"\n📥 Loading data...")
    data = load_jsonl(args.input)
    print(f"   Loaded {len(data)} samples")

    # Check label distribution
    labels = [s.get(args.label_key, 0) for s in data]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n📊 Label distribution:")
    for label, count in label_counts.items():
        label_name = "Pathogenic" if label == 1 else "Benign/VUS"
        print(f"   {label_name} ({label}): {count} ({count/len(data)*100:.1f}%)")

    # Split data
    print(f"\n✂️  Splitting data (stratified)...")
    train_data, test_data = stratified_split(
        data,
        test_split=args.test_split,
        label_key=args.label_key,
        random_seed=args.seed
    )

    print(f"   Train set: {len(train_data)} samples")
    print(f"   Test set: {len(test_data)} samples")

    # Verify stratification
    train_labels = [s.get(args.label_key, 0) for s in train_data]
    test_labels = [s.get(args.label_key, 0) for s in test_data]

    train_pos_rate = sum(train_labels) / len(train_labels) if train_labels else 0
    test_pos_rate = sum(test_labels) / len(test_labels) if test_labels else 0

    print(f"\n✅ Stratification check:")
    print(f"   Train positive rate: {train_pos_rate*100:.1f}%")
    print(f"   Test positive rate: {test_pos_rate*100:.1f}%")
    print(f"   Difference: {abs(train_pos_rate - test_pos_rate)*100:.1f}%")

    # Save data
    print(f"\n💾 Saving splits...")
    save_jsonl(train_data, args.train_out)
    save_jsonl(test_data, args.test_out)

    print(f"\n✅ Split complete!")
    print(f"   Train: {args.train_out}")
    print(f"   Test: {args.test_out}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
