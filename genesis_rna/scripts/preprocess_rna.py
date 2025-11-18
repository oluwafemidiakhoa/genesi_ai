#!/usr/bin/env python3
"""
Preprocess RNA sequences from FASTA format for Genesis RNA training.

This script:
1. Reads RNA sequences from FASTA files
2. Filters by length and quality
3. Converts T to U (DNA → RNA)
4. Saves in training-ready format (pickle or JSON)

Usage:
    python preprocess_rna.py --input data/rnacentral.fasta --output data/processed
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional
import sys

try:
    from Bio import SeqIO
except ImportError:
    print("ERROR: BioPython not installed. Install with: pip install biopython")
    sys.exit(1)

from tqdm import tqdm


def is_valid_rna_sequence(seq: str, min_len: int = 20, max_len: int = 2048) -> bool:
    """
    Check if sequence is valid for training.

    Args:
        seq: RNA sequence string
        min_len: Minimum sequence length
        max_len: Maximum sequence length

    Returns:
        True if valid, False otherwise
    """
    # Check length
    if len(seq) < min_len or len(seq) > max_len:
        return False

    # Check for valid nucleotides (allow ACGUN)
    valid_chars = set('ACGUN')
    seq_upper = seq.upper()

    # Allow some ambiguous bases, but not too many
    unknown_count = sum(1 for c in seq_upper if c not in valid_chars)
    if unknown_count > len(seq) * 0.1:  # Max 10% unknown
        return False

    return True


def convert_to_rna(seq: str) -> str:
    """Convert DNA sequence to RNA (T → U)"""
    return seq.upper().replace('T', 'U')


def preprocess_fasta(
    input_path: str,
    output_path: str,
    min_len: int = 20,
    max_len: int = 2048,
    max_sequences: Optional[int] = None,
    output_format: str = 'pickle',
):
    """
    Preprocess FASTA file into training format.

    Args:
        input_path: Path to input FASTA file
        output_path: Path to output directory
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        max_sequences: Maximum number of sequences to process (None = all)
        output_format: Output format ('pickle', 'json', or 'text')
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_path}...")
    print(f"Length range: {min_len} - {max_len}")
    print(f"Output format: {output_format}")

    sequences = []
    metadata = []

    # Count total sequences
    print("Counting sequences...")
    total_seqs = sum(1 for _ in SeqIO.parse(input_path, "fasta"))
    print(f"Total sequences in file: {total_seqs:,}")

    # Process sequences
    valid_count = 0
    filtered_count = 0

    with tqdm(total=min(total_seqs, max_sequences or total_seqs)) as pbar:
        for i, record in enumerate(SeqIO.parse(input_path, "fasta")):
            if max_sequences and valid_count >= max_sequences:
                break

            seq_str = str(record.seq)

            # Convert DNA to RNA
            seq_rna = convert_to_rna(seq_str)

            # Validate
            if not is_valid_rna_sequence(seq_rna, min_len, max_len):
                filtered_count += 1
                pbar.update(1)
                continue

            # Store sequence and metadata
            sequences.append(seq_rna)
            metadata.append({
                'id': record.id,
                'description': record.description,
                'length': len(seq_rna),
            })

            valid_count += 1
            pbar.update(1)

    print(f"\nProcessing complete!")
    print(f"Valid sequences: {valid_count:,}")
    print(f"Filtered sequences: {filtered_count:,}")

    # Save processed data
    if output_format == 'pickle':
        # Save as pickle (fast loading)
        output_file = output_path / 'sequences.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump({
                'sequences': sequences,
                'metadata': metadata,
            }, f)
        print(f"Saved to {output_file}")

    elif output_format == 'json':
        # Save as JSON (human-readable)
        output_file = output_path / 'sequences.json'
        with open(output_file, 'w') as f:
            json.dump({
                'sequences': sequences,
                'metadata': metadata,
            }, f, indent=2)
        print(f"Saved to {output_file}")

    elif output_format == 'text':
        # Save as plain text (one sequence per line)
        output_file = output_path / 'sequences.txt'
        with open(output_file, 'w') as f:
            for seq in sequences:
                f.write(seq + '\n')
        print(f"Saved to {output_file}")

        # Save metadata separately
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")

    # Save statistics
    stats = {
        'total_input_sequences': total_seqs,
        'valid_sequences': valid_count,
        'filtered_sequences': filtered_count,
        'min_length': min_len,
        'max_length': max_len,
        'avg_sequence_length': sum(m['length'] for m in metadata) / len(metadata) if metadata else 0,
    }

    stats_file = output_path / 'stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RNA sequences for Genesis RNA training"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input FASTA file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path'
    )
    parser.add_argument(
        '--min_len',
        type=int,
        default=20,
        help='Minimum sequence length (default: 20)'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )
    parser.add_argument(
        '--max_sequences',
        type=int,
        default=None,
        help='Maximum number of sequences to process (default: all)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='pickle',
        choices=['pickle', 'json', 'text'],
        help='Output format (default: pickle)'
    )

    args = parser.parse_args()

    preprocess_fasta(
        input_path=args.input,
        output_path=args.output,
        min_len=args.min_len,
        max_len=args.max_len,
        max_sequences=args.max_sequences,
        output_format=args.format,
    )


if __name__ == '__main__':
    main()
