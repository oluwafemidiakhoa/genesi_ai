#!/usr/bin/env python3
"""
Generate sample human ncRNA sequences for training.

This script creates a small dataset of realistic human ncRNA sequences
in the format expected by the Genesis RNA training pipeline.

The sequences are based on common human ncRNA families:
- microRNAs (miRNAs)
- long non-coding RNAs (lncRNAs)
- small nuclear RNAs (snRNAs)
- ribosomal RNAs (rRNAs)
- transfer RNAs (tRNAs)

Usage:
    python generate_sample_ncrna.py --output ../data/human_ncrna --num_samples 5000
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import List, Dict


# Common human ncRNA motifs and patterns
MIRNA_SEEDS = [
    "UGGAAUGUU",  # let-7 family
    "UAAAGUGCU",  # miR-1/206 family
    "UGUGCAAU",   # miR-21 family
    "UCACAGUGGU", # miR-15/16 family
    "CAGUGCAA",   # miR-29 family
]

# Common RNA structural motifs
GC_RICH_MOTIFS = ["GCGC", "CGCG", "GGCC", "CCGG"]
AU_RICH_MOTIFS = ["AUAU", "UAUA", "AAUU", "UUAA"]


def generate_mirna(seq_id: int) -> tuple[str, Dict]:
    """Generate a realistic microRNA sequence (18-25 nt)"""
    length = random.randint(18, 25)

    # Start with a seed sequence
    seed = random.choice(MIRNA_SEEDS)

    # Extend with random nucleotides
    nucleotides = ['A', 'C', 'G', 'U']
    remaining = length - len(seed)

    if remaining > 0:
        extension = ''.join(random.choices(nucleotides, k=remaining))
        sequence = seed + extension
    else:
        sequence = seed[:length]

    metadata = {
        'id': f'hsa-miR-sample-{seq_id}',
        'description': 'Human microRNA (sample)',
        'length': len(sequence),
        'type': 'miRNA',
        'organism': 'Homo sapiens',
    }

    return sequence, metadata


def generate_lncrna(seq_id: int) -> tuple[str, Dict]:
    """Generate a realistic long non-coding RNA sequence (200-2000 nt)"""
    length = random.randint(200, 2000)

    # Build sequence with some structure
    sequence = []
    nucleotides = ['A', 'C', 'G', 'U']

    # Add some structured regions
    num_motifs = random.randint(5, 15)
    remaining = length

    for _ in range(num_motifs):
        if remaining <= 0:
            break

        # Add a motif
        motif_type = random.choice(['GC', 'AU', 'random'])

        if motif_type == 'GC' and remaining >= 4:
            motif = random.choice(GC_RICH_MOTIFS)
        elif motif_type == 'AU' and remaining >= 4:
            motif = random.choice(AU_RICH_MOTIFS)
        else:
            motif_len = min(random.randint(10, 30), remaining)
            motif = ''.join(random.choices(nucleotides, k=motif_len))

        sequence.append(motif)
        remaining -= len(motif)

    # Fill remaining with random nucleotides
    if remaining > 0:
        sequence.append(''.join(random.choices(nucleotides, k=remaining)))

    final_sequence = ''.join(sequence)[:length]

    metadata = {
        'id': f'NONHSAT{seq_id:06d}',
        'description': 'Human long non-coding RNA (sample)',
        'length': len(final_sequence),
        'type': 'lncRNA',
        'organism': 'Homo sapiens',
    }

    return final_sequence, metadata


def generate_snrna(seq_id: int) -> tuple[str, Dict]:
    """Generate a realistic small nuclear RNA sequence (100-350 nt)"""
    length = random.randint(100, 350)

    # snRNAs have conserved regions
    nucleotides = ['A', 'C', 'G', 'U']

    # Start with Sm-binding site (common in snRNAs)
    sm_site = "AAUUUGUGG"

    # Add GC-rich regions (common in snRNAs)
    sequence = [sm_site]
    remaining = length - len(sm_site)

    while remaining > 0:
        # Alternate between structured and unstructured regions
        if random.random() < 0.4 and remaining >= 4:
            motif = random.choice(GC_RICH_MOTIFS)
        else:
            motif_len = min(random.randint(5, 20), remaining)
            motif = ''.join(random.choices(nucleotides, k=motif_len))

        sequence.append(motif)
        remaining -= len(motif)

    final_sequence = ''.join(sequence)[:length]

    metadata = {
        'id': f'RNU{random.randint(1,12)}-sample-{seq_id}',
        'description': 'Human small nuclear RNA (sample)',
        'length': len(final_sequence),
        'type': 'snRNA',
        'organism': 'Homo sapiens',
    }

    return final_sequence, metadata


def generate_trna(seq_id: int) -> tuple[str, Dict]:
    """Generate a realistic transfer RNA sequence (70-90 nt)"""
    length = random.randint(70, 90)

    # tRNAs have conserved structure (cloverleaf)
    # Simplified version with key motifs
    nucleotides = ['A', 'C', 'G', 'U']

    # D-loop region
    d_loop = "".join(random.choices(['G', 'U'], k=random.randint(7, 10)))

    # Anticodon loop (7 nucleotides)
    anticodon = "".join(random.choices(nucleotides, k=7))

    # T-loop (conserved UUCG motif)
    t_loop = "UUCG" + "".join(random.choices(['A', 'G'], k=random.randint(3, 5)))

    # Acceptor stem
    acceptor = "".join(random.choices(['G', 'C'], k=random.randint(7, 9)))

    # Combine regions
    sequence_parts = [acceptor, d_loop, anticodon, t_loop]
    combined = ''.join(sequence_parts)

    # Pad or trim to desired length
    if len(combined) < length:
        combined += ''.join(random.choices(nucleotides, k=length - len(combined)))
    else:
        combined = combined[:length]

    # Determine amino acid from anticodon (simplified)
    amino_acids = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                   'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
    aa = random.choice(amino_acids)

    metadata = {
        'id': f'tRNA-{aa}-sample-{seq_id}',
        'description': f'Human transfer RNA {aa} (sample)',
        'length': len(combined),
        'type': 'tRNA',
        'organism': 'Homo sapiens',
    }

    return combined, metadata


def generate_rrna(seq_id: int) -> tuple[str, Dict]:
    """Generate a realistic ribosomal RNA fragment (100-500 nt)"""
    # Full rRNAs are very long (1000s of nt), so generate fragments
    length = random.randint(100, 500)

    nucleotides = ['A', 'C', 'G', 'U']

    # rRNAs are GC-rich
    gc_bias = 0.6
    weights = [
        (1 - gc_bias) / 2,  # A
        gc_bias / 2,         # C
        gc_bias / 2,         # G
        (1 - gc_bias) / 2,  # U
    ]

    sequence = ''.join(random.choices(nucleotides, weights=weights, k=length))

    # Add some conserved motifs
    rrna_motifs = ["GGGAAACUGAAC", "CCGCGC", "GUAACAA"]
    for motif in rrna_motifs:
        if len(sequence) > 50:
            insert_pos = random.randint(0, len(sequence) - len(motif))
            sequence = sequence[:insert_pos] + motif + sequence[insert_pos + len(motif):]

    # Trim to exact length
    sequence = sequence[:length]

    rna_types = ['5S', '5.8S', '18S', '28S']
    rna_type = random.choice(rna_types)

    metadata = {
        'id': f'RNA{rna_type}-sample-{seq_id}',
        'description': f'Human {rna_type} ribosomal RNA fragment (sample)',
        'length': len(sequence),
        'type': 'rRNA',
        'organism': 'Homo sapiens',
    }

    return sequence, metadata


def generate_dataset(num_samples: int) -> tuple[List[str], List[Dict]]:
    """
    Generate a diverse dataset of human ncRNA sequences.

    Args:
        num_samples: Total number of sequences to generate

    Returns:
        Tuple of (sequences, metadata) lists
    """
    sequences = []
    metadata_list = []

    # Distribution of ncRNA types (approximate biological frequencies)
    type_distribution = {
        'miRNA': 0.25,
        'lncRNA': 0.40,
        'snRNA': 0.10,
        'tRNA': 0.15,
        'rRNA': 0.10,
    }

    generators = {
        'miRNA': generate_mirna,
        'lncRNA': generate_lncrna,
        'snRNA': generate_snrna,
        'tRNA': generate_trna,
        'rRNA': generate_rrna,
    }

    # Calculate number of each type
    type_counts = {
        rna_type: int(num_samples * fraction)
        for rna_type, fraction in type_distribution.items()
    }

    # Adjust for rounding
    total = sum(type_counts.values())
    if total < num_samples:
        type_counts['lncRNA'] += num_samples - total

    print(f"Generating {num_samples} sample sequences:")
    for rna_type, count in type_counts.items():
        print(f"  {rna_type}: {count}")

    seq_id = 1
    for rna_type, count in type_counts.items():
        generator = generators[rna_type]
        for _ in range(count):
            seq, meta = generator(seq_id)
            sequences.append(seq)
            metadata_list.append(meta)
            seq_id += 1

    # Shuffle to mix types
    combined = list(zip(sequences, metadata_list))
    random.shuffle(combined)
    sequences, metadata_list = zip(*combined)

    return list(sequences), list(metadata_list)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample human ncRNA sequences for training"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/human_ncrna',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5000,
        help='Number of sequences to generate (default: 5000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("="*70)
    print("GENESIS RNA - Sample ncRNA Data Generator")
    print("="*70)
    print(f"\nOutput directory: {args.output}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Random seed: {args.seed}\n")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    sequences, metadata = generate_dataset(args.num_samples)

    # Save as pickle
    output_file = output_path / 'sequences.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'sequences': sequences,
            'metadata': metadata,
        }, f)

    print(f"\n✅ Successfully generated {len(sequences)} sequences")
    print(f"📁 Saved to: {output_file}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total sequences: {len(sequences):,}")
    print(f"  Average length: {sum(len(s) for s in sequences) / len(sequences):.1f} nt")
    print(f"  Min length: {min(len(s) for s in sequences)} nt")
    print(f"  Max length: {max(len(s) for s in sequences)} nt")

    # Type breakdown
    type_counts = {}
    for meta in metadata:
        rna_type = meta['type']
        type_counts[rna_type] = type_counts.get(rna_type, 0) + 1

    print("\n  By type:")
    for rna_type, count in sorted(type_counts.items()):
        print(f"    {rna_type}: {count:,} ({count/len(sequences)*100:.1f}%)")

    print("\n" + "="*70)
    print("✅ Data generation complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run training: python -m genesis_rna.train_pretrain --data_path ../data/human_ncrna")
    print("  2. Or use the Colab script: python examples/colab_train_ncrna.py")


if __name__ == '__main__':
    main()
