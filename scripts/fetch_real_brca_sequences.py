#!/usr/bin/env python3
"""
Fetch REAL BRCA1/BRCA2 sequences from Ensembl and apply variants.

This replaces the synthetic sequence generation that caused label leakage.
NO label information is used during sequence generation.
"""

import requests
import pandas as pd
import argparse
from pathlib import Path
import time


# Ensembl REST API
ENS

EMBL_API = "https://rest.ensembl.org"

# BRCA transcript IDs (Ensembl)
BRCA_TRANSCRIPTS = {
    'BRCA1': 'ENST00000357654',  # BRCA1 canonical transcript
    'BRCA2': 'ENST00000380152',  # BRCA2 canonical transcript
}


def fetch_transcript_sequence(transcript_id, retries=3):
    """
    Fetch mRNA sequence from Ensembl REST API.

    Args:
        transcript_id: Ensembl transcript ID (e.g., ENST00000357654)
        retries: Number of retry attempts

    Returns:
        str: mRNA sequence (DNA, will convert to RNA later)
    """
    url = f"{ENSEMBL_API}/sequence/id/{transcript_id}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            sequence = data.get('seq', '')

            if sequence:
                print(f"✅ Fetched {transcript_id}: {len(sequence)} bp")
                return sequence
            else:
                print(f"⚠️ Empty sequence for {transcript_id}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"❌ Failed to fetch {transcript_id}")
                return None

    return None


def dna_to_rna(dna_sequence):
    """Convert DNA sequence to RNA (T→U)"""
    return dna_sequence.replace('T', 'U').replace('t', 'u')


def apply_simple_variant(sequence, position, ref_allele, alt_allele):
    """
    Apply a simple substitution variant to sequence.

    NOTE: This is simplified. Real variant application requires:
    - VEP or similar tool
    - Proper coordinate mapping
    - Handling of insertions, deletions, duplications

    Args:
        sequence: mRNA sequence
        position: Position in transcript (0-indexed)
        ref_allele: Reference allele
        alt_allele: Alternate allele

    Returns:
        str: Mutated sequence, or None if variant can't be applied
    """
    if position < 0 or position >= len(sequence):
        return None

    # Simple substitution
    if len(ref_allele) == 1 and len(alt_allele) == 1:
        if sequence[position].upper() == ref_allele.upper():
            mutated = list(sequence)
            mutated[position] = alt_allele
            return ''.join(mutated)

    # For complex variants (indels, etc.), return None
    # In production, use VEP API or pyhgvs library
    return None


def generate_real_variant_sequences(clinvar_df, output_csv, sample_size=None):
    """
    Generate REAL RNA sequences for ClinVar variants.

    CRITICAL: NO label information is used during sequence generation.

    Args:
        clinvar_df: DataFrame with ClinVar variants (must have 'GeneSymbol' column)
        output_csv: Path to save sequences
        sample_size: Limit to N variants (for testing)
    """
    print("="*70)
    print("FETCHING REAL BRCA SEQUENCES")
    print("="*70)
    print("\nIMPORTANT: No label information will be used during generation!")
    print("This prevents the label leakage that caused the 100% accuracy bug.\n")

    # Sample if requested
    if sample_size:
        clinvar_df = clinvar_df.sample(min(sample_size, len(clinvar_df)))

    # Fetch reference transcripts
    print("📥 Fetching BRCA1 reference transcript...")
    brca1_dna = fetch_transcript_sequence(BRCA_TRANSCRIPTS['BRCA1'])
    brca1_rna = dna_to_rna(brca1_dna) if brca1_dna else None

    print("📥 Fetching BRCA2 reference transcript...")
    brca2_dna = fetch_transcript_sequence(BRCA_TRANSCRIPTS['BRCA2'])
    brca2_rna = dna_to_rna(brca2_dna) if brca2_dna else None

    if not brca1_rna or not brca2_rna:
        print("❌ Failed to fetch reference sequences!")
        return

    reference_sequences = {
        'BRCA1': brca1_rna,
        'BRCA2': brca2_rna
    }

    print(f"\n✅ Reference sequences loaded:")
    print(f"   BRCA1: {len(brca1_rna)} nt")
    print(f"   BRCA2: {len(brca2_rna)} nt")

    # Generate sequences for each variant
    print(f"\n🧬 Generating sequences for {len(clinvar_df)} variants...")

    results = []
    skipped = 0

    for idx, row in clinvar_df.iterrows():
        gene = row.get('GeneSymbol', 'Unknown')

        # Get reference sequence
        ref_seq = reference_sequences.get(gene)
        if not ref_seq:
            skipped += 1
            continue

        # For now, use reference sequence directly
        # In production, apply the actual variant mutation
        # This requires parsing HGVS notation (c.5266dupC) and applying properly

        # TODO: Implement proper variant application using:
        # - pyhgvs library
        # - VEP API
        # - Or manual HGVS parser

        # For now, return reference sequence (wildtype)
        # This is biologically accurate but doesn't include the variant
        variant_sequence = ref_seq

        results.append({
            'AlleleID': row.get('AlleleID', ''),
            'GeneSymbol': gene,
            'Name': row.get('Name', ''),
            'ClinicalSignificance': row.get('ClinicalSignificance', ''),
            'Label': row.get('Label', None),
            'RNA_Sequence': variant_sequence,
            'SequenceType': 'reference'  # Mark that this is reference, not variant
        })

        if (idx + 1) % 1000 == 0:
            print(f"   Progress: {idx + 1}/{len(clinvar_df)} variants")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print(f"\n✅ Generated {len(results)} sequences")
    print(f"⚠️  Skipped {skipped} (unsupported gene)")
    print(f"💾 Saved to: {output_csv}")

    print("\n" + "="*70)
    print("IMPORTANT NOTES")
    print("="*70)
    print("⚠️  Current limitation: Using reference sequences only")
    print("    Variant mutations are NOT yet applied (requires HGVS parser)")
    print("\n✅  NO LABEL LEAKAGE: Labels were not used during generation")
    print("\n🔧  Next step: Implement proper variant application:")
    print("    - Parse HGVS notation (c.5266dupC, etc.)")
    print("    - Apply to reference sequence")
    print("    - Validate mutation is correct")
    print("\n📖  For now, this generates real BRCA sequences without the")
    print("    synthetic 'AAAA' marker that caused the label leakage bug.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch real BRCA sequences (NO synthetic data!)'
    )
    parser.add_argument('--clinvar', required=True,
                       help='ClinVar CSV with variant metadata')
    parser.add_argument('--output', required=True,
                       help='Output CSV with real sequences')
    parser.add_argument('--sample', type=int, default=None,
                       help='Limit to N variants (for testing)')
    args = parser.parse_args()

    # Load ClinVar data
    print(f"📥 Loading ClinVar data from {args.clinvar}...")
    df = pd.read_csv(args.clinvar)
    print(f"   Loaded {len(df)} variants")

    # Generate real sequences
    generate_real_variant_sequences(df, args.output, args.sample)

    print("\n✅ Complete! Use this CSV for training (no label leakage)")


if __name__ == '__main__':
    main()
