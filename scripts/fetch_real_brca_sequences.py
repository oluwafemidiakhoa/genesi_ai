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
import sys
from typing import Optional, Dict

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from hgvs_parser import HGVSParser, apply_hgvs_variant


# Ensembl REST API
ENSEMBL_API = "https://rest.ensembl.org"

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


def apply_variant_to_sequence(reference_dna: str, hgvs_notation: str) -> Optional[Dict[str, str]]:
    """
    Apply variant using HGVS parser.

    Args:
        reference_dna: Reference DNA sequence
        hgvs_notation: HGVS notation (e.g., "c.5266dupC")

    Returns:
        Dict with 'wildtype' and 'mutant' RNA sequences, or None if parsing fails
    """
    try:
        result = apply_hgvs_variant(reference_dna, hgvs_notation, convert_to_rna=True)
        return result
    except ValueError as e:
        # Variant couldn't be parsed or applied
        return None
    except Exception as e:
        # Other errors
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

    # Get reference DNA sequences (before RNA conversion)
    reference_dna = {
        'BRCA1': brca1_dna,
        'BRCA2': brca2_dna
    }

    # Generate sequences for each variant
    print(f"\n🧬 Generating sequences for {len(clinvar_df)} variants...")
    print("   Applying HGVS variants to reference sequences...")

    results = []
    skipped = 0
    variants_applied = 0
    variants_failed = 0

    for idx, row in clinvar_df.iterrows():
        gene = row.get('GeneSymbol', 'Unknown')
        hgvs_notation = row.get('Name', '')

        # Get reference DNA sequence
        ref_dna = reference_dna.get(gene)
        if not ref_dna:
            skipped += 1
            continue

        # Apply variant using HGVS parser
        variant_result = apply_variant_to_sequence(ref_dna, hgvs_notation)

        if variant_result:
            # Successfully applied variant
            wildtype_rna = variant_result['wildtype']
            mutant_rna = variant_result['mutant']
            variants_applied += 1

            results.append({
                'AlleleID': row.get('AlleleID', ''),
                'GeneSymbol': gene,
                'Name': hgvs_notation,
                'ClinicalSignificance': row.get('ClinicalSignificance', ''),
                'Label': row.get('Label', None),
                'RNA_Sequence': mutant_rna,
                'Wildtype_Sequence': wildtype_rna,
                'SequenceType': 'variant',
                'VariantType': variant_result.get('variant_type', 'unknown')
            })
        else:
            # Failed to apply variant - use reference sequence as fallback
            variants_failed += 1
            ref_rna = reference_sequences.get(gene)

            results.append({
                'AlleleID': row.get('AlleleID', ''),
                'GeneSymbol': gene,
                'Name': hgvs_notation,
                'ClinicalSignificance': row.get('ClinicalSignificance', ''),
                'Label': row.get('Label', None),
                'RNA_Sequence': ref_rna,
                'Wildtype_Sequence': ref_rna,
                'SequenceType': 'reference_fallback',
                'VariantType': 'parse_failed'
            })

        if (idx + 1) % 1000 == 0:
            print(f"   Progress: {idx + 1}/{len(clinvar_df)} variants")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print(f"\n✅ Generated {len(results)} sequences")
    print(f"   - Successfully applied: {variants_applied} variants")
    print(f"   - Failed to parse: {variants_failed} (using reference as fallback)")
    print(f"   - Skipped: {skipped} (unsupported gene)")
    print(f"💾 Saved to: {output_csv}")

    # Statistics
    success_rate = (variants_applied / len(results)) * 100 if results else 0

    print("\n" + "="*70)
    print("SEQUENCE GENERATION SUMMARY")
    print("="*70)
    print(f"✅ Variant Application Success Rate: {success_rate:.1f}%")
    print(f"✅ NO LABEL LEAKAGE: Labels were NOT used during sequence generation")
    print(f"\n📊 Sequence Types:")
    print(f"   - Real variants with mutations: {variants_applied}")
    print(f"   - Reference fallback: {variants_failed}")
    print(f"\n🧬 HGVS Parser Features:")
    print(f"   - Substitutions: c.123A>T")
    print(f"   - Deletions: c.123del, c.123_125del")
    print(f"   - Insertions: c.123_124insAT")
    print(f"   - Duplications: c.5266dupC")
    print(f"   - Indels: c.123delinsAT")
    print(f"\n📖 This replaces the synthetic generation that had the 'AAAA'")
    print(f"   marker causing 100% accuracy label leakage bug.")
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
