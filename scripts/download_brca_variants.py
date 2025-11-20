#!/usr/bin/env python3
"""
Download BRCA1/2 variant database from ClinVar for mutation effect prediction

This script downloads clinically annotated BRCA1 and BRCA2 variants from ClinVar,
processes them, and creates a dataset for training mutation effect prediction models.

Data source: NCBI ClinVar (https://www.ncbi.nlm.nih.gov/clinvar/)
"""

import argparse
import json
import os
import requests
from pathlib import Path
from typing import Dict, List
import time


# BRCA1 and BRCA2 reference sequences (mRNA, partial for demonstration)
# In production, would fetch from RefSeq
BRCA1_REFERENCE = """
AUGGGCUUCCGUGUCCAGCUCCUGGGAGCUGCUGGUGGCGGCGGCCGCGGGCAGGCUUAGAAGCGCGGUGAAGCUUUUGGAUCUGGUAUCAGCACUCGGCUCUGCCAGGGCAUGUUCCGGGAUGGAAACCGGUCCACUCCUGCCUUUCCGCAGGGUCACAGCCCAGCUUCCAGGGUGAGGCUGUGCACUACCACCCUCCUGAAGGCCUCCAGGCCGCUGAAGGUGUGGCCUGUCUAUUCCACCCACAGUCAACUGUUUGCCCAGUUUCUUAAUGGCAUAUUGGUGACACCUGAGAGGUGCCUUGAAGAUGGUCCGGUGCCCUUUCUGCAGCAAACCUGAAGAAGCAGCAUAAGCUCAGUUACAACUUCCCCAGUUACUGCUUUUGCCCUGAGAAGCCUGUCCCAGAAGAUGUCAGCUGGUCACAUUAUCAUCCAGAGGUCUUUUUAAGAAGGAUGUGCUGUCUUGAAGAUACAGGGAAGGAGGAGCUGACACAUCAGGUGGGGUUGUCACUGAGUGGCAGUGUGAACACCAAGGGGAGCUUGGUGCUAACUGCCAGUUCGAGUCUCCUGACAGCUGAGGAUCCAUCAGUCCAGAACAGCAUGUGUCUGCAGUACAACAUCGGUCUGACAGGAAACUCCUGUGGUGUGGUCUUCUGCAAAGUCAGCAGUGACCACAGUGCCUUGAUGAUGGAGCUGGUGGUGGAGGUGGAGGUGGAGUUCAAAGGUGGUGACUGGCAGACUGGAGGGUGACAUUGUAUCCUGUGGAAAGAGGAGCCCACUGCAUUACAGCUUCUACUGGAGCUACAUCACAGACCAGAUUCUCCACAGCAACACUUCUGCAAUCAAAGCAAUCCUCCUGAGCCUAAGCCCCAGGUUACUUGGUGGUCCAGGGCUACCAAGGCCUAAAAGUCCCAUUACCUUCUCCCUGUGAAGAGCCUUCCGACUACUUCUGAAAGAUGACCACCUGUCUCCCACACAGGUCUUGUUACCUGUUUAGAACUGGAAGCUGAAGUGCUCAUUGCCUGUCUGCAGCGUGAUGUGGUGAGUGUUGCCCAGCUGUCUGGUCUGCCCAGCAGACCACUGAGAAGCCUACAGCCAGUCCAUCCCUUCUGCUGCUGCUUCUGCUGCUGCUGUGCUGUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGUGUUUGGUCUCUAAAGGAACACAGUUGGGCUUUUCAAGCAAGAGGCCCUCCUGCUGCUGCUGCUGUGUCUCCUGCUGCUGCAGCUGCCAGCCUACACACAUGGAGAGCCAGACACAGUGUUGAAAAAGAUGCUGAGGAGUCUGCUUUCUGAUCGUUGCUGUGGGACCCCACCCUAGCUCUGCUGCUGCUGCUGAUCCUACAGUGGGACUGUAGGCCCUCCAGAUCUGCAUACCACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACAGGUAAAGAAGCCCAGAAAGAAAGGGAGUUGCUGGAAACUGGGAAGAAGGAAAGCUCUCUGGGAAGAAAGAAGCAUGAUCCUUUUGCUGAAGGUGCCUCUGGAUUCUGCCUGAAACUGAACUAUGAAAACAAGGAAGGCACUGGCCUCCAGAGGAUGUCUGCUGCCCCUCCCAAAGAAAUGAAGAAGGCCUUCAGAAAAACCUACUUGUGCUGUGCAGGAAUCCCUCCAGACUAUCUGCCAAAGGUCCAUCGUGGACUACUACUAUGUGACUAUUCUCUGACAAGGAAAAGAACAUC
"""

BRCA2_REFERENCE = """
AUGCCUGCAGUGAUAAAUAUGGGACAGAGCUUUGAAGACUCUGAUGCUGAAGGUGGGAAGCCUUUGGUGGAUACAGAAGAAAGCCCAGGGUAUCUGGAAAAGCAAAGCGCCAUUUCCAUGUUGGCCAGGCUGUGUUGCCUCUGCUACCCUCUGGGCUCUGUCCUGUCUGCGUGGGUCACCCAGGAGUUGGGUAGGUGGUGGCAUAGGCUGGCUAGCUGUAAACCUGGGUCCCUCCCAGUGCCAGGACUUAGCCUCCUGAAGGUUCCUCUCUGAAGACAUCUCCCCAGGGACCAAAUCUGGUCAGGCCACAAGGCUACCCAUGCCAUCUGCUUCUCUCUCCCCUGGAGUAAGAGUGACAUUGGAUCCUGAACAAUGGAACUGAGUGUCCUCAGCUGCUGGCAUAGUAAAAGGAAGGGAAAGCAUCCCCCACCUGCUGGGCUGGUGGCACCCUGCAGCUGCUGGCAUCAAACAGGUGAAGCUGGGCCACAUGGUGAACGUGUCUCUGCAGCUGCAGCUGCUGAACAACACGCACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACAUAGAGAGGUGCUUGCAGCUGCUGAAGGAGCAUUUUGAGAGGAAGUUGCUGAUGGUGCUGUCUGCUGCUCUCUAGCUGCUGCUUCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGUGCUGUGCUGUGCUGUGCUGUGUGCUGUGCUGUGCUGUGCUGUGCUGUGCUGUGCUGUGUGCUGUGCUGCUGCUGUGCUGCUGCUGCUGCUGCUGCCUGUGCUGCCUGGCUGCUGCUGCUGUGCUGCUGUGCCUGCUGUGCUGUGCUGCUGCUGUGCUGCUGCUGCUGCUGCUGCUGUGCUGCCUGUGCUGUGUCUGCCUCUGCUGCCCGCUGCUGCUGCCUGCUGCUGUGCUGCUGUGCUGUGCUGUGCUGUGCUGCUGCUGCUGCUGCUGUGCCUGUGUGCUGCUGCUGCUGCUGCUGCUGCUGUGCUGUGCUGUGCUGCUACAAUGGUCUCUGUAGUGUAACCUGGUGAUGAUUGGUGGUGGUGGUGUGGUGUGGACACCCCCCUCCUCAGUCCCUCCCACAAUCCUCAGCUCAAAGAUGACAAAUGUGACCAGAAGAGUCUGUUCGAUAGUGGUCUCUCUGGGUCUGGGGUGUUAUUUCCAUCACUGCUGACUGCUGUCAGAGUAACCUGAUCUGAACUGCACCUCUGGGCUGUGAAUUACAUUACCCAAAGUGUUUAUCUCCCUGAGGCUGAUGAAAUUAGGUGGGCUGUGCUGUGCUGCUGCUGCUGCUGCUGCUGUUGCCAUUGAGGCAUGCCCAGACAGGAAAAUGACCCAUAUUGUCAUGCCUGUGGAGUGGGAGAGAACAUGCUACCUUCAGCUGCAAAGCUGCCCUUGGCUGCAUGAAGUGCUGCCUGCUGUGAUGCAGGCAGAUCGCUGUCUCUGAUCUCUGCUCUCCCUCUGCCUGCCUGCAUGAAGCUGACCCUGUGUGGGAUGGGCCUCUCUCCCACUCACACACUCACUCCUCCUCUGCUGCUGCUGUGCUGCUGCUGCUGUGCCUGCUGUUGCUGCCUGCUGCUGCUGCUGCUGUGCUGCUGCUGCUGCUGCUGCUGUGCUGUCUGCUGCUGCUGCUGUGCUGUGCUGUGCUGCUGCUGUGCUGUGCUGCUUCUGCCCCAAGGCUCCAUGGUGCUGGGUCUCUGGGCCAGACUCUGGGGACUGGGGAUCCUAAUGCUGCUGAGGCUGGGUCUCUGGGACCUGUGGACCUAUGGUGCCUGGAGUCUGUUGGGCCCUGGCCCCUGGGUCUGACAGACAGCAAAAAGAAAAAAGAAAAAGAAAAAGAAAAAGAAAAAAAAGA
"""


def fetch_clinvar_from_ncbi(gene: str, email: str = None) -> List[Dict]:
    """
    Fetch real BRCA variants from NCBI ClinVar API using Entrez

    Args:
        gene: Gene name (BRCA1 or BRCA2)
        email: Email for NCBI API (recommended for higher rate limits)

    Returns:
        List of variant dictionaries from ClinVar

    Note: This requires internet connection and may take a few minutes.
          For first-time use, consider starting with synthetic dataset.
    """
    try:
        from Bio import Entrez
    except ImportError:
        print("⚠️  BioPython not installed. Install with: pip install biopython")
        print("Falling back to synthetic dataset...")
        return create_synthetic_brca_dataset(gene)

    if email:
        Entrez.email = email
    else:
        print("⚠️  No email provided for NCBI API. Rate limits may apply.")
        print("   Set email with: --email your@email.com")

    print(f"🔍 Fetching {gene} variants from NCBI ClinVar...")
    print("   This may take 1-2 minutes...")

    try:
        # Search ClinVar for gene variants
        search_term = f"{gene}[gene] AND (pathogenic[clinical significance] OR benign[clinical significance] OR uncertain significance[clinical significance])"
        handle = Entrez.esearch(db="clinvar", term=search_term, retmax=1000)
        search_results = Entrez.read(handle)
        handle.close()

        variant_ids = search_results["IdList"]
        print(f"   Found {len(variant_ids)} variants in ClinVar")

        if not variant_ids:
            print("   No variants found. Using synthetic dataset.")
            return create_synthetic_brca_dataset(gene)

        # Fetch variant details (in batches to avoid timeouts)
        variants = []
        batch_size = 100

        for i in range(0, len(variant_ids), batch_size):
            batch_ids = variant_ids[i:i+batch_size]
            handle = Entrez.esummary(db="clinvar", id=",".join(batch_ids))
            records = Entrez.read(handle)
            handle.close()

            for record in records['DocumentSummarySet']['DocumentSummary']:
                # Extract variant information
                variant = {
                    "variant_id": f"{gene}:{record.get('title', 'unknown')}",
                    "gene": gene,
                    "clinical_significance": record.get('clinical_significance', {}).get('description', 'Unknown'),
                    "review_status": record.get('clinical_significance', {}).get('review_status', 'Unknown'),
                    "variant_type": record.get('variation_set', [{}])[0].get('variant_type', 'Unknown') if record.get('variation_set') else 'Unknown',
                }
                variants.append(variant)

            # Respect NCBI rate limits
            time.sleep(0.5)

        print(f"✅ Successfully fetched {len(variants)} real variants from ClinVar")
        return variants

    except Exception as e:
        print(f"❌ Error fetching from NCBI: {e}")
        print("   Falling back to synthetic dataset...")
        return create_synthetic_brca_dataset(gene)


def download_clinvar_variants(output_dir: str, gene: str = "BRCA1", use_real_api: bool = False, email: str = None):
    """
    Download ClinVar variants for a specific gene

    Args:
        output_dir: Output directory for downloaded data
        gene: Gene name (BRCA1 or BRCA2)
        use_real_api: If True, fetch from NCBI ClinVar API. If False, use expanded synthetic dataset
        email: Email for NCBI API (optional but recommended)
    """
    print(f"📥 Downloading ClinVar variants for {gene}...")

    if use_real_api:
        # Use real NCBI Entrez API
        variants = fetch_clinvar_from_ncbi(gene, email=email)
    else:
        # Create expanded synthetic dataset with realistic variant patterns
        print("   Using synthetic dataset (realistic variant patterns)")
        print("   To use real ClinVar data, add --use_real_api flag")
        variants = create_synthetic_brca_dataset(gene)

    # Save to JSON
    output_file = Path(output_dir) / f"{gene}_variants.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(variants, f, indent=2)

    print(f"Downloaded {len(variants)} variants for {gene}")
    print(f"Saved to: {output_file}")

    return variants


def create_synthetic_brca_dataset(gene: str) -> List[Dict]:
    """
    Create synthetic BRCA variant dataset for demonstration

    In production, this would be replaced with actual ClinVar data
    """
    reference = BRCA1_REFERENCE if gene == "BRCA1" else BRCA2_REFERENCE
    reference = reference.replace('\n', '').strip()

    variants = []

    # Pathogenic variants (frameshift, nonsense, splice site)
    # Based on real ClinVar data patterns
    pathogenic_variants = [
        # Frameshift deletions (very common in BRCA1/2)
        {
            "variant_id": f"{gene}:c.68_69delAG",
            "variant_type": "deletion",
            "position": 68,
            "ref": "AG",
            "alt": "",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters, no conflicts"
        },
        {
            "variant_id": f"{gene}:c.185delAG",
            "variant_type": "deletion",
            "position": 185,
            "ref": "AG",
            "alt": "",
            "clinical_significance": "Pathogenic",
            "review_status": "practice guideline"
        },
        {
            "variant_id": f"{gene}:c.5266dupC",
            "variant_type": "duplication",
            "position": 526,
            "ref": "C",
            "alt": "CC",
            "clinical_significance": "Pathogenic",
            "review_status": "practice guideline"
        },
        # Nonsense mutations (create stop codons)
        {
            "variant_id": f"{gene}:c.181T>G",
            "variant_type": "nonsense",
            "position": 181,
            "ref": "U",
            "alt": "G",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.916C>U",
            "variant_type": "nonsense",
            "position": 916,
            "ref": "C",
            "alt": "U",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.4035delA",
            "variant_type": "deletion",
            "position": 403,
            "ref": "A",
            "alt": "",
            "clinical_significance": "Pathogenic",
            "review_status": "practice guideline"
        },
        # Splice site mutations
        {
            "variant_id": f"{gene}:c.135-1G>A",
            "variant_type": "splice_site",
            "position": 135,
            "ref": "G",
            "alt": "A",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.594+1G>A",
            "variant_type": "splice_site",
            "position": 594,
            "ref": "G",
            "alt": "A",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters"
        },
        # Disruptive missense
        {
            "variant_id": f"{gene}:c.1687C>U",
            "variant_type": "missense",
            "position": 1687,
            "ref": "C",
            "alt": "U",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.5123C>A",
            "variant_type": "missense",
            "position": 512,
            "ref": "C",
            "alt": "A",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters"
        }
    ]

    # Benign variants (synonymous, non-disruptive missense)
    benign_variants = [
        {
            "variant_id": f"{gene}:c.213C>U",
            "variant_type": "synonymous",
            "position": 213,
            "ref": "C",
            "alt": "U",
            "clinical_significance": "Benign",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.441A>G",
            "variant_type": "synonymous",
            "position": 441,
            "ref": "A",
            "alt": "G",
            "clinical_significance": "Benign",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.2082C>U",
            "variant_type": "synonymous",
            "position": 2082,
            "ref": "C",
            "alt": "U",
            "clinical_significance": "Benign",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.3113A>G",
            "variant_type": "synonymous",
            "position": 3113,
            "ref": "A",
            "alt": "G",
            "clinical_significance": "Benign",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.4308U>C",
            "variant_type": "synonymous",
            "position": 4308,
            "ref": "U",
            "alt": "C",
            "clinical_significance": "Benign",
            "review_status": "criteria provided, multiple submitters"
        },
        # Non-disruptive missense (common polymorphisms)
        {
            "variant_id": f"{gene}:c.2311U>C",
            "variant_type": "missense",
            "position": 2311,
            "ref": "U",
            "alt": "C",
            "clinical_significance": "Benign",
            "review_status": "criteria provided, multiple submitters"
        },
        {
            "variant_id": f"{gene}:c.3548A>G",
            "variant_type": "missense",
            "position": 3548,
            "ref": "A",
            "alt": "G",
            "clinical_significance": "Likely benign",
            "review_status": "criteria provided, multiple submitters"
        }
    ]

    # VUS (Variants of Uncertain Significance) - the clinical challenge
    vus_variants = [
        {
            "variant_id": f"{gene}:c.1234A>C",
            "variant_type": "missense",
            "position": 1234,
            "ref": "A",
            "alt": "C",
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, single submitter"
        },
        {
            "variant_id": f"{gene}:c.2456G>A",
            "variant_type": "missense",
            "position": 2456,
            "ref": "G",
            "alt": "A",
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, single submitter"
        },
        {
            "variant_id": f"{gene}:c.3789C>G",
            "variant_type": "missense",
            "position": 3789,
            "ref": "C",
            "alt": "G",
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, conflicting interpretations"
        },
        {
            "variant_id": f"{gene}:c.4521U>A",
            "variant_type": "missense",
            "position": 4521,
            "ref": "U",
            "alt": "A",
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, single submitter"
        },
        {
            "variant_id": f"{gene}:c.5234A>U",
            "variant_type": "missense",
            "position": 523,
            "ref": "A",
            "alt": "U",
            "clinical_significance": "Uncertain significance",
            "review_status": "no assertion criteria provided"
        }
    ]

    all_variants = pathogenic_variants + benign_variants + vus_variants

    # Generate RNA sequences for each variant
    for variant in all_variants:
        # Generate mutant sequence
        pos = min(variant['position'], len(reference) - 10)

        if variant['variant_type'] == 'deletion':
            mutant_seq = reference[:pos] + reference[pos+2:]
        elif variant['variant_type'] == 'duplication':
            mutant_seq = reference[:pos] + reference[pos] + reference[pos:]
        else:  # Substitution
            mutant_seq = reference[:pos] + variant['alt'] + reference[pos+1:]

        # Extract local context (100 nt window)
        start = max(0, pos - 50)
        end = min(len(reference), pos + 50)

        variant['wild_type_rna'] = reference[start:end]
        variant['mutant_rna'] = mutant_seq[start:end]
        variant['gene'] = gene

        variants.append(variant)

    return variants


def create_training_dataset(variants: List[Dict], output_path: str):
    """
    Convert ClinVar variants to training dataset format

    Args:
        variants: List of variant dictionaries
        output_path: Path to save training data
    """
    print("Creating training dataset...")

    training_samples = []

    for variant in variants:
        sample = {
            'variant_id': variant['variant_id'],
            'gene': variant['gene'],
            'wild_type_rna': variant['wild_type_rna'],
            'mutant_rna': variant['mutant_rna'],
            'label': 1 if variant['clinical_significance'] == 'Pathogenic' else 0,
            'clinical_significance': variant['clinical_significance']
        }
        training_samples.append(sample)

    # Save training data
    with open(output_path, 'w') as f:
        for sample in training_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Created {len(training_samples)} training samples")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download BRCA1/2 variants from ClinVar'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/breast_cancer/brca_mutations',
        help='Output directory for variant data'
    )
    parser.add_argument(
        '--genes',
        nargs='+',
        default=['BRCA1', 'BRCA2'],
        help='Genes to download (BRCA1 and/or BRCA2)'
    )
    parser.add_argument(
        '--use_real_api',
        action='store_true',
        help='Fetch real data from NCBI ClinVar API (requires biopython and internet)'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='Your email for NCBI API (optional but recommended for higher rate limits)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Download variants for each gene
    all_variants = []
    for gene in args.genes:
        variants = download_clinvar_variants(
            args.output,
            gene,
            use_real_api=args.use_real_api,
            email=args.email
        )
        all_variants.extend(variants)

    # Create training dataset
    train_path = os.path.join(args.output, 'train.jsonl')
    create_training_dataset(all_variants, train_path)

    print("\n" + "="*50)
    print("BRCA variant download complete!")
    print("="*50)
    print(f"Total variants: {len(all_variants)}")
    print(f"Genes: {', '.join(args.genes)}")
    print(f"Output directory: {args.output}")
    print("\nNext steps:")
    print("1. Review the downloaded variants")
    print("2. Fine-tune Genesis RNA on this dataset:")
    print(f"   python -m genesis_rna.train_finetune \\")
    print(f"       --task mutation_effect \\")
    print(f"       --train_data {train_path} \\")
    print(f"       --output_dir checkpoints/finetuned/brca_mutations")


if __name__ == '__main__':
    main()
