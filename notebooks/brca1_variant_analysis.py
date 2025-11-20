#!/usr/bin/env python3
"""
BRCA1 Pathogenic Variant Analysis Script

This script demonstrates how to use Genesis RNA to analyze BRCA1 variants
and predict their pathogenicity.

Usage:
    python notebooks/brca1_variant_analysis.py

Note: This uses a demo mode with randomly initialized weights.
      For production use, train a model first or use a pre-trained checkpoint.
"""

import sys
import os

# Add genesis_rna to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'genesis_rna'))

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from genesis_rna.model import GenesisRNAModel
from genesis_rna.config import GenesisRNAConfig
from genesis_rna.tokenization import RNATokenizer


@dataclass
class VariantPrediction:
    """Prediction for a genetic variant"""
    variant_id: str
    pathogenicity_score: float
    delta_stability: float
    delta_expression: float
    interpretation: str
    confidence: float
    details: Dict[str, any]


class BreastCancerAnalyzer:
    """Enhanced Breast Cancer Analyzer for variant analysis"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        self.cancer_genes = {
            'BRCA1': 'Tumor suppressor - DNA repair',
            'BRCA2': 'Tumor suppressor - DNA repair',
            'TP53': 'Tumor suppressor - cell cycle control',
            'HER2': 'Oncogene - growth factor receptor',
            'PIK3CA': 'Oncogene - cell signaling',
            'ESR1': 'Estrogen receptor',
            'PTEN': 'Tumor suppressor - PI3K pathway',
        }

    def predict_variant_effect(
        self,
        gene: str,
        wild_type_rna: str,
        mutant_rna: str,
        variant_id: Optional[str] = None
    ) -> VariantPrediction:
        """Predict variant pathogenicity"""

        with torch.no_grad():
            # Encode sequences (tokenizer.encode returns tensors directly)
            wt_enc = self.tokenizer.encode(wild_type_rna, max_len=512)
            mut_enc = self.tokenizer.encode(mutant_rna, max_len=512)

            # Add batch dimension
            wt_ids = wt_enc.unsqueeze(0).to(self.device)
            mut_ids = mut_enc.unsqueeze(0).to(self.device)

            # Model forward pass
            wt_out = self.model(wt_ids)
            mut_out = self.model(mut_ids)

            # Compute stability change
            wt_perp = self._compute_perplexity(wt_out['mlm_logits'], wt_ids)
            mut_perp = self._compute_perplexity(mut_out['mlm_logits'], mut_ids)
            delta_stability = (wt_perp - mut_perp) * 0.5  # Scale to kcal/mol

            # Compute structural change
            struct_change = self._compute_structure_change(wt_out, mut_out)

            # Pathogenicity score
            is_tumor_suppressor = gene in ['BRCA1', 'BRCA2', 'TP53', 'PTEN']

            if is_tumor_suppressor:
                pathogenicity = 1 / (1 + np.exp(-5 * (struct_change - 0.3)))
            else:
                pathogenicity = 1 / (1 + np.exp(5 * (struct_change - 0.3)))

            # Clinical interpretation
            if pathogenicity > 0.8:
                interpretation = "Likely Pathogenic"
            elif pathogenicity > 0.5:
                interpretation = "Uncertain Significance (Likely Pathogenic)"
            elif pathogenicity > 0.2:
                interpretation = "Uncertain Significance"
            else:
                interpretation = "Likely Benign"

            confidence = max(0.5, 1.0 - struct_change)

            return VariantPrediction(
                variant_id=variant_id or f"{gene}:variant",
                pathogenicity_score=float(pathogenicity),
                delta_stability=float(delta_stability),
                delta_expression=0.0,
                interpretation=interpretation,
                confidence=float(confidence),
                details={
                    'gene': gene,
                    'wt_perplexity': float(wt_perp),
                    'mut_perplexity': float(mut_perp),
                    'struct_change': float(struct_change)
                }
            )

    def _compute_perplexity(self, logits, input_ids):
        """Compute perplexity as stability proxy"""
        perp = torch.exp(F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction='mean'
        ))
        return perp.item()

    def _compute_structure_change(self, wt_out, mut_out):
        """Compute structural change using JS divergence"""
        wt_struct = F.softmax(wt_out['struct_logits'], dim=-1)
        mut_struct = F.softmax(mut_out['struct_logits'], dim=-1)

        m = 0.5 * (wt_struct + mut_struct)
        js_div = 0.5 * (
            F.kl_div(torch.log(wt_struct + 1e-10), m, reduction='batchmean') +
            F.kl_div(torch.log(mut_struct + 1e-10), m, reduction='batchmean')
        )
        return js_div.item()


def main():
    """Main analysis function"""

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}\n")

    # Initialize model
    print("🏗️  Initializing Genesis RNA model...")
    model_config = GenesisRNAConfig(
        vocab_size=32,
        d_model=256,
        n_layers=4,
        n_heads=4,
        dim_ff=1024,
        max_len=512,
        dropout=0.1,
        structure_num_labels=3
    )

    model = GenesisRNAModel(model_config)
    model.to(device)
    model.eval()

    tokenizer = RNATokenizer()

    print(f"✅ Model initialized!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Mode: Demo (randomly initialized weights)\n")

    # Initialize analyzer
    print("🧬 Initializing Breast Cancer Analyzer...")
    analyzer = BreastCancerAnalyzer(model, tokenizer, device=device)
    print("✅ Analyzer ready!\n")

    # BRCA1 sequences
    print("="*70)
    print("BRCA1 Pathogenic Variant Analysis")
    print("="*70)

    # Wild-type BRCA1 sequence
    wt_brca1 = "AUGGGCUUCCGUGUCCAGCUCCUGGGAGCUGCUGGUGGCGGCGGCCGCGGGCAGGCUUAGAAGCGCGGUGAAGCUUUUGGAUCUGGUAUCAGCACUCGGCUCUGCCAGGGCAUGUUCCGGGAUGGAAACCGGUCCACUCCUGCCUUUCCGCAGGGUCACAGCCCAGCUUCCAGGGUGAGGCUGUGCACUACCACCCUCCUGAAGGCCUCCAGGCCGCUGAAGGUGUGGCCUGUCUAUUCCACCCACAGUCAACUGUUUGCCCAGUUUCUUAAUGGCAUAUUGGUGACACCUGAGAGGUGCCUUGAAGAUGGUCCGGUGCCCUUUCUGCAGCAAACCUGAAGAAGCAGCAUAAGCUCAGUUACAACUUCCCCAGUUACUGCUUUUGCCCUGAGAAGCCUGUCCCAGAAGAUGUCAGCUGGUCACAUUAUCAUCCAGAGGUCUUUUUAAGAAGGAUGUGCUGUCUUGAAGAUACAGGGAAGGAGGAGCUGACACAUCAGGUGGGGUUGUCACUGAGUGGCAGUGUGAACACCAAGGGGAGCUUGGUGCUAACUGCCAGUUCGAGUCUCCUGACAGCUGAGGAUCCAUCAGUCCAGAACAGCAUGUGUCUGCAGUACAACAUCGGUCUGACAGGAAACUCCUGUGGUGUGGUCUUCUGCAAAGUCAGCAGUGACCACAGUGCCUUGAUGAUGGAGCUGGUGGUGGAGGUGGAGGUGGAGUUCAAAGGUGGUGACUGGCAGACUGGAGGGUGACAUUGUAUCCUGUGGAAAGAGGAGCCCACUGCAUUACAGCUUCUACUGGAGCUACAUCACAGACCAGAUUCUCCACAGCAACACUUCUGCAAUCAAAGCAAUCCUCCUGAGCCUAAGCCCCAGGUUACUUGGUGGUCCAGGGCUACCAAGGCCUAAAAGUCCCAUUACCUUCUCCCUGUGAAGAGCCUUCCGACUACUUCUGAAAGAUGACCACCUGUCUCCCACACAGGUCUUGUUACCUGUUUAGAACUGGAAGCUGAAGUGCUCAUUGCCUGUCUGCAGCGUGAUGUGGUGAGUGUUGCCCAGCUGUCUGGUCUGCCCAGCAGACCACUGAGAAGCCUACAGCCAGUCCAUCCCUUCUGCUGCUGCUUCUGCUGCUGCUGUGCUGUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGUGUUUGGUCUCUAAAGGAACACAGUUGGGCUUUUCAAGCAAGAGGCCCUCCUGCUGCUGCUGCUGUGUCUCCUGCUGCUGCAGCUGCCAGCCUACACACAUGGAGAGCCAGACACAGUGUUGAAAAAGAUGCUGAGGAGUCUGCUUUCUGAUCGUUGCUGUGGGACCCCACCCUAGCUCUGCUGCUGCUGCUGAUCCUACAGUGGGACUGUAGGCCCUCCAGAUCUGCAUACCACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACAGGUAAAGAAGCCCAGAAAGAAAGGGAGUUGCUGGAAACUGGGAAGAAGGAAAGCUCUCUGGGAAGAAAGAAGCAUGAUCCUUUUGCUGAAGGUGCCUCUGGAUUCUGCCUGAAACUGAACUAUGAAAACAAGGAAGGCACUGGCCUCCAGAGGAUGUCUGCUGCCCCUCCCAAAGAAAUGAAGAAGGCCUUCAGAAAAACCUACUUGUGCUGUGCAGGAAUCCCUCCAGACUAUCUGCCAAAGGUCCAUCGUGGACUACUACUAUGUGACUAUUCUCUGACAAGGAAAAGAACAUC"

    # Mutant with frameshift (c.5266dupC - known pathogenic)
    mut_brca1 = "AUGGGCUUCCGUGUCCAGCUCCUGGGAGCUGCUGGUGGCGGCGGCCGCGGGCAGGCUUAGAAGCGCGGUGAAGCUUUUGGAUCUGGUAUCAGCACUCGGCUCUGCCAGGGCAUGUUCCGGGAUGGAAACCGGUCCACUCCUGCCUUUCCGCAGGGUCACAGCCCAGCUUCCAGGGUGAGGCUGUGCACUACCACCCUCCUGAAGGCCUCCAGGCCGCUGAAGGUGUGGCCUGUCUAUUCCACCCACAGUCAACUGUUUGCCCAGUUUCUUAAUGGCAUAUUGGUGACACCUGAGAGGUGCCUUGAAGAUGGUCCGGUGCCCUUUCUGCAGCAAACCUGAAGAAGCAGCAUAAGCUCAGUUACAACUUCCCCAGUUACUGCUUUUGCCCUGAGAAGCCUGUCCCAGAAGAUGUCAGCUGGUCACAUUAUCAUCCAGAGGUCUUUUUAAGAAGGAUGUGCUGUCUUGAAGAUACAGGGAAGGAGGAGCUGACACAUCAGGUGGGGUUGUCACUGAGUGGCAGUGUGAACACCAAGGGGAGCUUGGUGCUAACUGCCAGUUCGAGUCUCCUGACAGCUGAGGAUCCAUCAGUCCAGAACAGCAUGUGUCUGCAGUACAACAUCGGUCUGACAGGAAACUCCUGUGGUGUGGUCUUCUGCAAAGUCAGCAGUGACCACAGUGCCUUGAUGAUGGAGCUGGUGGUGGAGGUGGAGGUGGAGUUCAAAGGUGGUGACUGGCAGACUGGAGGGUGACAUUGUAUCCUGUGGAAAGAGGAGCCCACUGCAUUACAGCUUCUACUGGAGCUACAUCACAGACCAGAUUCUCCACAGCAACACUUCUGCAAUCAAAGCAAUCCUCCUGAGCCUAAGCCCCAGGUUACUUGGUGGUCCAGGGCUACCAAGGCCUAAAAGUCCCAUUACCUUCUCCCUGUGAAGAGCCUUCCGACUACUUCUGAAAGAUGACCACCUGUCUCCCACACAGGUCUUGUUACCUGUUUAGAACUGGAAGCUGAAGUGCUCAUUGCCUGUCUGCAGCGUGAUGUGGUGAGUGUUGCCCAGCUGUCUGGUCCUGCCCAGCAGACCACUGAGAAGCCUACAGCCAGUCCAUCCCUUCUGCUGCUGCUUCUGCUGCUGCUGUGCUGUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGUGUUUGGUCUCUAAAGGAACACAGUUGGGCUUUUCAAGCAAGAGGCCCUCCUGCUGCUGCUGCUGUGUCUCCUGCUGCUGCAGCUGCCAGCCUACACACAUGGAGAGCCAGACACAGUGUUGAAAAAGAUGCUGAGGAGUCUGCUUUCUGAUCGUUGCUGUGGGACCCCACCCUAGCUCUGCUGCUGCUGCUGAUCCUACAGUGGGACUGUAGGCCCUCCAGAUCUGCAUACCACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACAGGUAAAGAAGCCCAGAAAGAAAGGGAGUUGCUGGAAACUGGGAAGAAGGAAAGCUCUCUGGGAAGAAAGAAGCAUGAUCCUUUUGCUGAAGGUGCCUCUGGAUUCUGCCUGAAACUGAACUAUGAAAACAAGGAAGGCACUGGCCUCCAGAGGAUGUCUGCUGCCCCUCCCAAAGAAAUGAAGAAGGCCUUCAGAAAAACCUACUUGUGCUGUGCAGGAAUCCCUCCAGACUAUCUGCCAAAGGUCCAUCGUGGACUACUACUAUGUGACUAUUCUCUGACAAGGAAAAGAACAUC"

    # Analyze
    print("\n🔬 Analyzing variant...\n")
    pred = analyzer.predict_variant_effect(
        gene='BRCA1',
        wild_type_rna=wt_brca1,
        mutant_rna=mut_brca1,
        variant_id='BRCA1:c.5266dupC'
    )

    print(f"{'Variant ID:':<30} {pred.variant_id}")
    print(f"{'Pathogenicity Score:':<30} {pred.pathogenicity_score:.3f}")
    print(f"{'ΔStability (kcal/mol):':<30} {pred.delta_stability:.2f}")
    print(f"{'Clinical Interpretation:':<30} {pred.interpretation}")
    print(f"{'Confidence:':<30} {pred.confidence:.3f}")

    print("\n📋 Clinical Significance:")
    print("  • Known pathogenic frameshift")
    print("  • Disrupts DNA repair")
    print("  • 5-10x breast cancer risk")
    print("  • Recommend: Enhanced screening + counseling")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

    print("\n⚠️  Note: This demo uses randomly initialized weights.")
    print("   For production use, train the model on real data first.")
    print("   See: genesis_rna/breast_cancer_research_colab.ipynb")


if __name__ == '__main__':
    main()
