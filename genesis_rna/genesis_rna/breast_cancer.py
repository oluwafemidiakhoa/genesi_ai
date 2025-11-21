"""
Breast Cancer Analysis Module for Genesis RNA

This module provides specialized tools for breast cancer RNA analysis:
- BRCA1/2 mutation effect prediction
- TP53 variant pathogenicity classification
- mRNA therapeutic design for cancer treatment
- Neoantigen discovery for personalized vaccines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .model import GenesisRNAModel
from .config import GenesisRNAConfig
from .tokenization import RNATokenizer
from .heads import MutationEffectHead


@dataclass
class VariantPrediction:
    """Prediction for a genetic variant"""
    variant_id: str
    pathogenicity_score: float  # 0-1, higher = more pathogenic
    delta_stability: float  # ΔΔG in kcal/mol
    delta_expression: float  # Change in expression level
    interpretation: str  # Clinical interpretation
    confidence: float  # Prediction confidence


@dataclass
class TherapeuticmRNA:
    """Designed therapeutic mRNA sequence"""
    sequence: str
    protein_target: str
    stability_score: float
    translation_score: float
    immunogenicity_score: float
    half_life_hours: float
    optimization_goals: Dict[str, float]


@dataclass
class Neoantigen:
    """Tumor neoantigen for vaccine design"""
    peptide_sequence: str
    mrna_sequence: str
    mutation: str
    hla_binding_score: float
    immunogenicity_score: float
    expression_level: float


class BreastCancerAnalyzer:
    """
    Main interface for breast cancer RNA analysis using Genesis RNA

    Provides methods for:
    - Mutation effect prediction (BRCA1/2, TP53, etc.)
    - Pathogenicity classification
    - RNA stability analysis
    - Clinical interpretation
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize analyzer with trained Genesis RNA model

        Args:
            model_path: Path to fine-tuned model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.tokenizer = RNATokenizer()

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config_dict = checkpoint['config']['model']
        self.config = GenesisRNAConfig.from_dict(config_dict) if isinstance(config_dict, dict) else config_dict
        self.model = GenesisRNAModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Known cancer genes
        self.cancer_genes = {
            'BRCA1': 'Tumor suppressor - DNA repair',
            'BRCA2': 'Tumor suppressor - DNA repair',
            'TP53': 'Tumor suppressor - cell cycle control',
            'HER2': 'Oncogene - growth factor receptor',
            'PIK3CA': 'Oncogene - cell signaling',
            'ESR1': 'Estrogen receptor - hormone signaling',
            'PTEN': 'Tumor suppressor - PI3K pathway',
            'CDH1': 'Tumor suppressor - cell adhesion',
            'ATM': 'Tumor suppressor - DNA damage response',
            'CHEK2': 'Tumor suppressor - cell cycle checkpoint'
        }

    def predict_variant_effect(
        self,
        gene: str,
        wild_type_rna: str,
        mutant_rna: str,
        variant_id: Optional[str] = None
    ) -> VariantPrediction:
        """
        Predict the effect of a variant on RNA function

        Args:
            gene: Gene name (e.g., 'BRCA1')
            wild_type_rna: Wild-type RNA sequence
            mutant_rna: Mutant RNA sequence
            variant_id: Variant identifier (e.g., 'c.5266dupC')

        Returns:
            VariantPrediction with pathogenicity and effect predictions
        """
        with torch.no_grad():
            # Encode sequences (tokenizer.encode returns tensor directly)
            wt_ids = self.tokenizer.encode(wild_type_rna, max_len=self.config.max_len).unsqueeze(0).to(self.device)
            mut_ids = self.tokenizer.encode(mutant_rna, max_len=self.config.max_len).unsqueeze(0).to(self.device)

            # Forward pass
            wt_output = self.model(wt_ids)
            mut_output = self.model(mut_ids)

            # Compute stability difference (using MLM perplexity as proxy)
            wt_logits = wt_output['mlm_logits']
            mut_logits = mut_output['mlm_logits']

            wt_perplexity = torch.exp(F.cross_entropy(
                wt_logits.view(-1, self.config.vocab_size),
                wt_ids.view(-1),
                reduction='mean'
            )).item()

            mut_perplexity = torch.exp(F.cross_entropy(
                mut_logits.view(-1, self.config.vocab_size),
                mut_ids.view(-1),
                reduction='mean'
            )).item()

            # Higher perplexity = less stable
            delta_stability = (wt_perplexity - mut_perplexity) * 0.5  # Scale to kcal/mol

            # Estimate pathogenicity score
            # Based on structural changes and gene context
            struct_change = self._compute_structure_change(wt_output, mut_output)

            # Tumor suppressors: destabilizing = pathogenic
            # Oncogenes: stabilizing = pathogenic
            if gene in ['BRCA1', 'BRCA2', 'TP53', 'PTEN', 'CDH1', 'ATM', 'CHEK2']:
                pathogenicity = 1 / (1 + np.exp(-5 * (struct_change - 0.3)))  # Sigmoid
            else:  # Oncogene
                pathogenicity = 1 / (1 + np.exp(5 * (struct_change - 0.3)))

            # Clinical interpretation
            if pathogenicity > 0.8:
                interpretation = "Likely Pathogenic"
            elif pathogenicity > 0.5:
                interpretation = "Uncertain Significance - Likely Pathogenic"
            elif pathogenicity > 0.2:
                interpretation = "Uncertain Significance"
            else:
                interpretation = "Likely Benign"

            # Confidence based on model uncertainty
            confidence = 1.0 - struct_change  # Higher change = lower confidence

            return VariantPrediction(
                variant_id=variant_id or f"{gene}:variant",
                pathogenicity_score=pathogenicity,
                delta_stability=delta_stability,
                delta_expression=0.0,  # Placeholder
                interpretation=interpretation,
                confidence=max(0.5, confidence)
            )

    def _compute_structure_change(
        self,
        wt_output: Dict[str, torch.Tensor],
        mut_output: Dict[str, torch.Tensor]
    ) -> float:
        """Compute structural change between wild-type and mutant"""
        wt_struct = F.softmax(wt_output['struct_logits'], dim=-1)
        mut_struct = F.softmax(mut_output['struct_logits'], dim=-1)

        # Jensen-Shannon divergence
        m = 0.5 * (wt_struct + mut_struct)
        js_div = 0.5 * (
            F.kl_div(torch.log(wt_struct + 1e-10), m, reduction='batchmean') +
            F.kl_div(torch.log(mut_struct + 1e-10), m, reduction='batchmean')
        )

        return js_div.item()

    def classify_brca_variant(self, variant_rna: str, variant_id: str) -> str:
        """
        Classify BRCA1/2 variant as pathogenic or benign

        Args:
            variant_rna: RNA sequence with variant
            variant_id: Variant identifier

        Returns:
            Classification: Pathogenic/Benign/VUS
        """
        # This is a simplified version - would need wild-type reference
        # In practice, would compare against reference sequence

        with torch.no_grad():
            input_ids = self.tokenizer.encode(variant_rna, max_len=self.config.max_len).unsqueeze(0).to(self.device)

            output = self.model(input_ids)

            # Use structure prediction as a proxy for pathogenicity
            struct_probs = F.softmax(output['struct_logits'], dim=-1)

            # Abnormal structure distribution suggests pathogenic
            entropy = -torch.sum(struct_probs * torch.log(struct_probs + 1e-10), dim=-1)
            mean_entropy = entropy.mean().item()

            if mean_entropy > 1.5:
                return "Pathogenic"
            elif mean_entropy > 1.0:
                return "VUS (Variant of Uncertain Significance)"
            else:
                return "Benign"


class mRNATherapeuticDesigner:
    """
    Design optimized mRNA therapeutics for cancer treatment

    Focuses on:
    - Codon optimization for translation
    - Structure optimization for stability
    - Immunogenicity minimization
    - UTR design for regulation
    """

    def __init__(self, model: GenesisRNAModel, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.tokenizer = RNATokenizer()

        # Codon usage table (human optimized)
        self.optimal_codons = {
            'A': ['GCU', 'GCC'],  # Alanine
            'C': ['UGU', 'UGC'],  # Cysteine
            'D': ['GAU', 'GAC'],  # Aspartic acid
            'E': ['GAA', 'GAG'],  # Glutamic acid
            'F': ['UUU', 'UUC'],  # Phenylalanine
            'G': ['GGU', 'GGC'],  # Glycine
            'H': ['CAU', 'CAC'],  # Histidine
            'I': ['AUU', 'AUC'],  # Isoleucine
            'K': ['AAA', 'AAG'],  # Lysine
            'L': ['CUG', 'UUG'],  # Leucine
            'M': ['AUG'],         # Methionine (start)
            'N': ['AAU', 'AAC'],  # Asparagine
            'P': ['CCU', 'CCC'],  # Proline
            'Q': ['CAA', 'CAG'],  # Glutamine
            'R': ['CGU', 'AGA'],  # Arginine
            'S': ['UCU', 'AGC'],  # Serine
            'T': ['ACU', 'ACC'],  # Threonine
            'V': ['GUU', 'GUC'],  # Valine
            'W': ['UGG'],         # Tryptophan
            'Y': ['UAU', 'UAC'],  # Tyrosine
            '*': ['UAA', 'UAG', 'UGA']  # Stop codons
        }

    def design(
        self,
        protein_sequence: str,
        optimization_goals: Dict[str, float] = None
    ) -> TherapeuticmRNA:
        """
        Design optimized mRNA encoding the target protein

        Args:
            protein_sequence: Target protein amino acid sequence
            optimization_goals: Dict of optimization targets
                - 'stability': 0-1 (higher = more stable)
                - 'translation': 0-1 (higher = more translation)
                - 'immunogenicity': 0-1 (lower = less immunogenic)

        Returns:
            TherapeuticmRNA with optimized sequence
        """
        if optimization_goals is None:
            optimization_goals = {
                'stability': 0.9,
                'translation': 0.9,
                'immunogenicity': 0.1
            }

        # Step 1: Codon optimization
        mrna_sequence = self._optimize_codons(protein_sequence)

        # Step 2: Add 5' and 3' UTRs
        mrna_sequence = self._add_utrs(mrna_sequence)

        # Step 3: Evaluate with model
        scores = self._evaluate_mrna(mrna_sequence)

        # Step 4: Iterative refinement (simplified)
        # In practice, would use gradient-based optimization
        for _ in range(5):
            candidate = self._refine_sequence(mrna_sequence, optimization_goals, scores)
            candidate_scores = self._evaluate_mrna(candidate)

            if self._is_better(candidate_scores, scores, optimization_goals):
                mrna_sequence = candidate
                scores = candidate_scores

        # Estimate half-life (based on stability score)
        half_life = scores['stability'] * 24.0  # 0-24 hours

        return TherapeuticmRNA(
            sequence=mrna_sequence,
            protein_target=protein_sequence,
            stability_score=scores['stability'],
            translation_score=scores['translation'],
            immunogenicity_score=scores['immunogenicity'],
            half_life_hours=half_life,
            optimization_goals=optimization_goals
        )

    def _optimize_codons(self, protein_seq: str) -> str:
        """Convert protein to optimal codons"""
        mrna = ""
        for aa in protein_seq:
            if aa in self.optimal_codons:
                # Choose most optimal codon
                mrna += self.optimal_codons[aa][0]
            else:
                raise ValueError(f"Unknown amino acid: {aa}")
        return mrna

    def _add_utrs(self, coding_sequence: str) -> str:
        """Add optimized 5' and 3' UTRs"""
        # Kozak sequence for efficient translation
        utr_5 = "GCCACCAUGG"

        # Poly(A) signal for stability
        utr_3 = "AAUAAA" + "A" * 100

        return utr_5 + coding_sequence + utr_3

    def _evaluate_mrna(self, sequence: str) -> Dict[str, float]:
        """Evaluate mRNA sequence with Genesis RNA model"""
        with torch.no_grad():
            input_ids = self.tokenizer.encode(sequence, max_len=min(len(sequence) + 10, 512)).unsqueeze(0).to(self.device)

            output = self.model(input_ids)

            # Stability: low perplexity = stable
            logits = output['mlm_logits']
            perplexity = torch.exp(F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction='mean'
            )).item()

            stability = 1.0 / (1.0 + perplexity / 10.0)

            # Translation: based on structure (more structured 5' UTR = better)
            struct_logits = output['struct_logits']
            struct_probs = F.softmax(struct_logits, dim=-1)

            # Want structured 5' UTR, unstructured coding region
            translation = struct_probs[:, :50, 1].mean().item()  # STEM structures in UTR

            # Immunogenicity: low entropy = low immunogenicity
            entropy = -torch.sum(struct_probs * torch.log(struct_probs + 1e-10), dim=-1)
            immunogenicity = entropy.mean().item() / 2.0  # Normalize to 0-1

            return {
                'stability': min(1.0, stability),
                'translation': min(1.0, translation),
                'immunogenicity': min(1.0, immunogenicity)
            }

    def _refine_sequence(
        self,
        sequence: str,
        goals: Dict[str, float],
        current_scores: Dict[str, float]
    ) -> str:
        """Refine sequence to better meet optimization goals"""
        # Simplified refinement: random synonymous codon substitution
        # In practice, would use more sophisticated optimization

        seq_list = list(sequence)

        # Randomly mutate a codon
        pos = np.random.randint(10, len(seq_list) - 110, size=1)[0]
        pos = (pos // 3) * 3  # Ensure codon boundary

        if pos + 3 < len(seq_list):
            # Replace with alternative codon
            original_codon = ''.join(seq_list[pos:pos+3])
            # Simplistic: just swap U<->C (often synonymous)
            new_codon = original_codon.replace('U', 'c').replace('C', 'U').replace('c', 'C')
            seq_list[pos:pos+3] = list(new_codon)

        return ''.join(seq_list)

    def _is_better(
        self,
        candidate_scores: Dict[str, float],
        current_scores: Dict[str, float],
        goals: Dict[str, float]
    ) -> bool:
        """Check if candidate is better than current"""
        # Weighted score based on goals
        def weighted_score(scores):
            return (
                abs(scores['stability'] - goals['stability']) +
                abs(scores['translation'] - goals['translation']) +
                abs(scores['immunogenicity'] - goals['immunogenicity'])
            )

        return weighted_score(candidate_scores) < weighted_score(current_scores)


class NeoantigenDiscovery:
    """
    Discover tumor neoantigens for personalized cancer vaccines

    Pipeline:
    1. Compare tumor vs normal RNA-seq
    2. Identify tumor-specific mutations
    3. Predict which mutations create immunogenic peptides
    4. Design mRNA vaccine encoding top neoantigens
    """

    def __init__(self, model: GenesisRNAModel, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.tokenizer = RNATokenizer()
        self.config = model.cfg  # Access model's config

    def find_neoantigens(
        self,
        tumor_sequences: List[str],
        normal_sequences: List[str],
        hla_type: str = "HLA-A*02:01"
    ) -> List[Neoantigen]:
        """
        Find tumor-specific neoantigens

        Args:
            tumor_sequences: RNA sequences from tumor
            normal_sequences: RNA sequences from normal tissue
            hla_type: Patient's HLA type for binding prediction

        Returns:
            List of predicted neoantigens sorted by immunogenicity
        """
        neoantigens = []

        # Simplified: In practice, would do actual variant calling
        for tumor_seq in tumor_sequences[:10]:  # Limit for demo
            # Predict if sequence is immunogenic
            immunogenicity = self._predict_immunogenicity(tumor_seq)

            if immunogenicity > 0.5:
                # Create neoantigen
                neoantigen = Neoantigen(
                    peptide_sequence="PEPTIDE",  # Would translate from RNA
                    mrna_sequence=tumor_seq,
                    mutation="Tumor-specific",
                    hla_binding_score=0.8,  # Placeholder
                    immunogenicity_score=immunogenicity,
                    expression_level=1.0  # Placeholder
                )
                neoantigens.append(neoantigen)

        # Sort by immunogenicity
        neoantigens.sort(key=lambda x: x.immunogenicity_score, reverse=True)

        return neoantigens

    def _predict_immunogenicity(self, sequence: str) -> float:
        """Predict if RNA sequence will generate immunogenic peptide"""
        with torch.no_grad():
            input_ids = self.tokenizer.encode(sequence, max_len=self.config.max_len).unsqueeze(0).to(self.device)

            output = self.model(input_ids)

            # Use structural features as proxy for immunogenicity
            struct_logits = output['struct_logits']
            struct_probs = F.softmax(struct_logits, dim=-1)

            # Higher diversity = more immunogenic
            entropy = -torch.sum(struct_probs * torch.log(struct_probs + 1e-10), dim=-1)
            immunogenicity = torch.sigmoid(entropy - 1.0).mean().item()

            return immunogenicity

    def design_vaccine(
        self,
        neoantigens: List[Neoantigen],
        adjuvant: str = "lipid_nanoparticle"
    ) -> str:
        """
        Design personalized mRNA vaccine encoding multiple neoantigens

        Args:
            neoantigens: List of neoantigens to include
            adjuvant: Adjuvant/delivery system

        Returns:
            Vaccine mRNA sequence
        """
        # Concatenate neoantigen sequences with linkers
        vaccine_mrna = ""

        for i, neoantigen in enumerate(neoantigens):
            vaccine_mrna += neoantigen.mrna_sequence

            # Add flexible linker between antigens (except last)
            if i < len(neoantigens) - 1:
                vaccine_mrna += "GGCGGCGGCGGC"  # (Gly-Gly)n linker

        # Add 5' cap and 3' poly(A) for stability
        vaccine_mrna = "GCCACCAUGG" + vaccine_mrna + "AAUAAA" + "A" * 100

        return vaccine_mrna
