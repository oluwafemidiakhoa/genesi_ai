#!/usr/bin/env python3
"""
Breast Cancer Analysis Demo

This demonstrates how to use Genesis RNA for breast cancer research
even without a fully trained model (uses dummy predictions for demo).
"""

import sys
sys.path.insert(0, 'genesis_rna')

from genesis_rna.breast_cancer import (
    BreastCancerAnalyzer,
    mRNATherapeuticDesigner,
    VariantPrediction
)


def demo_variant_classification():
    """Demo: Classify BRCA1 variant as pathogenic or benign"""
    print("="*60)
    print("DEMO 1: BRCA1 Variant Classification")
    print("="*60)

    # Example BRCA1 sequences (simplified for demo)
    wt_brca1 = "AUGGGCUUCCGUGUCCAGCUCCUGGGAGCUGCUGGUGGCGGCGGCCGCGGGC"

    # Known pathogenic mutation: c.5266dupC (frameshift)
    mut_brca1 = "AUGGGCUUCCGUGUCCAGCUCCUGGGAGCUGCUGGUGGCGGCGGCCCGCGGGC"

    print(f"\nWild-type BRCA1: {wt_brca1[:30]}...")
    print(f"Mutant BRCA1:    {mut_brca1[:30]}...")

    # For demo purposes, create a mock prediction
    # In real use, you'd load a trained model:
    # analyzer = BreastCancerAnalyzer('checkpoints/pretrained/base/best_model.pt')
    # prediction = analyzer.predict_variant_effect('BRCA1', wt_brca1, mut_brca1)

    # Mock prediction (demonstrates the output format)
    prediction = VariantPrediction(
        variant_id="BRCA1:c.5266dupC",
        pathogenicity_score=0.892,
        delta_stability=-2.34,
        delta_expression=-0.45,
        interpretation="Likely Pathogenic",
        confidence=0.856
    )

    print(f"\n{'Variant ID:':<20} {prediction.variant_id}")
    print(f"{'Pathogenicity:':<20} {prediction.pathogenicity_score:.3f} (0=benign, 1=pathogenic)")
    print(f"{'ΔStability:':<20} {prediction.delta_stability:.2f} kcal/mol")
    print(f"{'ΔExpression:':<20} {prediction.delta_expression:.2f}")
    print(f"{'Interpretation:':<20} {prediction.interpretation}")
    print(f"{'Confidence:':<20} {prediction.confidence:.3f}")

    print("\n" + "="*60)
    print("Clinical Interpretation:")
    print("="*60)
    print("This BRCA1 variant (c.5266dupC) is a known pathogenic")
    print("frameshift mutation that:")
    print("  • Disrupts DNA repair function")
    print("  • Increases breast cancer risk 5-10x")
    print("  • Indicates need for enhanced screening")
    print("  • May qualify for PARP inhibitor therapy")


def demo_therapeutic_design():
    """Demo: Design mRNA therapeutic for p53"""
    print("\n\n" + "="*60)
    print("DEMO 2: mRNA Therapeutic Design")
    print("="*60)

    # p53 protein sequence (first 50 amino acids for demo)
    p53_protein = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDD"

    print(f"\nTarget: p53 tumor suppressor")
    print(f"Protein sequence (partial): {p53_protein}")
    print(f"\nOptimization Goals:")
    print(f"  • High stability (0.95)")
    print(f"  • High translation (0.90)")
    print(f"  • Low immunogenicity (0.05)")

    # For demo, show what the output would look like
    print(f"\nDesigned mRNA Properties:")
    print(f"{'Sequence length:':<25} 1,853 nucleotides")
    print(f"{'Stability score:':<25} 0.891")
    print(f"{'Translation score:':<25} 0.856")
    print(f"{'Immunogenicity score:':<25} 0.112")
    print(f"{'Predicted half-life:':<25} 21.4 hours")

    # Example optimized codon usage
    optimized_mrna = "AUGGAAGAGCCACAGUCCGACGUCGUGAGCCGCCGCUG..."
    print(f"\nOptimized mRNA (first 50 nt):")
    print(f"  {optimized_mrna}")

    print("\n" + "="*60)
    print("Therapeutic Application:")
    print("="*60)
    print("This mRNA therapeutic could:")
    print("  • Deliver functional p53 to p53-mutant tumors")
    print("  • Restore cell cycle control and apoptosis")
    print("  • Work in ~30% of breast cancers with p53 mutations")
    print("  • Be delivered via lipid nanoparticles (like COVID vaccines)")


def demo_research_workflow():
    """Demo: Complete research workflow"""
    print("\n\n" + "="*60)
    print("DEMO 3: Complete Research Workflow")
    print("="*60)

    print("\nStep 1: Variant Discovery")
    print("  • Sequence patient's tumor DNA/RNA")
    print("  • Identify mutations in BRCA1, BRCA2, TP53, etc.")
    print("  • Found: BRCA1 c.5266dupC mutation")

    print("\nStep 2: Variant Classification (using Genesis RNA)")
    print("  • Predict pathogenicity: 0.89 (Likely Pathogenic)")
    print("  • Estimate functional impact: Loss of DNA repair")
    print("  • Clinical action: Enhanced screening, consider PARP inhibitors")

    print("\nStep 3: Therapeutic Design")
    print("  • Tumor has BRCA1 deficiency → DNA repair defect")
    print("  • Design mRNA encoding functional BRCA1")
    print("  • Optimize for tumor delivery and expression")

    print("\nStep 4: Personalized Vaccine (if applicable)")
    print("  • Identify tumor-specific mutations (neoantigens)")
    print("  • Design mRNA vaccine encoding these antigens")
    print("  • Train immune system to attack cancer cells")

    print("\nStep 5: Treatment Plan")
    print("  ✓ PARP inhibitor (exploits DNA repair defect)")
    print("  ✓ mRNA therapeutic (restores BRCA1 function)")
    print("  ✓ Personalized vaccine (immune activation)")
    print("  ✓ Standard chemotherapy (if needed)")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("GENESIS RNA: BREAST CANCER CURE RESEARCH DEMO")
    print("="*60)
    print("\nThis demonstrates how Genesis RNA accelerates")
    print("breast cancer research through AI-powered RNA analysis.")
    print("\nNote: This demo uses mock predictions for illustration.")
    print("For real predictions, train Genesis RNA first (see README).")

    demo_variant_classification()
    demo_therapeutic_design()
    demo_research_workflow()

    print("\n\n" + "="*60)
    print("Next Steps to Start Real Research:")
    print("="*60)
    print("\n1. Train Genesis RNA foundation model:")
    print("   python -m genesis_rna.train_pretrain \\")
    print("       --config configs/train_t4_optimized.yaml \\")
    print("       --num_epochs 10")

    print("\n2. Download breast cancer data:")
    print("   python scripts/download_brca_variants.py \\")
    print("       --output data/breast_cancer")

    print("\n3. Run real variant predictions:")
    print("   python examples/breast_cancer_analysis.py")

    print("\n4. Read the documentation:")
    print("   • BREAST_CANCER_RESEARCH.md - Comprehensive guide")
    print("   • BREAST_CANCER_QUICKSTART.md - Quick start tutorial")

    print("\n" + "="*60)
    print("Together, we can cure breast cancer! 🎗️")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
