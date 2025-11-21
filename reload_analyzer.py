#!/usr/bin/env python3
"""
Helper script to reload all cancer research tools with the latest fixes
Run this in your notebook or as a standalone script
"""

import sys
import importlib

def reload_all_tools(model_path, device='cuda'):
    """
    Reload all cancer research tools with latest fixes

    Args:
        model_path: Path to your trained model
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (analyzer, designer, neoantigen_designer)
    """
    print("🔄 Reloading all cancer research tools with latest fixes...")

    # Remove cached modules
    modules_to_remove = [
        'genesis_rna.breast_cancer',
        'genesis_rna.model',
        'genesis_rna.tokenization',
        'genesis_rna.config'
    ]

    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
            print(f"   ✓ Removed cached {module}")

    # Re-import
    import torch
    sys.path.insert(0, r'c:\Users\adminidiakhoa\genesi_ai\genesis_rna')

    from genesis_rna.breast_cancer import (
        BreastCancerAnalyzer,
        mRNATherapeuticDesigner,
        NeoantigenDesigner
    )

    # Initialize all tools
    analyzer = BreastCancerAnalyzer(model_path, device=device)
    designer = mRNATherapeuticDesigner(model_path, device=device)
    neoantigen_designer = NeoantigenDesigner(model_path, device=device)

    print(f"✅ All tools reloaded successfully on {device}!")
    print(f"\nAvailable tools:")
    print(f"  • analyzer - BreastCancerAnalyzer")
    print(f"  • designer - mRNATherapeuticDesigner")
    print(f"  • neoantigen_designer - NeoantigenDesigner")

    print(f"\nSupported cancer genes:")
    for gene, desc in analyzer.cancer_genes.items():
        print(f"  • {gene}: {desc}")

    return analyzer, designer, neoantigen_designer

# Legacy function for backward compatibility
def reload_analyzer(model_path, device='cuda'):
    """Legacy function - returns only analyzer"""
    analyzer, _, _ = reload_all_tools(model_path, device)
    return analyzer


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Reload All Cancer Research Tools Helper                         ║
    ╚══════════════════════════════════════════════════════════════════╝

    Usage in your notebook:

    from reload_analyzer import reload_all_tools
    analyzer, designer, neoantigen_designer = reload_all_tools(MODEL_PATH, device='cuda')

    Or for just the analyzer (legacy):
    from reload_analyzer import reload_analyzer
    analyzer = reload_analyzer(MODEL_PATH, device='cuda')

    Then run your analysis!
    """)
