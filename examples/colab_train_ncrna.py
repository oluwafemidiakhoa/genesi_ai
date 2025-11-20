"""
Google Colab Training Script for Genesis RNA

This script is designed to be run in Google Colab after preprocessing ncRNA data.

Usage in Colab:
    1. Upload this file to Colab
    2. Run: %cd /content/genesi_ai
    3. Run: !python examples/colab_train_ncrna.py
"""

import os
import sys
from pathlib import Path

# Colab-specific setup
if 'google.colab' in sys.modules:
    print("Running in Google Colab")
    # Assuming genesi_ai is in /content/genesi_ai
    os.chdir('/content/genesi_ai')
else:
    print("Not running in Colab - using current directory")

# Add to path
sys.path.insert(0, 'genesis_rna')

import argparse
import torch
from genesis_rna.train_pretrain import main as train_main


def find_data_file():
    """Find the processed data file"""
    # Common locations for processed data
    possible_paths = [
        '../data/human_ncrna/sequences.pkl',
        'data/human_ncrna/sequences.pkl',
        '/content/data/human_ncrna/sequences.pkl',
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Found data at: {path}")
            return path

    print("❌ Could not find processed data file!")
    print("\nSearched in:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nPlease ensure you've run the preprocessing script first.")
    return None


def main():
    print("="*70)
    print("GENESIS RNA TRAINING - GOOGLE COLAB")
    print("="*70)

    # Find data
    data_path = find_data_file()
    if not data_path:
        return 1

    # Set up arguments for training
    output_dir = 'checkpoints/pretrained/base'

    print(f"\nConfiguration:")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Model: base")
    print(f"  Batch size: 16")
    print(f"  Epochs: 10")
    print(f"  Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    if not torch.cuda.is_available():
        print("\n⚠️ Warning: No GPU detected. Training will be slow.")
        print("   In Colab: Runtime > Change runtime type > GPU")

    print("="*70 + "\n")

    # Prepare arguments
    sys.argv = [
        'train_pretrain',
        '--data_path', data_path,
        '--output_dir', output_dir,
        '--model_size', 'base',
        '--batch_size', '16',
        '--num_epochs', '10',
        '--learning_rate', '1e-4',
        '--use_ast',
        '--ast_target_activation', '0.4',
    ]

    # Run training
    try:
        print("Starting training...\n")
        train_main()

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nModel saved to: {output_dir}/best_model.pt")

        # Check if file exists
        model_path = Path(output_dir) / 'best_model.pt'
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"Model file size: {size_mb:.2f} MB")
            print("\n✅ Model is ready to use!")
        else:
            print("\n⚠️ Warning: Model file not found at expected location")

        print("\nNext steps:")
        print("  1. Test the model with breast cancer analysis:")
        print("     !python examples/breast_cancer_analysis.py")
        print("\n  2. Or use the model programmatically:")
        print("     from genesis_rna.breast_cancer import BreastCancerAnalyzer")
        print(f"     analyzer = BreastCancerAnalyzer('{output_dir}/best_model.pt')")
        print("="*70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
