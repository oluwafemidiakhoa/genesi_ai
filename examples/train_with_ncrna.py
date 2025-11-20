#!/usr/bin/env python3
"""
Train Genesis RNA with Real ncRNA Data

This script trains the Genesis RNA model using processed ncRNA sequences
from Ensembl (Homo_sapiens.GRCh38.ncrna.fa).

Usage:
    # Default: Use data from ../data/human_ncrna/sequences.pkl
    python examples/train_with_ncrna.py

    # Custom data path
    python examples/train_with_ncrna.py --data_path /path/to/sequences.pkl

    # Limit samples for faster testing
    python examples/train_with_ncrna.py --max_samples 10000 --num_epochs 3
"""

import sys
import os
from pathlib import Path

# Add genesis_rna to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'genesis_rna'))

import argparse
import torch
from genesis_rna.train_pretrain import main as train_main


def check_data_exists(data_path):
    """Check if data file exists"""
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("\nPlease run the preprocessing script first:")
        print("  python genesis_rna/scripts/preprocess_rna.py \\")
        print("      --input /path/to/Homo_sapiens.GRCh38.ncrna.fa \\")
        print("      --output data/human_ncrna \\")
        print("      --min_len 50 \\")
        print("      --max_len 512 \\")
        print("      --format pickle")
        return False
    return True


def print_training_info(args):
    """Print training information"""
    print("\n" + "="*70)
    print("GENESIS RNA TRAINING WITH REAL ncRNA DATA")
    print("="*70)
    print(f"\nData path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model size: {args.model_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"AST enabled: {args.use_ast}")
    if args.use_ast:
        print(f"AST target activation: {args.ast_target_activation}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Genesis RNA with real ncRNA data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/human_ncrna/sequences.pkl',
        help='Path to processed ncRNA data (pickle file)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (None = all)'
    )

    # Model arguments
    parser.add_argument(
        '--model_size',
        type=str,
        default='base',
        choices=['small', 'base', 'large'],
        help='Model size'
    )

    # Training arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/pretrained/base',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    # AST arguments
    parser.add_argument(
        '--use_ast',
        action='store_true',
        default=True,
        help='Enable Adaptive Sparse Training'
    )
    parser.add_argument(
        '--no_ast',
        action='store_true',
        help='Disable Adaptive Sparse Training'
    )
    parser.add_argument(
        '--ast_target_activation',
        type=float,
        default=0.4,
        help='AST target activation rate (fraction of samples to train on)'
    )

    args = parser.parse_args()

    # Handle no_ast flag
    if args.no_ast:
        args.use_ast = False

    # Check if data exists
    if not check_data_exists(args.data_path):
        return 1

    # Print training info
    print_training_info(args)

    # Prepare sys.argv for train_pretrain.py
    sys.argv = [
        'train_pretrain',
        '--data_path', str(args.data_path),
        '--output_dir', str(args.output_dir),
        '--model_size', args.model_size,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--ast_target_activation', str(args.ast_target_activation),
    ]

    if args.max_samples:
        sys.argv.extend(['--max_samples', str(args.max_samples)])

    if args.use_ast:
        sys.argv.append('--use_ast')

    # Run training
    try:
        train_main()
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nModel saved to: {args.output_dir}/best_model.pt")
        print("\nYou can now use this model for:")
        print("  • Breast cancer variant analysis")
        print("  • RNA structure prediction")
        print("  • RNA design and optimization")
        print("  • Fine-tuning on specific tasks")
        print("\nExample usage:")
        print("  python examples/breast_cancer_analysis.py")
        print("="*70 + "\n")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
