#!/usr/bin/env python3
"""
Visualize Training Metrics for Genesis RNA

Reads training_metrics.csv and creates visualizations of:
- Loss curves (train/val)
- Learning rate schedule
- Pair prediction metrics (precision, recall, F1)
- Structure and MLM accuracy
- AST activation rate
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_losses(df, output_dir):
    """Plot training and validation losses"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Losses', fontsize=16)

    # Total loss
    ax = axes[0, 0]
    train_df = df[df['phase'] == 'train']
    val_df = df[df['phase'] == 'val']
    ax.plot(train_df['epoch'], train_df['loss'], label='Train', marker='o')
    ax.plot(val_df['epoch'], val_df['loss'], label='Val', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MLM loss
    ax = axes[0, 1]
    ax.plot(train_df['epoch'], train_df['mlm_loss'], label='Train', marker='o')
    ax.plot(val_df['epoch'], val_df['mlm_loss'], label='Val', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('MLM Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Structure loss
    ax = axes[1, 0]
    ax.plot(train_df['epoch'], train_df['structure_loss'], label='Train', marker='o')
    ax.plot(val_df['epoch'], val_df['structure_loss'], label='Val', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Structure Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pair loss
    ax = axes[1, 1]
    ax.plot(train_df['epoch'], train_df['pair_loss'], label='Train', marker='o')
    ax.plot(val_df['epoch'], val_df['pair_loss'], label='Val', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Pair Loss (Focal)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'losses.png', dpi=150, bbox_inches='tight')
    print(f"Saved losses plot to {output_dir / 'losses.png'}")
    plt.close()


def plot_accuracies(df, output_dir):
    """Plot accuracy metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Accuracy Metrics', fontsize=16)

    val_df = df[df['phase'] == 'val']

    # MLM and Structure accuracy
    ax = axes[0]
    ax.plot(val_df['epoch'], val_df['mlm_accuracy'] * 100, label='MLM', marker='o')
    ax.plot(val_df['epoch'], val_df['structure_accuracy'] * 100, label='Structure', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('MLM & Structure Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pair metrics (precision, recall, F1)
    ax = axes[1]
    ax.plot(val_df['epoch'], val_df['pair_precision'] * 100, label='Precision', marker='o')
    ax.plot(val_df['epoch'], val_df['pair_recall'] * 100, label='Recall', marker='s')
    ax.plot(val_df['epoch'], val_df['pair_f1'] * 100, label='F1', marker='^')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score (%)')
    ax.set_title('Pair Prediction Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracies.png', dpi=150, bbox_inches='tight')
    print(f"Saved accuracies plot to {output_dir / 'accuracies.png'}")
    plt.close()


def plot_lr_and_activation(df, output_dir):
    """Plot learning rate schedule and AST activation rate"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Learning Rate & AST Activation', fontsize=16)

    train_df = df[df['phase'] == 'train']

    # Learning rate
    ax = axes[0]
    ax.plot(train_df['epoch'], train_df['learning_rate'], marker='o', color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Activation rate
    ax = axes[1]
    ax.plot(train_df['epoch'], train_df['activation_rate'] * 100, marker='o', color='green')
    ax.axhline(y=40, color='r', linestyle='--', label='Target (40%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Activation Rate (%)')
    ax.set_title('AST Sample Activation Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'lr_activation.png', dpi=150, bbox_inches='tight')
    print(f"Saved LR & activation plot to {output_dir / 'lr_activation.png'}")
    plt.close()


def plot_summary(df, output_dir):
    """Create summary plot showing key metrics"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    train_df = df[df['phase'] == 'train']
    val_df = df[df['phase'] == 'val']

    # Total Loss
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train_df['epoch'], train_df['loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(val_df['epoch'], val_df['loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress: Total Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Component losses
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(val_df['epoch'], val_df['mlm_loss'], marker='o', color='blue')
    ax2.set_title('MLM Loss')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(val_df['epoch'], val_df['structure_loss'], marker='o', color='orange')
    ax3.set_title('Structure Loss')
    ax3.set_xlabel('Epoch')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(val_df['epoch'], val_df['pair_loss'], marker='o', color='green')
    ax4.set_title('Pair Loss (Focal)')
    ax4.set_xlabel('Epoch')
    ax4.grid(True, alpha=0.3)

    # Accuracies
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(val_df['epoch'], val_df['mlm_accuracy'] * 100, marker='o', color='blue')
    ax5.set_title('MLM Accuracy')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy (%)')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(val_df['epoch'], val_df['structure_accuracy'] * 100, marker='o', color='orange')
    ax6.set_title('Structure Accuracy')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Accuracy (%)')
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(val_df['epoch'], val_df['pair_f1'] * 100, marker='o', color='green', label='F1')
    ax7.plot(val_df['epoch'], val_df['pair_precision'] * 100, marker='s', color='lightgreen', alpha=0.7, label='Precision')
    ax7.plot(val_df['epoch'], val_df['pair_recall'] * 100, marker='^', color='darkgreen', alpha=0.7, label='Recall')
    ax7.set_title('Pair Metrics')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Score (%)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    fig.suptitle('Genesis RNA Training Summary', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved summary plot to {output_dir / 'summary.png'}")
    plt.close()


def print_stats(df):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("TRAINING STATISTICS SUMMARY")
    print("="*70)

    val_df = df[df['phase'] == 'val']

    if len(val_df) > 0:
        print(f"\nFinal Epoch Metrics (Epoch {val_df['epoch'].max()}):")
        final = val_df.iloc[-1]
        print(f"  Total Loss:          {final['loss']:.4f}")
        print(f"  MLM Accuracy:        {final['mlm_accuracy']*100:.2f}%")
        print(f"  Structure Accuracy:  {final['structure_accuracy']*100:.2f}%")
        print(f"  Pair F1:             {final['pair_f1']*100:.2f}%")
        print(f"  Pair Precision:      {final['pair_precision']*100:.2f}%")
        print(f"  Pair Recall:         {final['pair_recall']*100:.2f}%")

        print(f"\nBest Validation Metrics:")
        best_idx = val_df['loss'].idxmin()
        best = val_df.loc[best_idx]
        print(f"  Best Loss: {best['loss']:.4f} (Epoch {best['epoch']})")

        best_f1_idx = val_df['pair_f1'].idxmax()
        best_f1 = val_df.loc[best_f1_idx]
        print(f"  Best Pair F1: {best_f1['pair_f1']*100:.2f}% (Epoch {best_f1['epoch']})")

        print(f"\nImprovement:")
        first = val_df.iloc[0]
        print(f"  Loss:          {first['loss']:.4f} → {final['loss']:.4f} ({(final['loss']/first['loss']-1)*100:+.1f}%)")
        print(f"  MLM Accuracy:  {first['mlm_accuracy']*100:.2f}% → {final['mlm_accuracy']*100:.2f}% ({(final['mlm_accuracy']/first['mlm_accuracy']-1)*100:+.1f}%)")
        print(f"  Structure Acc: {first['structure_accuracy']*100:.2f}% → {final['structure_accuracy']*100:.2f}% ({(final['structure_accuracy']/first['structure_accuracy']-1)*100:+.1f}%)")
        if final['pair_f1'] > 0:
            print(f"  Pair F1:       {first['pair_f1']*100:.2f}% → {final['pair_f1']*100:.2f}% ({(final['pair_f1']/first['pair_f1']-1)*100:+.1f}%)")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize Genesis RNA training metrics')
    parser.add_argument('--metrics_file', type=str, default='training_metrics.csv',
                       help='Path to training metrics CSV file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save plots')

    args = parser.parse_args()

    metrics_file = Path(args.metrics_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        return

    # Load metrics
    print(f"Loading metrics from {metrics_file}...")
    df = pd.read_csv(metrics_file)

    # Print stats
    print_stats(df)

    # Create plots
    print("\nGenerating visualizations...")
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    plot_losses(df, output_dir)
    plot_accuracies(df, output_dir)
    plot_lr_and_activation(df, output_dir)
    plot_summary(df, output_dir)

    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("   - losses.png: Training and validation losses")
    print("   - accuracies.png: Accuracy metrics")
    print("   - lr_activation.png: Learning rate schedule and AST activation")
    print("   - summary.png: Comprehensive training summary")


if __name__ == '__main__':
    main()
