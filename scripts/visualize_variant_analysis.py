#!/usr/bin/env python3
"""
Visualization Tool for Variant Analysis Results

Creates comprehensive visualizations of Genesis RNA variant predictions:
- Pathogenicity score distributions
- Concordance analysis
- Gene-specific performance
- Clinical interpretation breakdown
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(csv_file: str) -> pd.DataFrame:
    """Load variant analysis results from CSV"""
    return pd.read_csv(csv_file)


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualization suite"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Pathogenicity Score Distribution by Known Significance
    ax1 = fig.add_subplot(gs[0, :2])
    for sig in df['known_significance'].unique():
        subset = df[df['known_significance'] == sig]
        ax1.hist(subset['predicted_score'], alpha=0.6, label=sig, bins=20)
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    ax1.set_xlabel('Pathogenicity Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Pathogenicity Scores by Known Clinical Significance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Concordance Pie Chart
    ax2 = fig.add_subplot(gs[0, 2])
    concordance_counts = df['concordance'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for match, red for mismatch
    labels = ['Concordant', 'Discordant']
    ax2.pie(
        concordance_counts.values,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax2.set_title('ClinVar Concordance')

    # 3. Scatter Plot: Predicted Score vs Known Significance
    ax3 = fig.add_subplot(gs[1, :2])
    significance_map = {
        'Pathogenic': 1.0,
        'Likely Pathogenic': 0.75,
        'Uncertain': 0.5,
        'Likely Benign': 0.25,
        'Benign': 0.0
    }
    df['known_numeric'] = df['known_significance'].map(significance_map)

    # Color by concordance
    colors_scatter = df['concordance'].map({True: '#2ecc71', False: '#e74c3c'})

    ax3.scatter(
        df['known_numeric'],
        df['predicted_score'],
        c=colors_scatter,
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect prediction')
    ax3.set_xlabel('Known Clinical Significance (numeric)')
    ax3.set_ylabel('Predicted Pathogenicity Score')
    ax3.set_title('Predicted vs Known Significance (Green=Match, Red=Mismatch)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Confidence by Gene
    ax4 = fig.add_subplot(gs[1, 2])
    gene_confidence = df.groupby('gene')['confidence'].mean()
    gene_confidence.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_xlabel('Gene')
    ax4.set_ylabel('Average Confidence')
    ax4.set_title('Average Prediction Confidence by Gene')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Variant Performance Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')

    # Create table data
    table_data = []
    for _, row in df.iterrows():
        status = "✅" if row['concordance'] else "❌"
        table_data.append([
            row['variant_id'][:20],  # Truncate long IDs
            row['gene'],
            row['known_significance'][:10],
            f"{row['predicted_score']:.3f}",
            row['predicted_interpretation'][:10],
            f"{row['confidence']:.3f}",
            status
        ])

    table = ax5.table(
        cellText=table_data,
        colLabels=['Variant ID', 'Gene', 'Known', 'Score', 'Predicted', 'Conf.', 'Match'],
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.1, 0.15, 0.1, 0.15, 0.1, 0.08]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(weight='bold', color='white')
        elif table_data[i-1][-1] == "✅":
            cell.set_facecolor('#d5f4e6')  # Light green for matches
        else:
            cell.set_facecolor('#fadbd8')  # Light red for mismatches

    fig.suptitle('Genesis RNA - Breast Cancer Variant Analysis Report',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_file = output_path / 'variant_analysis_report.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive report to {output_file}")

    # Create additional detailed plots

    # Gene-specific performance
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Performance by gene
    gene_perf = df.groupby('gene').agg({
        'concordance': 'mean',
        'predicted_score': 'mean',
        'confidence': 'mean'
    }).reset_index()

    axes[0].bar(gene_perf['gene'], gene_perf['concordance'], color='steelblue')
    axes[0].set_ylabel('Concordance Rate')
    axes[0].set_xlabel('Gene')
    axes[0].set_title('Prediction Concordance by Gene')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(gene_perf['concordance']):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

    # Mutation type performance
    mut_perf = df.groupby('mutation_type').agg({
        'concordance': 'mean',
        'predicted_score': 'mean'
    }).reset_index()

    axes[1].bar(mut_perf['mutation_type'], mut_perf['concordance'], color='coral')
    axes[1].set_ylabel('Concordance Rate')
    axes[1].set_xlabel('Mutation Type')
    axes[1].set_title('Prediction Concordance by Mutation Type')
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mut_perf['concordance']):
        axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

    plt.tight_layout()
    output_file2 = output_path / 'gene_performance.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✅ Saved gene performance plot to {output_file2}")

    # Create confusion matrix-style plot
    fig3, ax = plt.subplots(figsize=(8, 6))

    # Create 2x2 matrix
    df['known_binary'] = df['known_significance'].isin(['Pathogenic', 'Likely Pathogenic'])
    df['pred_binary'] = df['predicted_interpretation'].isin(['Pathogenic', 'Likely Pathogenic'])

    tp = ((df['known_binary'] == True) & (df['pred_binary'] == True)).sum()
    tn = ((df['known_binary'] == False) & (df['pred_binary'] == False)).sum()
    fp = ((df['known_binary'] == False) & (df['pred_binary'] == True)).sum()
    fn = ((df['known_binary'] == True) & (df['pred_binary'] == False)).sum()

    confusion = np.array([[tn, fp], [fn, tp]])

    im = ax.imshow(confusion, cmap='RdYlGn', alpha=0.8)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, confusion[i, j],
                          ha="center", va="center", color="black", fontsize=20, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Benign', 'Predicted Pathogenic'])
    ax.set_yticklabels(['Known Benign', 'Known Pathogenic'])
    ax.set_xlabel('Prediction', fontsize=12)
    ax.set_ylabel('Ground Truth (ClinVar)', fontsize=12)
    ax.set_title('Confusion Matrix - Variant Classification', fontsize=14, fontweight='bold')

    # Add labels for each quadrant
    ax.text(0, -0.4, f'TN={tn}', ha='center', fontsize=10, style='italic')
    ax.text(1, -0.4, f'FP={fp}', ha='center', fontsize=10, style='italic')
    ax.text(0, 1.4, f'FN={fn}', ha='center', fontsize=10, style='italic')
    ax.text(1, 1.4, f'TP={tp}', ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    output_file3 = output_path / 'confusion_matrix.png'
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix to {output_file3}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Visualize variant analysis results')
    parser.add_argument('--results', required=True, help='Path to results CSV file')
    parser.add_argument('--output', default='plots', help='Output directory for plots')

    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"❌ Results file not found: {args.results}")
        sys.exit(1)

    print(f"📊 Loading results from {args.results}...")
    df = load_results(args.results)
    print(f"✅ Loaded {len(df)} variant results")

    print(f"\n🎨 Creating visualizations...")
    create_visualizations(df, args.output)

    print(f"\n✅ All visualizations saved to {args.output}/")
    print(f"   • variant_analysis_report.png - Comprehensive overview")
    print(f"   • gene_performance.png - Gene-specific metrics")
    print(f"   • confusion_matrix.png - Classification matrix")


if __name__ == '__main__':
    main()
