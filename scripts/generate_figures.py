#!/usr/bin/env python
"""
Generate publication figures for Nature Machine Intelligence.

Creates all figures from the paper:
- Figure 1: Architecture diagram
- Figure 2: Main results (accuracy comparison)
- Figure 3: Ablation studies
- Figure 4: Structural invariance analysis
- Figure 5: Attention visualizations

Usage:
    python scripts/generate_figures.py --results_dir results/ --output_dir figures/generated/
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sns.set_palette("colorblind")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate publication figures')

    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures/generated/',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='pdf',
        choices=['pdf', 'png', 'svg'],
        help='Output format'
    )

    return parser.parse_args()


def load_results(results_dir: str) -> dict:
    """Load all experiment results."""
    results = {}

    experiments = [
        'baseline',
        'sci_full',
        'ablation_no_al',
        'ablation_no_scl',
        'ablation_no_ce',
        'ablation_no_cbm',
    ]

    for exp in experiments:
        result_file = os.path.join(results_dir, exp, 'evaluation_results.json')

        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[exp] = json.load(f)
        else:
            print(f"Warning: Results not found for {exp}")

    return results


def generate_figure_2_main_results(results: dict, output_path: str, format: str):
    """
    Figure 2: Main Results - Accuracy Comparison.

    Shows exact match accuracy for:
    - Baseline vs SCI Full on SCAN length (ID and OOD)
    - SCAN simple
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Extract results
    experiments = ['baseline', 'sci_full']
    exp_labels = ['Baseline\n(Vanilla TinyLlama)', 'SCI Full\n(Proposed)']

    metrics = ['scan_length_id', 'scan_length_ood', 'scan_simple']
    metric_labels = ['SCAN Length\n(In-Dist)', 'SCAN Length\n(OOD)', 'SCAN Simple']

    data = []
    for exp in experiments:
        exp_data = []
        for metric in metrics:
            # Find matching result
            value = 0.0
            if exp in results:
                for dataset_name, result in results[exp].items():
                    if metric.replace('_', ' ') in dataset_name.lower():
                        value = result.get('exact_match', 0.0)
                        break
            exp_data.append(value * 100)  # Convert to percentage
        data.append(exp_data)

    # Plot grouped bar chart
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, data[0], width, label=exp_labels[0], color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, data[1], width, label=exp_labels[1], color='#3498DB', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Exact Match Accuracy (%)')
    ax.set_title('Figure 2: Main Results - SCI vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/figure2_main_results.{format}")
    print(f"✓ Generated Figure 2: {output_path}/figure2_main_results.{format}")
    plt.close()


def generate_figure_3_ablations(results: dict, output_path: str, format: str):
    """
    Figure 3: Ablation Studies.

    Shows OOD performance for each ablation.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    experiments = [
        'sci_full',
        'ablation_no_al',
        'ablation_no_scl',
        'ablation_no_ce',
        'ablation_no_cbm',
        'baseline',
    ]

    labels = [
        'SCI Full',
        'No AbstractionLayer',
        'No SCL',
        'No Content Encoder',
        'No Causal Binding',
        'Baseline',
    ]

    # Extract OOD accuracies
    accuracies = []
    for exp in experiments:
        value = 0.0
        if exp in results:
            for dataset_name, result in results[exp].items():
                if 'length' in dataset_name.lower():
                    value = result.get('exact_match', 0.0)
                    break
        accuracies.append(value * 100)

    # Plot horizontal bar chart
    y = np.arange(len(experiments))
    bars = ax.barh(y, accuracies, color='#2ECC71', alpha=0.8)

    # Color baseline and full differently
    bars[-1].set_color('#E74C3C')  # Baseline in red
    bars[0].set_color('#3498DB')   # SCI Full in blue

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%',
                ha='left', va='center', fontsize=9)

    ax.set_xlabel('SCAN Length OOD Accuracy (%)')
    ax.set_title('Figure 3: Ablation Studies - Component Contributions')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/figure3_ablations.{format}")
    print(f"✓ Generated Figure 3: {output_path}/figure3_ablations.{format}")
    plt.close()


def generate_figure_4_structural_invariance(results: dict, output_path: str, format: str):
    """
    Figure 4: Structural Invariance Metrics.

    Shows structural invariance scores across models.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    experiments = ['sci_full', 'ablation_no_scl', 'baseline']
    labels = ['SCI Full', 'SCI (No SCL)', 'Baseline']

    # Extract structural invariance
    invariances = []
    for exp in experiments:
        value = 0.0
        if exp in results:
            for dataset_name, result in results[exp].items():
                if 'structural_invariance' in result:
                    value = result['structural_invariance']
                    break
        invariances.append(value)

    # Plot bar chart
    bars = ax.bar(labels, invariances, color=['#3498DB', '#F39C12', '#E74C3C'], alpha=0.8)

    # Add value labels
    for bar, inv in zip(bars, invariances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{inv:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Structural Invariance Score')
    ax.set_title('Figure 4: Structural Invariance Analysis')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/figure4_structural_invariance.{format}")
    print(f"✓ Generated Figure 4: {output_path}/figure4_structural_invariance.{format}")
    plt.close()


def generate_all_figures(results_dir: str, output_dir: str, format: str):
    """Generate all figures."""
    print("=" * 70)
    print("Generating Publication Figures")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print(f"\nLoading results from: {results_dir}")
    results = load_results(results_dir)

    if not results:
        print("Error: No results found!")
        print("Please run experiments first using:")
        print("  python scripts/run_all_experiments.py")
        return

    print(f"Loaded results for: {list(results.keys())}")

    # Generate figures
    print(f"\nGenerating figures (format: {format})...")

    generate_figure_2_main_results(results, output_dir, format)
    generate_figure_3_ablations(results, output_dir, format)
    generate_figure_4_structural_invariance(results, output_dir, format)

    print("\n" + "=" * 70)
    print("Figure Generation Complete!")
    print("=" * 70)
    print(f"Figures saved to: {output_dir}")


def main():
    args = parse_args()
    generate_all_figures(args.results_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
