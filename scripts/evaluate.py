#!/usr/bin/env python
"""
Evaluate SCI Model on SCAN Benchmark.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/sci_full/best --config configs/sci_full.yaml
    python scripts/evaluate.py --checkpoint checkpoints/baseline/best --splits length simple
"""

import argparse
import os
import sys
import torch
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sci.config.config_loader import load_config
from sci.models.sci_model import SCIModel
from sci.evaluation.evaluator import SCIEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SCI model on SCAN')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (if not in checkpoint dir)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=None,
        help='SCAN splits to evaluate (e.g., length simple template)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--compute_structural_invariance',
        action='store_true',
        help='Compute structural invariance metric (slower)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("SCI Evaluation Script")
    print("=" * 70)

    # Load config
    if args.config:
        config_path = args.config
    else:
        # Try to find config in checkpoint directory
        config_path = os.path.join(args.checkpoint, 'config.json')
        if not os.path.exists(config_path):
            # Try YAML in parent directory
            parent_dir = os.path.dirname(args.checkpoint)
            config_name = os.path.basename(args.checkpoint).replace('checkpoints/', '')
            config_path = f"configs/{config_name}.yaml"

    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    # Override evaluation settings
    config.evaluation.batch_size = args.batch_size

    if args.compute_structural_invariance:
        config.evaluation.compute_structural_invariance = True

    # Set evaluation datasets
    if args.splits:
        config.evaluation.datasets = [
            {'name': 'scan', 'split': split, 'subset': 'test'}
            for split in args.splits
        ]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation splits: {[d['split'] for d in config.evaluation.datasets]}")

    # Load model
    print("\nLoading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SCIModel.from_pretrained(args.checkpoint, config=config)
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")

    # Create evaluator
    evaluator = SCIEvaluator(config)

    # Run evaluation
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)

    results = evaluator.evaluate(model)

    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Exact Match: {metrics['exact_match']:.2%} ({metrics['num_exact_matches']}/{metrics['num_examples']})")
        print(f"  Token Accuracy: {metrics['token_accuracy']:.2%}")

        if 'structural_invariance' in metrics:
            print(f"  Structural Invariance: {metrics['structural_invariance']:.3f}")

    # Save results to JSON
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join('results', os.path.basename(args.checkpoint))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'evaluation_results.json')

    print(f"\nSaving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved")

    # Compare with expected results if available
    if config.expected_results is not None:
        print("\n" + "=" * 70)
        print("Comparison with Expected Results")
        print("=" * 70)

        for key, expected_value in config.expected_results.__dict__.items():
            if expected_value == 0.0:
                continue  # Skip unset expected values
            # Try to find matching result
            for dataset_name, metrics in results.items():
                if key in dataset_name.lower():
                    actual_value = metrics.get('exact_match', 0.0)
                    diff = actual_value - expected_value
                    status = "✓" if abs(diff) < 0.05 else "✗"

                    print(f"{status} {key}:")
                    print(f"    Expected: {expected_value:.2%}")
                    print(f"    Actual: {actual_value:.2%}")
                    print(f"    Difference: {diff:+.2%}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
