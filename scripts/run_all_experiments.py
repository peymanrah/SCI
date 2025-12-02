#!/usr/bin/env python
"""
Run all SCI experiments: Full SCI + Baseline + Ablations.

This script runs the complete experimental suite for the Nature MI paper:
1. Baseline (vanilla TinyLlama)
2. SCI Full (all components)
3. 4 Ablations (removing one component each)

Usage:
    python scripts/run_all_experiments.py --max_epochs 50
    python scripts/run_all_experiments.py --quick_test  # 2 epochs for testing
"""

import argparse
import os
import sys
import subprocess
from typing import List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


EXPERIMENTS = [
    {
        'name': 'Baseline',
        'config': 'configs/baseline.yaml',
        'description': 'Vanilla TinyLlama (no SCI components)',
    },
    {
        'name': 'SCI Full',
        'config': 'configs/sci_full.yaml',
        'description': 'Complete SCI architecture',
    },
    {
        'name': 'Ablation: No AbstractionLayer',
        'config': 'configs/ablations/no_abstraction_layer.yaml',
        'description': 'SCI without AbstractionLayer',
    },
    {
        'name': 'Ablation: No SCL',
        'config': 'configs/ablations/no_scl.yaml',
        'description': 'SCI without contrastive learning',
    },
    {
        'name': 'Ablation: No Content Encoder',
        'config': 'configs/ablations/no_content_encoder.yaml',
        'description': 'SCI without separate content encoding',
    },
    {
        'name': 'Ablation: No Causal Binding',
        'config': 'configs/ablations/no_causal_binding.yaml',
        'description': 'SCI without causal binding mechanism',
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run all SCI experiments')

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help='Maximum epochs for each experiment'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Quick test mode (2 epochs, small batch)'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        default=None,
        help='Specific experiments to run (e.g., "Baseline" "SCI Full")'
    )
    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='Skip evaluation after training'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable WandB logging'
    )

    return parser.parse_args()


def run_training(config_path: str, max_epochs: int, no_wandb: bool = False) -> bool:
    """
    Run training for a single experiment.

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable,
        'scripts/train_sci.py',
        '--config', config_path,
        '--max_epochs', str(max_epochs),
    ]

    if no_wandb:
        cmd.append('--no_wandb')

    print(f"\nRunning command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False


def run_evaluation(checkpoint_dir: str, config_path: str) -> bool:
    """
    Run evaluation for a trained model.

    Returns:
        True if successful, False otherwise
    """
    # Find best checkpoint
    best_checkpoint = os.path.join(checkpoint_dir, 'best')

    if not os.path.exists(best_checkpoint):
        # Try final checkpoint
        best_checkpoint = os.path.join(checkpoint_dir, 'final')

    if not os.path.exists(best_checkpoint):
        print(f"Warning: No checkpoint found at {checkpoint_dir}")
        return False

    cmd = [
        sys.executable,
        'scripts/evaluate.py',
        '--checkpoint', best_checkpoint,
        '--config', config_path,
        '--splits', 'length', 'simple',
        '--compute_structural_invariance',
    ]

    print(f"\nRunning command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        return False


def main():
    args = parse_args()

    print("=" * 80)
    print("SCI Complete Experimental Suite")
    print("=" * 80)

    # Filter experiments if specified
    if args.experiments:
        experiments_to_run = [
            exp for exp in EXPERIMENTS
            if exp['name'] in args.experiments
        ]
    else:
        experiments_to_run = EXPERIMENTS

    # Quick test mode
    if args.quick_test:
        max_epochs = 2
        print("\n⚠️  QUICK TEST MODE: 2 epochs only")
    else:
        max_epochs = args.max_epochs

    print(f"\nRunning {len(experiments_to_run)} experiments:")
    for exp in experiments_to_run:
        print(f"  - {exp['name']}")

    print(f"\nSettings:")
    print(f"  Max epochs: {max_epochs}")
    print(f"  WandB: {'DISABLED' if args.no_wandb else 'ENABLED'}")
    print(f"  Evaluation: {'DISABLED' if args.skip_evaluation else 'ENABLED'}")

    # Confirm
    response = input("\nProceed? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Run experiments
    results = []

    for i, exp in enumerate(experiments_to_run):
        print("\n" + "=" * 80)
        print(f"Experiment {i+1}/{len(experiments_to_run)}: {exp['name']}")
        print(f"Description: {exp['description']}")
        print("=" * 80)

        # Train
        success = run_training(
            config_path=exp['config'],
            max_epochs=max_epochs,
            no_wandb=args.no_wandb,
        )

        if not success:
            print(f"\n✗ Training failed for {exp['name']}")
            results.append({'name': exp['name'], 'status': 'FAILED'})
            continue

        print(f"\n✓ Training completed for {exp['name']}")

        # Evaluate
        if not args.skip_evaluation:
            # Extract checkpoint dir from config
            from sci.config.config_loader import load_config
            config = load_config(exp['config'])
            checkpoint_dir = config.training.checkpoint_dir

            eval_success = run_evaluation(
                checkpoint_dir=checkpoint_dir,
                config_path=exp['config'],
            )

            if eval_success:
                print(f"✓ Evaluation completed for {exp['name']}")
                results.append({'name': exp['name'], 'status': 'SUCCESS'})
            else:
                print(f"✗ Evaluation failed for {exp['name']}")
                results.append({'name': exp['name'], 'status': 'EVAL_FAILED'})
        else:
            results.append({'name': exp['name'], 'status': 'TRAINED'})

    # Summary
    print("\n" + "=" * 80)
    print("Experimental Suite Complete")
    print("=" * 80)

    print("\nResults:")
    for result in results:
        status_icon = {
            'SUCCESS': '✓',
            'TRAINED': '✓',
            'EVAL_FAILED': '⚠',
            'FAILED': '✗'
        }[result['status']]

        print(f"  {status_icon} {result['name']}: {result['status']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
