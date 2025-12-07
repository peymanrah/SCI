#!/usr/bin/env python
"""
Ablation Study Runner - Section 7.1-7.2

Implements:
- Systematic ablation of SCI components
- Automated training of each ablation configuration
- Comparison of ablation results
- Generation of ablation comparison tables
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import subprocess
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ablation studies for SCI')
    parser.add_argument('--ablations-config', type=str, default='configs/ablations.yaml',
                       help='Path to ablations config file')
    parser.add_argument('--base-config', type=str, default='configs/sci_full.yaml',
                       help='Path to base config file')
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                       help='Directory to save ablation results')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs (useful for quick testing)')
    parser.add_argument('--wandb-project', type=str, default='sci-ablations',
                       help='Weights & Biases project name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--ablations', type=str, nargs='+', default=None,
                       help='Specific ablations to run (default: all)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only generate comparison table')
    return parser.parse_args()


def load_ablations_config(config_path):
    """Load ablations configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_base_config(config_path):
    """Load base configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_ablation_config(base_config, ablation_spec, output_dir):
    """
    Create ablation config by applying overrides to base config.

    Args:
        base_config: Base configuration dict
        ablation_spec: Ablation specification dict
        output_dir: Directory to save ablation config

    Returns:
        Path to created config file
    """
    import copy

    # Deep copy base config
    ablation_config = copy.deepcopy(base_config)

    # Update experiment name
    ablation_config['experiment']['name'] = ablation_spec['name']

    # Apply overrides
    if 'overrides' in ablation_spec:
        overrides = ablation_spec['overrides']
        _apply_overrides(ablation_config, overrides)

    # Save ablation config
    config_path = output_dir / f"{ablation_spec['name']}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(ablation_config, f, default_flow_style=False)

    return config_path


def _apply_overrides(config, overrides, prefix=''):
    """
    Recursively apply overrides to config.

    Args:
        config: Config dict to modify
        overrides: Override dict
        prefix: Current key prefix for nested access
    """
    for key, value in overrides.items():
        if isinstance(value, dict):
            if key in config and isinstance(config[key], dict):
                _apply_overrides(config[key], value, prefix + key + '.')
            else:
                config[key] = value
        else:
            config[key] = value
            print(f"  Override: {prefix}{key} = {value}")


def run_training(config_path, output_dir, args):
    """
    Run training for an ablation.

    Args:
        config_path: Path to ablation config
        output_dir: Output directory for this ablation
        args: Command line arguments

    Returns:
        dict: Training results
    """
    ablation_name = config_path.stem

    print(f"\n{'='*80}")
    print(f"Training ablation: {ablation_name}")
    print(f"{'='*80}")

    # Build training command
    train_script = Path(__file__).parent.parent / 'train.py'
    cmd = [
        sys.executable,
        str(train_script),
        '--config', str(config_path),
        '--output-dir', str(output_dir),
        '--wandb-project', args.wandb_project,
        '--wandb-run-name', ablation_name,
    ]

    if args.no_wandb:
        cmd.append('--no-wandb')

    # Run training
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Training completed successfully
        return {
            'status': 'success',
            'output': result.stdout,
        }

    except subprocess.CalledProcessError as e:
        print(f"Training failed for {ablation_name}")
        print(f"Error: {e.stderr}")
        return {
            'status': 'failed',
            'error': e.stderr,
        }


def run_evaluation(config_path, checkpoint_path, output_dir):
    """
    Run evaluation for an ablation.

    Args:
        config_path: Path to ablation config
        checkpoint_path: Path to best checkpoint
        output_dir: Output directory for evaluation results

    Returns:
        dict: Evaluation results
    """
    ablation_name = config_path.stem

    print(f"\n{'='*80}")
    print(f"Evaluating ablation: {ablation_name}")
    print(f"{'='*80}")

    # Build evaluation command
    eval_script = Path(__file__).parent.parent / 'evaluate.py'
    cmd = [
        sys.executable,
        str(eval_script),
        '--config', str(config_path),
        '--checkpoint', str(checkpoint_path),
        '--split', 'length',
        '--subset', 'test',
        '--output-dir', str(output_dir),
    ]

    # Run evaluation
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Load evaluation results
        results_path = output_dir / f"{ablation_name}_length_test" / 'results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                eval_results = json.load(f)
            return eval_results
        else:
            print(f"Warning: Results file not found at {results_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {ablation_name}")
        print(f"Error: {e.stderr}")
        return None


def find_best_checkpoint(ablation_output_dir):
    """
    Find best checkpoint for an ablation.

    Args:
        ablation_output_dir: Output directory for ablation training

    Returns:
        Path to best checkpoint, or None if not found
    """
    checkpoint_dir = ablation_output_dir / 'checkpoints'

    if not checkpoint_dir.exists():
        return None

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))

    if not checkpoints:
        return None

    # Return the most recent checkpoint (assumes best is saved last)
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def generate_comparison_table(ablation_results, output_dir):
    """
    Generate comparison table for ablation results.

    Args:
        ablation_results: Dict mapping ablation names to results
        output_dir: Output directory

    Returns:
        Path to comparison table file
    """
    print(f"\n{'='*80}")
    print("Generating Ablation Comparison Table")
    print(f"{'='*80}")

    # Create comparison table
    table_lines = []
    table_lines.append("# Ablation Study Results")
    table_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    table_lines.append("| Ablation | Exact Match | Token Acc | Description |")
    table_lines.append("|----------|-------------|-----------|-------------|")

    # Sort by exact match (descending)
    sorted_results = sorted(
        ablation_results.items(),
        key=lambda x: x[1].get('exact_match', 0) if x[1] else 0,
        reverse=True
    )

    for ablation_name, results in sorted_results:
        if results:
            exact_match = results.get('exact_match', 0) * 100
            token_acc = results.get('token_accuracy', 0) * 100
            description = results.get('description', 'N/A')

            table_lines.append(
                f"| {ablation_name} | {exact_match:.2f}% | {token_acc:.2f}% | {description} |"
            )
        else:
            table_lines.append(
                f"| {ablation_name} | N/A | N/A | Training/Eval failed |"
            )

    # Save table
    table_path = output_dir / 'ablation_comparison.md'
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines))

    # Print table
    print('\n'.join(table_lines))
    print(f"\nComparison table saved to: {table_path}")

    return table_path


def main():
    """Main ablation study runner."""
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configurations
    print(f"Loading ablations config from {args.ablations_config}")
    ablations_config = load_ablations_config(args.ablations_config)

    print(f"Loading base config from {args.base_config}")
    base_config = load_base_config(args.base_config)

    # Get list of ablations to run
    all_ablations = ablations_config['ablations']

    if args.ablations:
        # Filter to specified ablations
        ablations_to_run = [
            a for a in all_ablations if a['name'] in args.ablations
        ]
        if not ablations_to_run:
            print(f"Error: No matching ablations found for: {args.ablations}")
            print(f"Available ablations: {[a['name'] for a in all_ablations]}")
            return
    else:
        ablations_to_run = all_ablations

    print(f"\nRunning {len(ablations_to_run)} ablations:")
    for ablation in ablations_to_run:
        print(f"  - {ablation['name']}: {ablation['description']}")

    # Run ablations
    ablation_results = {}

    for ablation_spec in ablations_to_run:
        ablation_name = ablation_spec['name']

        # Create ablation config
        print(f"\n{'='*80}")
        print(f"Setting up ablation: {ablation_name}")
        print(f"{'='*80}")
        print(f"Description: {ablation_spec['description']}")

        ablation_config_dir = output_dir / 'configs'
        ablation_config_dir.mkdir(exist_ok=True)

        config_path = create_ablation_config(
            base_config,
            ablation_spec,
            ablation_config_dir
        )

        print(f"Ablation config saved to: {config_path}")

        if not args.skip_training:
            # Run training
            ablation_output_dir = output_dir / ablation_name
            training_result = run_training(config_path, ablation_output_dir, args)

            if training_result['status'] == 'success':
                # Find best checkpoint
                best_checkpoint = find_best_checkpoint(ablation_output_dir / ablation_name)

                if best_checkpoint:
                    print(f"Best checkpoint: {best_checkpoint}")

                    # Run evaluation
                    eval_output_dir = output_dir / 'evaluations'
                    eval_output_dir.mkdir(exist_ok=True)

                    eval_results = run_evaluation(
                        config_path,
                        best_checkpoint,
                        eval_output_dir
                    )

                    if eval_results:
                        eval_results['description'] = ablation_spec['description']
                        ablation_results[ablation_name] = eval_results
                    else:
                        print(f"Warning: Evaluation failed for {ablation_name}")
                        ablation_results[ablation_name] = None
                else:
                    print(f"Warning: No checkpoint found for {ablation_name}")
                    ablation_results[ablation_name] = None
            else:
                print(f"Training failed for {ablation_name}")
                ablation_results[ablation_name] = None
        else:
            # Skip training, try to load existing results
            eval_results_path = (
                output_dir / 'evaluations' / f"{ablation_name}_length_test" / 'results.json'
            )

            if eval_results_path.exists():
                with open(eval_results_path, 'r') as f:
                    eval_results = json.load(f)
                eval_results['description'] = ablation_spec['description']
                ablation_results[ablation_name] = eval_results
            else:
                print(f"Warning: No results found for {ablation_name}")
                ablation_results[ablation_name] = None

    # Generate comparison table
    generate_comparison_table(ablation_results, output_dir)

    # Save all results
    all_results_path = output_dir / 'all_ablation_results.json'
    with open(all_results_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Ablation Study Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Comparison table: {output_dir / 'ablation_comparison.md'}")
    print(f"All results: {all_results_path}")


if __name__ == '__main__':
    main()
