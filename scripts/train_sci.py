#!/usr/bin/env python
"""
Train SCI Model on SCAN Benchmark.

Usage:
    python scripts/train_sci.py --config configs/sci_full.yaml
    python scripts/train_sci.py --config configs/ablations/no_scl.yaml --output_dir checkpoints/ablation_no_scl
"""

import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sci.config.config_loader import load_config
from sci.training.trainer import SCITrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train SCI model on SCAN')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/sci_full.yaml)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Maximum epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--force_regenerate_pairs',
        action='store_true',
        help='Force regeneration of structural pairs cache'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("SCI Training Script")
    print("=" * 70)

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.output_dir:
        config.training.checkpoint_dir = args.output_dir
        print(f"  Output directory: {args.output_dir}")

    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
        print(f"  Max epochs: {args.max_epochs}")

    if args.batch_size:
        config.training.batch_size = args.batch_size
        print(f"  Batch size: {args.batch_size}")

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
        print(f"  Learning rate: {args.learning_rate}")

    if args.no_wandb:
        config.logging.use_wandb = False
        print("  WandB logging: DISABLED")

    if args.seed:
        config.seed = args.seed

    if args.force_regenerate_pairs:
        config.training.force_regenerate_pairs = True
        print("  Force regenerate pairs: ENABLED")

    # Set random seed
    seed = getattr(config, 'seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"\nRandom seed: {seed}")

    # Print configuration summary
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Model: {config.model.base_model}")
    print(f"Dataset: {config.training.dataset}")
    print(f"Split: {config.training.split}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Max epochs: {config.training.max_epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"SCL weight: {config.loss.scl_weight} (warmup: {config.loss.scl_warmup_epochs} epochs)")

    print("\nSCI Components:")
    print(f"  Structural Encoder: {'ENABLED' if config.model.structural_encoder.enabled else 'DISABLED'}")
    print(f"  Content Encoder: {'ENABLED' if config.model.content_encoder.enabled else 'DISABLED'}")
    print(f"  Causal Binding: {'ENABLED' if config.model.causal_binding.enabled else 'DISABLED'}")

    if config.model.structural_encoder.enabled:
        print(f"  - AbstractionLayer injection: {config.model.structural_encoder.abstraction_layer.injection_layers}")
        print(f"  - Structural slots: {config.model.structural_encoder.num_slots}")

    if config.model.causal_binding.enabled:
        print(f"  - CBM injection layers: {config.model.causal_binding.injection_layers}")

    print(f"\nCheckpoint directory: {config.training.checkpoint_dir}")
    print("=" * 70)

    # Confirm before training
    response = input("\nProceed with training? [y/N]: ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SCITrainer(config)

    # Start training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted')
        print("Checkpoint saved.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving checkpoint...")
        trainer.save_checkpoint('error')
        raise

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best loss: {trainer.best_loss:.4f}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
