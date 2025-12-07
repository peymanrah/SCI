#!/usr/bin/env python
"""
Train SCI Model on SCAN Benchmark (SCITrainer class-based entry point).

#25 NOTE: This script uses the SCITrainer class from sci.training.trainer.
For the standalone script approach, use train.py in the project root.
Both approaches are fully functional and produce equivalent results.

Usage:
    python scripts/train_sci.py --config configs/sci_full.yaml
    python scripts/train_sci.py --config configs/ablations/no_scl.yaml --output_dir checkpoints/ablation_no_scl
    
Alternative (standalone script):
    python train.py --config configs/sci_full.yaml
"""

import argparse
import os
import sys
import random
import numpy as np
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
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (e.g., checkpoints/best/training_state.pt)'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (non-interactive mode)'
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
    # FIX: Use correct config paths (checkpointing.save_dir, not training.checkpoint_dir)
    if args.output_dir:
        config.checkpointing.save_dir = args.output_dir
        print(f"  Output directory: {args.output_dir}")

    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
        print(f"  Max epochs: {args.max_epochs}")

    if args.batch_size:
        config.training.batch_size = args.batch_size
        print(f"  Batch size: {args.batch_size}")

    # FIX: Use correct config path (training.optimizer.base_lr, not training.learning_rate)
    if args.learning_rate:
        config.training.optimizer.base_lr = args.learning_rate
        # Also update the default lr for compatibility
        config.training.optimizer.lr = args.learning_rate
        print(f"  Learning rate: {args.learning_rate}")

    if args.no_wandb:
        config.logging.use_wandb = False
        print("  WandB logging: DISABLED")

    if args.seed:
        config.seed = args.seed

    # FIX: Use correct config path (data.force_regenerate_pairs, not training.force_regenerate_pairs)
    if args.force_regenerate_pairs:
        config.data.force_regenerate_pairs = True
        print("  Force regenerate pairs: ENABLED")

    # Set random seed for full reproducibility
    seed = getattr(config, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms for reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"\nRandom seed: {seed} (applied to random, numpy, torch)")

    # Print configuration summary
    # FIX: Use correct config paths (data.dataset/split, training.optimizer.base_lr, checkpointing.save_dir)
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Model: {config.model.base_model_name}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Split: {config.data.split}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Max epochs: {config.training.max_epochs}")
    print(f"Learning rate: {config.training.optimizer.base_lr}")
    print(f"SCL weight: {config.loss.scl_weight} (warmup: {getattr(config.loss, 'scl_warmup_epochs', 2)} epochs)")

    print("\nSCI Components:")
    print(f"  Structural Encoder: {'ENABLED' if config.model.structural_encoder.enabled else 'DISABLED'}")
    print(f"  Content Encoder: {'ENABLED' if config.model.content_encoder.enabled else 'DISABLED'}")
    print(f"  Causal Binding: {'ENABLED' if config.model.causal_binding.enabled else 'DISABLED'}")

    if config.model.structural_encoder.enabled:
        print(f"  - AbstractionLayer injection: {config.model.structural_encoder.abstraction_layer.injection_layers}")
        print(f"  - Structural slots: {config.model.structural_encoder.num_slots}")

    if config.model.causal_binding.enabled:
        print(f"  - CBM injection layers: {config.model.causal_binding.injection_layers}")

    # FIX: Use correct config path (checkpointing.save_dir, not training.checkpoint_dir)
    print(f"\nCheckpoint directory: {config.checkpointing.save_dir}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 70)

    # Confirm before training (skip if --yes flag is set)
    if not args.yes:
        response = input("\nProceed with training? [y/N]: ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    else:
        print("\nNon-interactive mode: proceeding with training...")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SCITrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint from: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  Resumed from epoch {trainer.epoch + 1}, global step {trainer.global_step}")
        print(f"  Best loss so far: {trainer.best_loss:.4f}")

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
    # FIX: Use correct config path (checkpointing.save_dir, not training.checkpoint_dir)
    print(f"Checkpoints saved to: {config.checkpointing.save_dir}")


if __name__ == "__main__":
    main()
