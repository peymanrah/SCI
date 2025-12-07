#!/usr/bin/env python
"""
SCI Training Script - Section 3.1-3.2 & 5.1-5.3

Implements:
- Fair comparison training protocol
- Optimizer groups with separate learning rates
- Dynamic loss weighting with SCL warmup
- Checkpoint management and training resumption
- Early stopping with overfitting detection
- Comprehensive logging and metrics tracking

LOW #85: Training Time Estimates
---------------------------------
Expected training times on different hardware:

RTX 3090 (24GB):
  - SCI Full (batch=4, grad_accum=8): ~2 hours for 50 epochs
  - Baseline (batch=4, grad_accum=8): ~1.5 hours for 50 epochs

V100 (32GB):
  - SCI Full (batch=8, grad_accum=4): ~1.5 hours for 50 epochs
  - Baseline (batch=8, grad_accum=4): ~1 hour for 50 epochs

A100 (40GB):
  - SCI Full (batch=16, grad_accum=2): ~1 hour for 50 epochs
  - Baseline (batch=16, grad_accum=2): ~40 minutes for 50 epochs

Note: Times assume SCAN length split with ~16k training examples
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from sci.config.config_loader import load_config
from sci.models.sci_model import SCIModel
from sci.data.datasets.scan_dataset import SCANDataset
from sci.data.scan_data_collator import SCANDataCollator
from sci.data.pair_generators.scan_pair_generator import SCANPairGenerator
from sci.models.losses.combined_loss import SCICombinedLoss
from sci.evaluation.scan_evaluator import SCANEvaluator
from sci.training.checkpoint_manager import CheckpointManager, TrainingResumer
from sci.training.early_stopping import EarlyStopping, OverfittingDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SCI model on SCAN')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/sci_full.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--wandb-project', type=str, default='sci-training',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def create_optimizer_groups(model, config):
    """
    Create optimizer groups with separate learning rates.

    Section 3.2: Base model parameters use base_lr=2e-5,
    SCI components use sci_lr=5e-5.

    HIGH #28: Bias and LayerNorm parameters should not have weight decay.
    """
    base_params = []
    sci_params = []
    no_decay_params = []

    # Separate parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # HIGH #28: Exclude bias and layer norm params from weight decay
        if any(nd in name for nd in ['bias', 'LayerNorm.weight', 'layer_norm.weight']):
            no_decay_params.append(param)
        # SCI components have specific prefixes
        elif any(x in name for x in ['structural_encoder', 'content_encoder',
                                      'causal_binding', 'abstraction']):
            sci_params.append(param)
        else:
            base_params.append(param)

    optimizer_groups = [
        {'params': base_params, 'lr': config['training']['optimizer']['base_lr']},
        {'params': sci_params, 'lr': config['training']['optimizer']['sci_lr']},
        {'params': no_decay_params, 'lr': config['training']['optimizer']['base_lr'], 'weight_decay': 0.0},
    ]

    print(f"  Base params: {len(base_params)}")
    print(f"  SCI params: {len(sci_params)}")
    print(f"  No decay params (bias/LayerNorm): {len(no_decay_params)}")

    return optimizer_groups


def compute_scl_weight_warmup(epoch, warmup_epochs=2):
    """
    Compute SCL weight with linear warmup.

    Section 3.2: SCL loss should warm up over first 2 epochs.
    """
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def train_epoch(model, train_loader, optimizer, scheduler, criterion,
                evaluator, device, epoch, config, pair_generator):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_lm_loss = 0
    total_scl_loss = 0
    total_ortho_loss = 0
    num_batches = 0

    # SCL warmup
    warmup_epochs = config['training']['loss'].get('scl_warmup_epochs', 2)
    warmup_factor = compute_scl_weight_warmup(epoch, warmup_epochs)
    scl_weight = config['training']['loss']['scl_weight'] * warmup_factor

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        instruction_mask = batch['instruction_mask'].to(device)

        # Get pair labels for this batch
        commands = batch.get('commands', [])
        pair_labels = pair_generator.get_batch_pair_labels(commands).to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            instruction_mask=instruction_mask,
        )

        # Compute loss
        losses = criterion(outputs, pair_labels, scl_weight_override=scl_weight)
        loss = losses['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        max_grad_norm = config['training']['optimizer'].get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_lm_loss += losses['lm_loss'].item()
        total_scl_loss += losses['scl_loss'].item()
        total_ortho_loss += losses['orthogonality_loss'].item()
        num_batches += 1

        # Log every N batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f} "
                  f"LM={losses['lm_loss'].item():.4f} "
                  f"SCL={losses['scl_loss'].item():.4f} "
                  f"Ortho={losses['orthogonality_loss'].item():.4f}")

    # Return average metrics
    return {
        'train_loss': total_loss / num_batches,
        'train_lm_loss': total_lm_loss / num_batches,
        'train_scl_loss': total_scl_loss / num_batches,
        'train_ortho_loss': total_ortho_loss / num_batches,
        'scl_weight': scl_weight,
        'warmup_factor': warmup_factor,  # CRITICAL #16: Save warmup_factor to metrics
    }


def validate(model, val_loader, criterion, evaluator, device, pair_generator):
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_lm_loss = 0
    total_scl_loss = 0
    total_ortho_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            instruction_mask = batch['instruction_mask'].to(device)

            # Get pair labels
            commands = batch.get('commands', [])
            pair_labels = pair_generator.get_batch_pair_labels(commands).to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                instruction_mask=instruction_mask,
            )

            # Compute loss
            losses = criterion(outputs, pair_labels)

            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_lm_loss += losses['lm_loss'].item()
            total_scl_loss += losses['scl_loss'].item()
            total_ortho_loss += losses['orthogonality_loss'].item()
            num_batches += 1

    # Compute exact match accuracy
    eval_results = evaluator.evaluate(model, val_loader, device=device)

    return {
        'val_loss': total_loss / num_batches,
        'val_lm_loss': total_lm_loss / num_batches,
        'val_scl_loss': total_scl_loss / num_batches,
        'val_ortho_loss': total_ortho_loss / num_batches,
        'val_exact_match': eval_results['exact_match'],
        'val_token_accuracy': eval_results['token_accuracy'],
    }


def main():
    """Main training function."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Setup output directory
    output_dir = Path(args.output_dir) / config['experiment']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Initialize Weights & Biases
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or config['experiment']['name'],
            config=config,
        )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        cache_dir='.cache/models'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    print("Loading SCAN dataset...")
    split = config['data']['scan_split']
    
    # BUG #88 FIX: Pass tokenizer to SCANDataset constructor
    train_dataset = SCANDataset(
        tokenizer=tokenizer,
        split_name=split,
        subset='train',
        max_length=config['data'].get('max_seq_length', 512),
        cache_dir='.cache/scan',
    )
    val_dataset = SCANDataset(
        tokenizer=tokenizer,
        split_name=split,
        subset='test',  # SCAN uses 'test' not 'val'
        max_length=config['data'].get('max_seq_length', 512),
        cache_dir='.cache/scan',
    )

    # MEDIUM #58: Log dataset sizes
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Validation: {len(val_dataset):,} examples")

    # BUG #86, #87 FIX: Create data collator with pair generator for proper concatenation
    collator = SCANDataCollator(
        tokenizer=tokenizer,
        max_length=config['data'].get('max_seq_length', 512),
        pair_generator=train_dataset.pair_generator,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Create model
    print("Creating SCI model...")
    model = SCIModel(config)
    model = model.to(device)

    # Create optimizer
    print("Creating optimizer...")
    optimizer_groups = create_optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=config['training']['optimizer']['weight_decay'],
    )

    # Create scheduler (optional)
    scheduler = None
    if config['training']['optimizer'].get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config['training']['epochs'] * len(train_loader),
        )

    # Create loss function
    criterion = SCICombinedLoss(
        scl_weight=config['training']['loss']['scl_weight'],
        ortho_weight=config['training']['loss']['orthogonality_weight'],
        temperature=config['training']['loss']['temperature'],
    )

    # Create evaluator with eval config
    eval_config = config.get('evaluation', {})
    evaluator = SCANEvaluator(tokenizer, eval_config=eval_config)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        save_total_limit=config['training']['checkpointing']['save_total_limit'],
    )

    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        mode='max',  # Maximize exact match
    )

    overfitting_detector = OverfittingDetector(
        threshold_ratio=config['training']['early_stopping'].get('overfitting_threshold', 1.5),
        window_size=3,
        min_epochs=5,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    training_state = {
        'best_val_metric': 0.0,
        'best_epoch': 0,
        'early_stopping_counter': 0,
    }

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resumer = TrainingResumer(args.resume)
        resumer.load_checkpoint(model, optimizer, scheduler)
        start_epoch = resumer.checkpoint['epoch'] + 1
        training_state = {
            'best_val_metric': resumer.checkpoint.get('best_val_metric', 0.0),
            'best_epoch': resumer.checkpoint.get('best_epoch', 0),
            'early_stopping_counter': resumer.checkpoint.get('early_stopping_counter', 0),
        }

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"Total epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['training']['epochs']} ===")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            evaluator, device, epoch, config, pair_generator
        )

        # Validate
        print("Validating...")
        val_metrics = validate(
            model, val_loader, criterion, evaluator, device, pair_generator
        )

        # Combine metrics
        metrics = {**train_metrics, **val_metrics}

        # Print metrics
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Val Loss: {metrics['val_loss']:.4f}")
        print(f"  Val Exact Match: {metrics['val_exact_match']:.4f}")
        print(f"  SCL Weight: {metrics['scl_weight']:.4f}")

        # Log to W&B
        if not args.no_wandb:
            wandb.log({**metrics, 'epoch': epoch}, step=epoch)

        # Check for best model
        if metrics['val_exact_match'] > training_state['best_val_metric']:
            training_state['best_val_metric'] = metrics['val_exact_match']
            training_state['best_epoch'] = epoch

            # Save best checkpoint
            print(f"  New best model! Exact Match: {metrics['val_exact_match']:.4f}")
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, epoch * len(train_loader),
                metrics, config, training_state
            )

        # Check early stopping
        should_stop = early_stopping(metrics['val_exact_match'], epoch)
        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best epoch: {training_state['best_epoch']+1}")
            print(f"Best val exact match: {training_state['best_val_metric']:.4f}")
            break

        # CRITICAL #14: Check overfitting - properly use return value (tuple)
        overfitting_detector.update(metrics['train_loss'], metrics['val_loss'], epoch)
        is_overfitting, loss_ratio = overfitting_detector.is_overfitting()
        if is_overfitting:
            print(f"\nOverfitting detected at epoch {epoch+1}")
            print(f"Train/Val loss ratio: {loss_ratio:.4f}")
            break

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    checkpoint_manager.save_checkpoint(
        model, optimizer, scheduler, epoch, epoch * len(train_loader),
        metrics, config, training_state
    )

    print("\n=== Training Complete ===")
    print(f"Best epoch: {training_state['best_epoch']+1}")
    print(f"Best val exact match: {training_state['best_val_metric']:.4f}")
    print(f"Output directory: {output_dir}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
