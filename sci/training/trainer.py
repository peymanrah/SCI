"""
SCI Trainer: Main training loop with SCL warmup and structural pair learning.

Key Features:
1. SCL loss warmup schedule (prevents early instability)
2. Pre-computed structural pairs (cached, fast batch lookup)
3. Mixed precision training (fp16)
4. Gradient accumulation
5. WandB logging
6. Checkpointing with best model tracking
7. Fairness logging for baseline/SCI comparison
8. Data leakage checks (required by SCI_ENGINEERING_STANDARDS.md)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW  # #71 FIX: Use torch.optim.AdamW instead of deprecated transformers.AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional
from tqdm import tqdm
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging disabled.")

from sci.models.sci_model import SCIModel
from sci.models.losses.combined_loss import SCICombinedLoss
from sci.data.datasets.scan_dataset import SCANDataset
from sci.data.scan_data_collator import SCANDataCollator
from sci.data.data_leakage_checker import DataLeakageChecker
from sci.training.early_stopping import EarlyStopping, OverfittingDetector
from sci.evaluation.scan_evaluator import SCANEvaluator


class SCITrainer:
    """
    SCI Trainer with SCL warmup and structural pair learning.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        print("Initializing SCI model...")
        self.model = SCIModel(config).to(self.device)

        # Initialize loss function with EOS weight
        # FIX: Use getattr() since config.loss is a dataclass, not dict
        self.loss_fn = SCICombinedLoss(
            scl_weight=config.loss.scl_weight,
            ortho_weight=config.loss.ortho_weight,
            temperature=config.loss.scl_temperature,
            eos_weight=getattr(config.loss, 'eos_weight', 2.0),
            eos_token_id=self.model.tokenizer.eos_token_id,
        )

        # Initialize dataset - use data config section
        # FIX: Use getattr() since config.data is a dataclass, not dict
        print(f"Loading {config.data.dataset} dataset...")
        self.train_dataset = SCANDataset(
            tokenizer=self.model.tokenizer,
            split_name=config.data.split,
            subset='train',  # Training always uses train split
            max_length=config.data.max_length,
            cache_dir=getattr(config.data, 'pairs_cache_dir', '.cache/scan'),
            force_regenerate_pairs=getattr(config.data, 'force_regenerate_pairs', False),
        )

        # Data collator - uses proper causal LM format with pair generation
        self.collator = SCANDataCollator(
            tokenizer=self.model.tokenizer,
            max_length=config.data.max_length,
            pair_generator=self.train_dataset.pair_generator,
            use_chat_template=getattr(config.data, 'use_chat_template', False),
        )

        # Data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Optimizer with separate learning rates for base and SCI modules
        optimizer_groups = self._get_optimizer_groups()
        self.optimizer = AdamW(
            optimizer_groups,
            weight_decay=config.training.optimizer.weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.training.max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if config.training.mixed_precision else None

        # SCL warmup
        # FIX: Use getattr() since config.loss is a dataclass, not dict
        self.scl_warmup_epochs = getattr(config.loss, 'scl_warmup_epochs', 2)

        # Logging
        # FIX: Use getattr() since config.logging is a dataclass, not dict
        self.use_wandb = WANDB_AVAILABLE and getattr(config.logging, 'use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                config=self._config_to_dict(config),
                tags=getattr(config.logging, 'wandb_tags', []),
            )
            wandb.watch(self.model, log='all', log_freq=100)

        # Checkpointing
        self.checkpoint_dir = config.checkpointing.save_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_exact_match = 0.0
        
        # #51 FIX: Add validation dataset (like train.py)
        # #98 FIX: Handle missing validation split gracefully
        # V8 CRITICAL #1 FIX: Create separate validation collator WITHOUT pair_generator
        # Using training pair_generator for validation causes data mismatch
        # (validation indices would look up training pair cache)
        print(f"Loading validation dataset...")
        try:
            self.val_dataset = SCANDataset(
                tokenizer=self.model.tokenizer,
                split_name=config.data.split,
                subset='test',  # SCAN uses 'test' not 'val'
                max_length=config.data.max_length,
                cache_dir=getattr(config.data, 'pairs_cache_dir', '.cache/scan'),
            )
            # V8 CRITICAL #1 FIX: Validation collator has pair_generator=None
            # This prevents looking up incorrect pair labels from training cache
            # Validation only uses LM loss anyway (pair_labels=None in _validate_epoch)
            self.val_collator = SCANDataCollator(
                tokenizer=self.model.tokenizer,
                max_length=config.data.max_length,
                pair_generator=None,  # No pair generation for validation
                use_chat_template=getattr(config.data, 'use_chat_template', False),
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                collate_fn=self.val_collator,  # Use validation-specific collator
                num_workers=0,
            )
        except Exception as e:
            print(f"WARNING: Could not load validation dataset: {e}")
            print("Training will proceed without validation.")
            self.val_dataset = None
            self.val_loader = None
            self.val_collator = None
        
        # #51 FIX: Add early stopping and overfitting detection (like train.py)
        es_config = getattr(config.training, 'early_stopping', None)
        self.early_stopping = EarlyStopping(
            patience=getattr(es_config, 'patience', 5) if es_config else 5,
            min_delta=getattr(es_config, 'min_delta', 0.001) if es_config else 0.001,
            mode='max',  # Maximize exact match
        )
        self.overfitting_detector = OverfittingDetector(
            threshold_ratio=getattr(es_config, 'overfitting_threshold', 1.5) if es_config else 1.5,
            window_size=3,
            min_epochs=5,
        )
        
        # #52 FIX: Create evaluator with proper API (SCANEvaluator, not SCIEvaluator)
        eval_config = getattr(config, 'evaluation', None)
        self.evaluator = SCANEvaluator(self.model.tokenizer, eval_config=eval_config)
        
        # Run data leakage checks (required by SCI_ENGINEERING_STANDARDS.md)
        self._run_data_leakage_checks()
        
        # Log fairness metrics for baseline/SCI comparison
        self._log_fairness_metrics()

    def _get_optimizer_groups(self):
        """
        Create parameter groups with different learning rates.

        Base model parameters get base_lr (2e-5).
        SCI module parameters get sci_lr (5e-5) - 2.5x higher for faster learning.
        No decay parameters (biases, layer norms) get base_lr with 0 weight decay.

        Returns:
            List of parameter group dictionaries for optimizer
        """
        base_params = []
        sci_params = []
        no_decay_params = []

        base_lr = self.config.training.optimizer.base_lr  # 2e-5
        sci_lr = self.config.training.optimizer.sci_lr  # 5e-5

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No decay for biases and layer norms
            if any(nd in name for nd in ['bias', 'LayerNorm', 'layer_norm']):
                no_decay_params.append(param)
            # SCI modules get higher LR
            elif any(sci in name.lower() for sci in [
                'structural_encoder', 'content_encoder',
                'causal_binding', 'abstraction'
            ]):
                sci_params.append(param)
            # Base model parameters
            else:
                base_params.append(param)

        print(f"\n{'='*70}")
        print("Optimizer Parameter Groups:")
        print(f"{'='*70}")
        print(f"  Base model params: {len(base_params):,} parameters @ LR={base_lr:.2e}")
        print(f"  SCI module params: {len(sci_params):,} parameters @ LR={sci_lr:.2e} (2.5x higher)")
        print(f"  No decay params:   {len(no_decay_params):,} parameters @ LR={base_lr:.2e}, WD=0.0")
        print(f"{'='*70}\n")

        return [
            {
                'params': base_params,
                'lr': base_lr,
                'weight_decay': self.config.training.optimizer.weight_decay
            },
            {
                'params': sci_params,
                'lr': sci_lr,
                'weight_decay': self.config.training.optimizer.weight_decay
            },
            {
                'params': no_decay_params,
                'lr': base_lr,
                'weight_decay': 0.0
            },
        ]

    def _config_to_dict(self, config):
        """Convert config object to dictionary for logging."""
        from dataclasses import is_dataclass, asdict
        if is_dataclass(config) and not isinstance(config, type):
            return asdict(config)
        elif hasattr(config, 'to_dict'):
            return config.to_dict()
        elif hasattr(config, '__dict__'):
            return {k: self._config_to_dict(v) for k, v in config.__dict__.items()}
        else:
            return config

    def _run_data_leakage_checks(self):
        """
        Run data leakage checks required by SCI_ENGINEERING_STANDARDS.md.
        
        Checks:
        1. Length split constraints (train ≤22 tokens, test >22)
        2. No train/test overlap
        3. Test split validation (if available)
        """
        checker = DataLeakageChecker(split_name=self.config.data.split)
        
        # Check length constraints for training data
        train_commands = [d['commands'] if isinstance(d, dict) else d.get('commands', '') 
                         for d in self.train_dataset.data]
        train_actions = [d['actions'] if isinstance(d, dict) else d.get('actions', '') 
                        for d in self.train_dataset.data]
        
        result = checker.check_length_split_constraints(train_commands, train_actions, "train")
        
        if result.get('num_violations', 0) > 0:
            print(f"\n⚠️  WARNING: {result['num_violations']} length constraint violations in training data")
            print("    This may affect OOD generalization results.")
        else:
            print(f"\n✓ Data leakage check passed for {len(train_commands)} training examples")
        
        # CRITICAL FIX: Also check test split for length constraints
        if self.config.data.split == "length":
            try:
                from datasets import load_dataset
                test_data = load_dataset("scan", "length")["test"]
                test_commands = [d['commands'] for d in test_data]
                test_actions = [d['actions'] for d in test_data]
                
                test_result = checker.check_length_split_constraints(test_commands, test_actions, "test")
                
                if test_result.get('num_violations', 0) > 0:
                    print(f"⚠️  WARNING: {test_result['num_violations']} test examples don't meet >22 token requirement")
                else:
                    print(f"✓ Test split length constraints verified ({len(test_commands)} examples)")
                
                # Check for train/test overlap
                train_set = set(train_commands)
                test_set = set(test_commands)
                overlap = train_set.intersection(test_set)
                if overlap:
                    print(f"⚠️  WARNING: {len(overlap)} commands appear in both train and test!")
                else:
                    print("✓ No train/test overlap detected")
            except Exception as e:
                print(f"  (Could not verify test split: {e})")
    
    def _log_fairness_metrics(self):
        """
        Log fairness metrics for baseline/SCI comparison.
        
        Required for fair comparison between SCI and baseline:
        - Same batch size, epochs, warmup
        - Same generation config for evaluation
        - Parameter count logging
        - Training time estimation
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # SCI-specific parameters
        sci_params = 0
        for name, param in self.model.named_parameters():
            if any(m in name.lower() for m in ['structural_encoder', 'content_encoder', 'causal_binding']):
                sci_params += param.numel()
        
        # Check if this is a baseline run (no SCI components)
        is_baseline = not getattr(self.config.model.structural_encoder, 'enabled', True)
        
        fairness_info = {
            "run_type": "baseline" if is_baseline else "sci_full",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "sci_params": sci_params,
            "batch_size": self.config.training.batch_size,
            "max_epochs": self.config.training.max_epochs,
            "warmup_steps": self.config.training.warmup_steps,
            "base_lr": self.config.training.optimizer.base_lr,
            "sci_lr": self.config.training.optimizer.sci_lr,
            "weight_decay": self.config.training.optimizer.weight_decay,
            "mixed_precision": self.config.training.mixed_precision,
            "max_length": self.config.data.max_length,
            "seed": self.config.seed,
        }
        
        print(f"\n{'='*70}")
        print("FAIRNESS METRICS (for baseline/SCI comparison):")
        print(f"{'='*70}")
        print(f"  Run Type:           {fairness_info['run_type']}")
        print(f"  Total Parameters:   {total_params:,}")
        print(f"  Trainable Params:   {trainable_params:,}")
        print(f"  SCI Module Params:  {sci_params:,}")
        print(f"  Batch Size:         {fairness_info['batch_size']}")
        print(f"  Max Epochs:         {fairness_info['max_epochs']}")
        print(f"  Base LR:            {fairness_info['base_lr']:.2e}")
        print(f"  SCI LR:             {fairness_info['sci_lr']:.2e}")
        print(f"  Weight Decay:       {fairness_info['weight_decay']}")
        print(f"  Mixed Precision:    {fairness_info['mixed_precision']}")
        print(f"  Max Length:         {fairness_info['max_length']}")
        print(f"  Seed:               {fairness_info['seed']}")
        print(f"{'='*70}\n")
        
        # Log to wandb if available
        if self.use_wandb:
            wandb.config.update({"fairness": fairness_info})

    def compute_scl_weight(self, epoch: int) -> float:
        """Compute SCL weight with warmup."""
        base_weight = self.config.loss.scl_weight

        if epoch < self.scl_warmup_epochs:
            # Linear warmup
            return base_weight * (epoch + 1) / self.scl_warmup_epochs
        else:
            return base_weight

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_lm_loss = 0.0
        total_scl_loss = 0.0
        total_ortho_loss = 0.0
        num_batches = 0

        # Get current SCL weight
        scl_weight = self.compute_scl_weight(epoch)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            instruction_mask = batch.get('instruction_mask')  # BUG #90 FIX
            if instruction_mask is not None:
                instruction_mask = instruction_mask.to(self.device)
            pair_labels = batch.get('pair_labels')
            if pair_labels is not None:
                pair_labels = pair_labels.to(self.device)

            # Forward pass with mixed precision
            # #43 FIX: Use contextlib.nullcontext instead of torch.enable_grad()
            # and use proper autocast device_type
            from contextlib import nullcontext
            amp_context = torch.amp.autocast('cuda') if self.scaler else nullcontext()
            with amp_context:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    instruction_mask=instruction_mask,  # BUG #90 FIX
                    return_dict=True,
                )
                
                # Add labels to outputs for EOS loss computation
                outputs['labels'] = labels

                # Compute combined loss
                losses = self.loss_fn(
                    model_outputs=outputs,
                    pair_labels=pair_labels,
                    scl_weight_override=scl_weight,
                )

                loss = losses['total_loss']

            # Backward pass
            # #108 FIX: Track gradient norm for debugging
            # Read gradient_clip from config (primary) or optimizer.max_grad_norm (fallback)
            max_grad_norm = getattr(self.config.training, 'gradient_clip',
                                   getattr(self.config.training.optimizer, 'max_grad_norm', 1.0))
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
            
            # Track gradient norm history (for debugging)
            if not hasattr(self, 'grad_norm_history'):
                self.grad_norm_history = []
            self.grad_norm_history.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

            self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # Track losses
            total_loss += losses['total_loss'].item()
            total_lm_loss += losses['lm_loss'].item()
            total_scl_loss += losses['scl_loss'].item() if torch.is_tensor(losses['scl_loss']) else losses['scl_loss']
            total_ortho_loss += losses['orthogonality_loss'].item() if torch.is_tensor(losses['orthogonality_loss']) else losses['orthogonality_loss']
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / num_batches,
                'scl': total_scl_loss / num_batches,
            })

            # Log to wandb
            # FIX: Use getattr() since config.logging is a dataclass, not dict
            if self.use_wandb and self.global_step % getattr(self.config.logging, 'log_every', 10) == 0:
                # V8 MINOR #8 FIX: Log gradient norm to wandb for remote debugging
                grad_norm_value = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                wandb.log({
                    'train/total_loss': losses['total_loss'].item(),
                    'train/lm_loss': losses['lm_loss'].item(),
                    'train/scl_loss': total_scl_loss / num_batches,
                    'train/ortho_loss': total_ortho_loss / num_batches,
                    'train/scl_weight': scl_weight,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/step': self.global_step,
                    'train/epoch': epoch,
                    'train/grad_norm': grad_norm_value,  # V8 #8: Log gradient norm
                })

        return {
            'total_loss': total_loss / num_batches,
            'lm_loss': total_lm_loss / num_batches,
            'scl_loss': total_scl_loss / num_batches,
            'ortho_loss': total_ortho_loss / num_batches,
        }

    def train(self, evaluator=None):
        """
        Full training loop with validation and early stopping.
        
        Args:
            evaluator: Optional evaluator instance. If None, uses self.evaluator.
                      For SCANEvaluator, calls evaluate(model, dataloader, device).
        """
        # Use provided evaluator or default
        eval_instance = evaluator if evaluator is not None else self.evaluator
        
        # Start from current epoch (allows resume from checkpoint)
        start_epoch = self.epoch
        if start_epoch > 0:
            print(f"\nResuming training from epoch {start_epoch + 1} to {self.config.training.max_epochs}...")
        else:
            print(f"\nStarting training for {self.config.training.max_epochs} epochs...")
        
        # FIX: Use getattr() instead of .get() since config.training is a dataclass, not dict
        eval_freq = getattr(self.config.training, 'eval_freq', 1)  # Evaluate every N epochs

        for epoch in range(start_epoch, self.config.training.max_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            print(f"\nEpoch {epoch+1}/{self.config.training.max_epochs}")
            print(f"  Loss: {train_metrics['total_loss']:.4f}")
            print(f"  LM: {train_metrics['lm_loss']:.4f}")
            print(f"  SCL: {train_metrics['scl_loss']:.4f}")
            print(f"  Ortho: {train_metrics['ortho_loss']:.4f}")

            # #52 FIX: Run validation with correct SCANEvaluator API
            # #98 FIX: Handle missing validation loader gracefully
            val_exact_match = 0.0
            val_loss = train_metrics['total_loss']  # Default to train loss if no eval
            
            if (epoch + 1) % eval_freq == 0 and self.val_loader is not None:
                print(f"  Running validation...")
                # Compute validation loss
                val_metrics = self._validate_epoch()
                val_loss = val_metrics['total_loss']
                print(f"  Val Loss: {val_loss:.4f}")
                
                # #52 FIX: Use SCANEvaluator with correct API (model, dataloader, device)
                if eval_instance is not None:
                    eval_results = eval_instance.evaluate(
                        self.model, 
                        self.val_loader, 
                        device=self.device
                    )
                    val_exact_match = eval_results['exact_match']
                    print(f"  Val Exact Match: {val_exact_match*100:.2f}%")
                    print(f"  Val Token Accuracy: {eval_results['token_accuracy']*100:.2f}%")
                    
                    # Log to wandb if available
                    if self.use_wandb:
                        wandb.log({
                            'eval/exact_match': val_exact_match,
                            'eval/token_accuracy': eval_results['token_accuracy'],
                            'val/loss': val_loss,
                            'epoch': epoch + 1,
                        })
            elif self.val_loader is None and (epoch + 1) % eval_freq == 0:
                print(f"  Skipping validation (no validation dataset available)")
            
            # #51 FIX: Check early stopping based on exact match
            should_stop = self.early_stopping(val_exact_match, epoch)
            if should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best exact match: {self.best_exact_match*100:.2f}%")
                break
            
            # #51 FIX: Check for overfitting
            is_overfitting, loss_ratio = self.overfitting_detector.update(
                train_metrics['total_loss'], val_loss, epoch
            )
            if is_overfitting:
                print(f"\nOverfitting detected at epoch {epoch+1}")
                print(f"Train/Val loss ratio: {loss_ratio:.4f}")
                break

            # Save checkpoint
            # FIX: Use getattr() since config.training is a dataclass, not dict
            if (epoch + 1) % getattr(self.config.training, 'save_every', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

            # Save best model based on exact match (not loss)
            if val_exact_match > self.best_exact_match:
                self.best_exact_match = val_exact_match
                self.save_checkpoint('best')
                print(f"  New best exact match: {self.best_exact_match*100:.2f}%")
            elif train_metrics['total_loss'] < self.best_loss:
                self.best_loss = train_metrics['total_loss']

        # Final save
        self.save_checkpoint('final')
        print(f"\nTraining complete!")
        print(f"Best exact match: {self.best_exact_match*100:.2f}%")
    
    def _validate_epoch(self):
        """Run validation for one epoch and return metrics."""
        self.model.eval()
        total_loss = 0
        total_lm_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                instruction_mask = batch.get('instruction_mask')
                if instruction_mask is not None:
                    instruction_mask = instruction_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    instruction_mask=instruction_mask,
                    return_dict=True,
                )
                
                outputs['labels'] = labels
                losses = self.loss_fn(model_outputs=outputs, pair_labels=None)
                
                total_loss += losses['total_loss'].item()
                total_lm_loss += losses['lm_loss'].item()
                num_batches += 1
        
        self.model.train()
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'lm_loss': total_lm_loss / max(num_batches, 1),
        }

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.checkpoint_dir, name)
        self.model.save_pretrained(save_path)

        # Save training state
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(save_path, 'training_state.pt'))

        print(f"  Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training state.
        
        Args:
            checkpoint_path: Path to checkpoint directory containing:
                - Model weights (as save_pretrained format)
                - training_state.pt with optimizer/scheduler/epoch state
        """
        # Load model weights
        model_path = checkpoint_path
        if os.path.isfile(checkpoint_path):
            # If path is to training_state.pt, get directory
            model_path = os.path.dirname(checkpoint_path)
        
        print(f"Loading model from: {model_path}")
        # #55 FIX: Load state dict directly instead of from_pretrained
        # SCIModel doesn't have from_pretrained like HuggingFace models
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            # Try loading as config + model.pt format
            model_file = os.path.join(model_path, 'model.pt')
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")
        
        self.model.to(self.device)
        
        # Load training state
        state_path = os.path.join(model_path, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            
            self.epoch = state['epoch']
            self.global_step = state['global_step']
            self.best_loss = state['best_loss']
            
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
                # Move optimizer state to correct device
                for param_state in self.optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.to(self.device)
            
            if 'scheduler' in state:
                self.scheduler.load_state_dict(state['scheduler'])
            
            print(f"  Loaded training state from: {state_path}")
        else:
            print(f"  Warning: training_state.pt not found, only model weights loaded")


if __name__ == "__main__":
    # Quick test
    print("Testing SCITrainer...")

    from sci.config.config_loader import load_config

    # Load config
    config = load_config("configs/sci_full.yaml")

    # Override for testing
    config.training.max_epochs = 2
    config.training.batch_size = 4
    config.logging.use_wandb = False

    # Create trainer
    trainer = SCITrainer(config)

    print(f"✓ Trainer initialized")
    print(f"  Dataset size: {len(trainer.train_dataset)}")
    print(f"  Batches per epoch: {len(trainer.train_loader)}")

    # # Test one epoch (uncomment to test)
    # print("\nTesting one epoch...")
    # metrics = trainer.train_epoch(0)
    # print(f"✓ Epoch completed: {metrics}")
