"""
Checkpoint Manager for SCI Training

Manages checkpoints for training resumption.
Required by: SCI_ENGINEERING_STANDARDS.md Section 5.2
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path


class CheckpointManager:
    """Manages checkpoints for training resumption."""

    def __init__(self, checkpoint_dir, save_total_limit=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_total_limit = save_total_limit
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_config(self, config):
        """
        Serialize config to dict for checkpointing.
        
        #44 FIX: Properly handle dataclass configs.
        """
        if isinstance(config, dict):
            return config
        elif hasattr(config, 'to_dict'):
            return config.to_dict()
        else:
            # Fallback for dataclasses
            from dataclasses import asdict, is_dataclass
            if is_dataclass(config):
                return asdict(config)
            else:
                # Last resort - try vars()
                try:
                    return vars(config)
                except TypeError:
                    return str(config)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, step,
                       metrics, config, training_state):
        """Save full training checkpoint."""

        checkpoint = {
            # Model state
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,

            # Training progress
            'epoch': epoch,
            'global_step': step,
            'best_val_metric': training_state.get('best_val_metric', 0.0),
            'best_epoch': training_state.get('best_epoch', 0),
            'early_stopping_counter': training_state.get('early_stopping_counter', 0),

            # Metrics history
            'metrics_history': metrics,

            # Configuration - #44 FIX: Properly serialize dataclass configs
            'config': self._serialize_config(config),

            # Metadata
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }

        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)

        # Save latest pointer
        latest_path = self.checkpoint_dir / "latest_checkpoint.json"
        with open(latest_path, 'w') as f:
            json.dump({
                'checkpoint_name': checkpoint_name,
                'epoch': epoch,
                'step': step,
                'timestamp': checkpoint['timestamp']
            }, f, indent=2)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        print(f"[OK] Saved checkpoint: {checkpoint_path}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint for resumption."""

        if checkpoint_path is None:
            # Load latest
            latest_path = self.checkpoint_dir / "latest_checkpoint.json"
            if not latest_path.exists():
                return None
            with open(latest_path, 'r') as f:
                latest_info = json.load(f)
            checkpoint_path = self.checkpoint_dir / latest_info['checkpoint_name']
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # HIGH #43: Handle checkpoint corruption
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            print("Checkpoint may be corrupted. Skipping...")
            return None

        # Validate checkpoint structure
        required_keys = ['epoch', 'global_step', 'model_state_dict']
        if not all(key in checkpoint for key in required_keys):
            print(f"Warning: Checkpoint missing required keys: {required_keys}")
            return None

        print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only save_total_limit."""
        checkpoints = []
        for f in self.checkpoint_dir.glob("checkpoint_*.pt"):
            mtime = f.stat().st_mtime
            checkpoints.append((mtime, f))

        # Sort by modification time (newest first)
        checkpoints.sort(reverse=True)

        # Remove old checkpoints
        for _, path in checkpoints[self.save_total_limit:]:
            path.unlink()
            print(f"[OK] Removed old checkpoint: {path.name}")


class TrainingResumer:
    """Handles training resumption from checkpoint."""

    def __init__(self, checkpoint_path):
        """
        CRITICAL #19: Initialize with checkpoint path, not manager object.

        Args:
            checkpoint_path: Path to checkpoint directory or file
        """
        self.checkpoint_path = checkpoint_path

    def load_checkpoint(self, model, optimizer=None, scheduler=None,
                       checkpoint_path=None, device='cuda'):
        """
        Load checkpoint into model, optimizer, and scheduler.

        CRITICAL #19: Add load_checkpoint() method for API compatibility.
        This is an alias for resume_training() with simplified return.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Optional path override
            device: Device to load checkpoint to
        """
        # Use instance checkpoint_path if no specific path provided
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        # Load checkpoint directly
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            self.checkpoint = None

        if self.checkpoint is None:
            print("[OK] No checkpoint found")
            return

        # Load model state with backward compatibility for new parameters
        # Use strict=False to handle checkpoints without new EOS predictor parameters
        missing_keys, unexpected_keys = model.load_state_dict(
            self.checkpoint['model_state_dict'], 
            strict=False
        )
        
        # Log any missing/unexpected keys for debugging
        if missing_keys:
            # Filter out expected new parameters for cleaner output
            new_params = ['eos_query', 'eos_key', 'eos_value', 'eos_head', 
                         'base_position_query', 'broadcast_pos_encoding']
            truly_missing = [k for k in missing_keys 
                            if not any(p in k for p in new_params)]
            if truly_missing:
                print(f"[WARNING] Missing keys in checkpoint: {truly_missing}")
            else:
                print(f"[OK] New parameters initialized from scratch: {len(missing_keys)} keys")
        
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys}")

        # Load optimizer state if provided
        if optimizer is not None:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

            # Move optimizer state to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        # Load scheduler state if provided
        if scheduler is not None and self.checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

    def resume_training(self, model, optimizer, scheduler,
                       checkpoint_path=None, device='cuda'):
        """Resume training from checkpoint."""

        # Use instance checkpoint_path if no specific path provided
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        # Load checkpoint directly
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = None

        if checkpoint is None:
            print("[OK] No checkpoint found, starting fresh training")
            return {
                'start_epoch': 0,
                'global_step': 0,
                'best_val_metric': 0.0,
                'best_epoch': 0,
                'early_stopping_counter': 0,
                'metrics_history': []
            }

        # Load model state with backward compatibility
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # Load scheduler state
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return {
            'start_epoch': checkpoint['epoch'] + 1,  # Start from next epoch
            'global_step': checkpoint['global_step'],
            'best_val_metric': checkpoint['best_val_metric'],
            'best_epoch': checkpoint['best_epoch'],
            'early_stopping_counter': checkpoint['early_stopping_counter'],
            'metrics_history': checkpoint.get('metrics_history', [])
        }
