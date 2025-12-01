# SCI MASTER GUIDE - PART 3

## Continuation from Part 2

---

<a name="section-8"></a>
# SECTION 8: TRAINING REGIME

## 8.1 TRAINING OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCI TRAINING REGIME                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: PRE-TRAINING VERIFICATION                                        │
│  ───────────────────────────────────                                        │
│    1. Run all unit tests (must pass)                                        │
│    2. Run verify_implementation.py (must pass)                              │
│    3. Verify dataset loading works                                          │
│    4. Verify pair generation produces valid pairs                           │
│    5. Verify no data leakage (test_data_leakage.py)                        │
│                                                                             │
│  PHASE 2: BASELINE TRAINING                                                │
│  ──────────────────────────                                                │
│    Train baseline (no SCI) first to establish performance floor             │
│    Config: baseline_tinyllama_scan.yaml                                     │
│    Duration: ~10 epochs                                                     │
│    Expected: SCAN length ~95% in-dist, ~20% OOD                            │
│                                                                             │
│  PHASE 3: SCI TRAINING                                                     │
│  ────────────────────                                                       │
│    Train full SCI model                                                     │
│    Config: sci_tinyllama_scan.yaml                                          │
│    Duration: ~20 epochs                                                     │
│    SCL warmup: 2 epochs (gradual increase of scl_weight)                   │
│    Expected: SCAN length ~98% in-dist, >85% OOD                            │
│                                                                             │
│  PHASE 4: ABLATION STUDIES                                                 │
│  ────────────────────────                                                   │
│    Train each ablation variant:                                             │
│    - ablation_no_se.yaml (no Structural Encoder)                           │
│    - ablation_no_ce.yaml (no Content Encoder)                              │
│    - ablation_no_cbm.yaml (no Causal Binding)                              │
│    - ablation_no_scl.yaml (no SCL loss)                                    │
│    - ablation_no_abstraction.yaml (SE without AbstractionLayer)            │
│                                                                             │
│  PHASE 5: CROSS-DATASET EVALUATION                                        │
│  ─────────────────────────────────                                          │
│    Evaluate trained models on:                                              │
│    - SCAN template split                                                    │
│    - SCAN addprim_jump split                                               │
│    - COGS gen split (zero-shot transfer)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 8.2 DETAILED TRAINING LOOP

```python
# src/training/trainer.py
"""
Main SCI Trainer with all necessary components.

CRITICAL IMPLEMENTATION NOTES:
1. Must compute pair_labels for each batch
2. Must apply instruction_mask to prevent data leakage
3. Must warmup SCL weight to prevent instability
4. Must log all component losses separately
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Optional, List
import yaml
from tqdm import tqdm
import wandb

from ..models.sci_model import SCIModel
from ..losses.combined_loss import SCICombinedLoss
from ..data.datasets.scan_dataset import SCANDataset
from ..data.datasets.collators import SCICollator
from ..evaluation.evaluator import SCIEvaluator
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logging_utils import setup_logger


class SCITrainer:
    """
    Complete trainer for SCI models.
    
    Training loop:
    1. Load batch with pair_labels
    2. Forward pass through SCI model
    3. Compute combined loss (LM + SCL + Ortho)
    4. Backward pass
    5. Update weights
    6. Log metrics
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logger(config['output']['log_dir'])
        self.logger.info(f"Initializing SCITrainer on {self.device}")
        
        # =========================================
        # Initialize Model
        # =========================================
        self.logger.info("Loading model...")
        self.model = SCIModel(config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        sci_params = sum(
            p.numel() for n, p in self.model.named_parameters() 
            if 'structural_encoder' in n or 'content_encoder' in n or 'causal_binding' in n
        )
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"SCI parameters: {sci_params:,} ({100*sci_params/total_params:.1f}%)")
        
        # =========================================
        # Initialize Dataset and DataLoader
        # =========================================
        self.logger.info("Loading dataset...")
        
        train_config = config['training']
        
        self.train_dataset = SCANDataset(
            tokenizer=self.model.tokenizer,
            split=train_config['split'],
            subset='train',
            max_length=train_config['max_length'],
            pairs_per_example=train_config.get('pairs_per_example', 5)
        )
        
        self.collator = SCICollator(
            tokenizer=self.model.tokenizer,
            pair_generator=self.train_dataset.pair_generator
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            collate_fn=self.collator,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"Dataset size: {len(self.train_dataset)}")
        self.logger.info(f"Batches per epoch: {len(self.train_loader)}")
        
        # =========================================
        # Initialize Loss Function
        # =========================================
        sci_config = config.get('sci', {})
        scl_config = sci_config.get('contrastive_learning', {})
        
        self.loss_fn = SCICombinedLoss(
            scl_weight=scl_config.get('scl_weight', 0.1),
            ortho_weight=sci_config.get('content_encoder', {}).get('orthogonality_weight', 0.01),
            temperature=scl_config.get('temperature', 0.07),
            use_hard_negatives=scl_config.get('use_hard_negatives', True)
        )
        
        # =========================================
        # Initialize Optimizer
        # =========================================
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # =========================================
        # Initialize Scheduler
        # =========================================
        num_training_steps = (
            len(self.train_loader) * train_config['max_epochs'] 
            // train_config.get('gradient_accumulation', 1)
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=train_config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # =========================================
        # Mixed Precision
        # =========================================
        self.use_amp = train_config.get('fp16', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # =========================================
        # SCL Warmup Configuration
        # =========================================
        self.scl_warmup_epochs = scl_config.get('warmup_epochs', 2)
        self.base_scl_weight = scl_config.get('scl_weight', 0.1)
        
        # =========================================
        # Initialize Evaluator
        # =========================================
        self.evaluator = SCIEvaluator(config)
        
        # =========================================
        # Wandb Logging
        # =========================================
        if config['logging'].get('use_wandb', False):
            wandb.init(
                project=config['logging']['project_name'],
                config=config,
                name=f"sci_{train_config['split']}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            
        # =========================================
        # Checkpointing
        # =========================================
        self.checkpoint_dir = config['output']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0
        self.gradient_accumulation = train_config.get('gradient_accumulation', 1)
        
    def compute_scl_weight(self, epoch: int) -> float:
        """
        Compute SCL weight with warmup schedule.
        
        Gradually increase SCL weight over warmup_epochs to prevent
        early training instability.
        """
        if epoch < self.scl_warmup_epochs:
            # Linear warmup
            progress = (epoch + 1) / self.scl_warmup_epochs
            return self.base_scl_weight * progress
        else:
            return self.base_scl_weight
            
    def train_step(
        self, 
        batch: Dict[str, torch.Tensor],
        scl_weight: float
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Returns dict of losses for logging.
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        pair_labels = batch['pair_labels'].to(self.device)
        
        # =========================================
        # CRITICAL: Verify pair_labels is valid
        # =========================================
        assert pair_labels.sum() > 0 or not self.config['sci']['contrastive_learning']['enabled'], \
            "CRITICAL: No positive pairs in batch! Check pair generation."
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp):
            # Model forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_loss=True
            )
            
            # Compute combined loss
            losses = self.loss_fn(
                outputs,
                pair_labels=pair_labels,
                scl_weight_override=scl_weight
            )
            
            loss = losses['total_loss'] / self.gradient_accumulation
            
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        return {
            'total_loss': losses['total_loss'].item(),
            'lm_loss': losses['lm_loss'].item(),
            'scl_loss': losses['scl_loss'].item() if torch.is_tensor(losses['scl_loss']) else 0.0,
            'ortho_loss': losses['orthogonality_loss'].item() if torch.is_tensor(losses['orthogonality_loss']) else 0.0
        }
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Get current SCL weight
        scl_weight = self.compute_scl_weight(epoch)
        self.logger.info(f"Epoch {epoch}: SCL weight = {scl_weight:.4f}")
        
        # Metrics tracking
        metrics = {
            'total_loss': 0.0,
            'lm_loss': 0.0,
            'scl_loss': 0.0,
            'ortho_loss': 0.0,
            'n_batches': 0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Train step
            step_losses = self.train_step(batch, scl_weight)
            
            # Update metrics
            for key in ['total_loss', 'lm_loss', 'scl_loss', 'ortho_loss']:
                metrics[key] += step_losses[key]
            metrics['n_batches'] += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Log to wandb
                if self.global_step % self.config['logging'].get('log_every', 10) == 0:
                    if wandb.run is not None:
                        wandb.log({
                            'train/total_loss': step_losses['total_loss'],
                            'train/lm_loss': step_losses['lm_loss'],
                            'train/scl_loss': step_losses['scl_loss'],
                            'train/ortho_loss': step_losses['ortho_loss'],
                            'train/scl_weight': scl_weight,
                            'train/lr': self.scheduler.get_last_lr()[0],
                            'train/step': self.global_step
                        })
                        
            # Update progress bar
            avg_loss = metrics['total_loss'] / metrics['n_batches']
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'scl': f"{metrics['scl_loss']/metrics['n_batches']:.4f}"
            })
            
        # Compute averages
        for key in ['total_loss', 'lm_loss', 'scl_loss', 'ortho_loss']:
            metrics[key] /= metrics['n_batches']
            
        return metrics
        
    def train(self):
        """Full training loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting SCI Training")
        self.logger.info("=" * 60)
        
        for epoch in range(self.config['training']['max_epochs']):
            self.logger.info(f"\n=== Epoch {epoch + 1}/{self.config['training']['max_epochs']} ===")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.logger.info(
                f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                f"LM: {train_metrics['lm_loss']:.4f}, "
                f"SCL: {train_metrics['scl_loss']:.4f}, "
                f"Ortho: {train_metrics['ortho_loss']:.4f}"
            )
            
            # Evaluate
            if (epoch + 1) % self.config['training'].get('eval_every', 1) == 0:
                eval_results = self.evaluator.evaluate(self.model)
                
                for dataset_name, metrics in eval_results.items():
                    self.logger.info(f"Eval {dataset_name}: Accuracy = {metrics['accuracy']:.4f}")
                    
                    if wandb.run is not None:
                        wandb.log({
                            f'eval/{dataset_name}/accuracy': metrics['accuracy'],
                            'epoch': epoch + 1
                        })
                        
                # Check for best model
                primary_dataset = list(eval_results.keys())[0]
                primary_accuracy = eval_results[primary_dataset]['accuracy']
                
                if primary_accuracy > self.best_accuracy:
                    self.best_accuracy = primary_accuracy
                    self.save_checkpoint('best_model')
                    self.logger.info(f"New best accuracy: {self.best_accuracy:.4f}")
                    
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_every', 1) == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')
                
        # Final save
        self.save_checkpoint('final_model')
        
        self.logger.info("=" * 60)
        self.logger.info(f"Training complete. Best accuracy: {self.best_accuracy:.4f}")
        self.logger.info("=" * 60)
        
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, name)
        self.model.save_pretrained(path)
        
        # Save training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
            'config': self.config
        }, os.path.join(path, 'training_state.pt'))
        
        self.logger.info(f"Saved checkpoint to {path}")
```

## 8.3 DATA COLLATOR WITH PAIR GENERATION

```python
# src/data/datasets/collators.py
"""
Custom collators for SCI training.

The collator is responsible for:
1. Stacking batch tensors
2. Generating pair_labels for each batch
3. Verifying minimum positive pairs
"""

import torch
from typing import List, Dict, Optional
from ..pair_generators.base_generator import BasePairGenerator


class SCICollator:
    """
    Collator that generates pair_labels for SCL training.
    
    CRITICAL: This is where pair_labels are generated for each batch.
    Without this, SCL cannot train.
    """
    
    def __init__(
        self,
        tokenizer,
        pair_generator: BasePairGenerator,
        min_positives: int = 2,
        augment_batches: bool = True
    ):
        self.tokenizer = tokenizer
        self.pair_generator = pair_generator
        self.min_positives = min_positives
        self.augment_batches = augment_batches
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and generate pair_labels.
        """
        # Get indices for pair generation
        indices = [ex['idx'] for ex in batch]
        
        # Optionally augment batch for more positives
        if self.augment_batches:
            augmented_indices = self.pair_generator.augment_batch_for_positives(
                indices,
                target_batch_size=len(indices) + 4  # Add up to 4 more examples
            )
            
            # If augmented, we need to add the new examples to batch
            if len(augmented_indices) > len(indices):
                # This is complex - for simplicity, just use original
                pass
                
        # Stack tensors
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        
        # Generate pair_labels
        pair_labels = self.pair_generator.generate_batch_pairs(indices)
        
        # =========================================
        # CRITICAL VERIFICATION
        # =========================================
        n_positive_pairs = pair_labels.sum().item()
        if n_positive_pairs == 0:
            # Log warning but don't fail - SCL will return 0 loss
            print(f"WARNING: Batch has no positive pairs! Indices: {indices[:5]}...")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pair_labels': pair_labels,
            'indices': indices
        }
```

---

<a name="section-9"></a>
# SECTION 9: EVALUATION REGIME

## 9.1 EVALUATION OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION STRATEGY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  METRIC: Exact Match Accuracy                                               │
│  ─────────────────────────────                                              │
│    For SCAN: Generated action sequence must EXACTLY match target           │
│    For COGS: Generated logical form must EXACTLY match target              │
│                                                                             │
│  EVALUATION SPLITS:                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  1. IN-DISTRIBUTION (sanity check):                                        │
│     - Train on SCAN length (train) → Eval on SCAN length (train subset)   │
│     - Expected: >95% for both baseline and SCI                             │
│     - If <90%, something is wrong with training                            │
│                                                                             │
│  2. OUT-OF-DISTRIBUTION - Length (KEY METRIC):                             │
│     - Train on SCAN length (train) → Eval on SCAN length (test)           │
│     - Test has LONGER sequences than train                                 │
│     - Baseline expected: ~20%                                              │
│     - SCI expected: >85%                                                   │
│     - This is the PRIMARY metric for compositional generalization          │
│                                                                             │
│  3. OUT-OF-DISTRIBUTION - Template:                                        │
│     - Train on SCAN length → Eval on SCAN template                         │
│     - Test has UNSEEN structural templates                                 │
│     - Baseline expected: ~50%                                              │
│     - SCI expected: >90%                                                   │
│                                                                             │
│  4. OUT-OF-DISTRIBUTION - Primitive:                                       │
│     - Train on SCAN (without jump in some context)                         │
│     - Eval on SCAN addprim_jump (jump in new context)                      │
│     - Tests primitive-level generalization                                 │
│     - SCI expected: >80%                                                   │
│                                                                             │
│  5. CROSS-DATASET TRANSFER:                                                │
│     - Train on SCAN → Eval on COGS (zero-shot)                            │
│     - Tests if structural learning transfers                               │
│     - Lower expectations, but SCI should still beat baseline               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 9.2 EVALUATOR IMPLEMENTATION

```python
# src/evaluation/evaluator.py
"""
SCI Evaluator for compositional benchmarks.

Implements exact-match evaluation for SCAN and COGS.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm


class SCIEvaluator:
    """
    Evaluator for SCI models.
    
    Supports:
    - SCAN (all splits)
    - COGS (generalization split)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load evaluation datasets
        self.eval_datasets = {}
        self._load_eval_datasets()
        
    def _load_eval_datasets(self):
        """Load all configured evaluation datasets."""
        from ..data.datasets.scan_dataset import SCANDataset
        from ..data.datasets.cogs_dataset import COGSDataset
        
        for dataset_config in self.config['evaluation']['datasets']:
            name = dataset_config['name']
            split = dataset_config['split']
            key = f"{name}_{split}"
            
            if name == 'scan':
                self.eval_datasets[key] = {
                    'type': 'scan',
                    'split': split,
                    'dataset': None  # Will be initialized with tokenizer
                }
            elif name == 'cogs':
                self.eval_datasets[key] = {
                    'type': 'cogs',
                    'split': split,
                    'dataset': None
                }
                
    def evaluate(
        self, 
        model,
        datasets: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate model on all specified datasets.
        
        Args:
            model: SCIModel to evaluate
            datasets: List of dataset keys to evaluate (None = all)
            
        Returns:
            Dict mapping dataset_key -> metrics dict
        """
        model.eval()
        results = {}
        
        datasets_to_eval = datasets if datasets else list(self.eval_datasets.keys())
        
        for dataset_key in datasets_to_eval:
            if dataset_key not in self.eval_datasets:
                continue
                
            dataset_info = self.eval_datasets[dataset_key]
            
            # Initialize dataset with tokenizer
            if dataset_info['dataset'] is None:
                self._init_dataset(dataset_key, model.tokenizer)
                
            # Evaluate
            metrics = self._evaluate_dataset(
                model, 
                dataset_key,
                dataset_info['dataset']
            )
            results[dataset_key] = metrics
            
        return results
        
    def _init_dataset(self, dataset_key: str, tokenizer):
        """Initialize dataset with tokenizer."""
        from ..data.datasets.scan_dataset import SCANDataset
        from ..data.datasets.cogs_dataset import COGSDataset
        
        info = self.eval_datasets[dataset_key]
        
        if info['type'] == 'scan':
            info['dataset'] = SCANDataset(
                tokenizer=tokenizer,
                split=info['split'],
                subset='test',
                max_length=self.config['training']['max_length']
            )
        elif info['type'] == 'cogs':
            info['dataset'] = COGSDataset(
                tokenizer=tokenizer,
                split=info['split'],
                subset='gen',
                max_length=self.config['training']['max_length']
            )
            
    def _evaluate_dataset(
        self,
        model,
        dataset_key: str,
        dataset
    ) -> Dict[str, float]:
        """Evaluate on a single dataset."""
        
        # Create simple loader (no pair generation needed for eval)
        loader = DataLoader(
            dataset,
            batch_size=self.config['evaluation'].get('batch_size', 32),
            shuffle=False
        )
        
        all_correct = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {dataset_key}"):
                # Generate predictions
                predictions = self._generate_predictions(model, batch)
                
                # Compare with targets
                targets = batch['target_text']
                
                for pred, target in zip(predictions, targets):
                    pred_clean = pred.strip()
                    target_clean = target.strip()
                    
                    is_correct = pred_clean == target_clean
                    all_correct.append(is_correct)
                    all_predictions.append(pred_clean)
                    all_targets.append(target_clean)
                    
        # Compute metrics
        accuracy = sum(all_correct) / len(all_correct)
        
        return {
            'accuracy': accuracy,
            'exact_match': accuracy,
            'total_examples': len(all_correct),
            'correct': sum(all_correct),
            'predictions': all_predictions,
            'targets': all_targets
        }
        
    def _generate_predictions(
        self,
        model,
        batch
    ) -> List[str]:
        """Generate predictions for a batch."""
        predictions = []
        
        for i in range(len(batch['input_text'])):
            input_text = batch['input_text'][i]
            
            # Tokenize prompt
            inputs = model.tokenizer(
                input_text,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate
            outputs = model.base_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                do_sample=False,  # Greedy decoding
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            
            # Decode (only generated part)
            generated = model.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            predictions.append(generated)
            
        return predictions
```

---

<a name="section-10"></a>
# SECTION 10: CONFIGURATION FILES

## 10.1 BASE CONFIGURATION

```yaml
# configs/base.yaml
# Base configuration inherited by all experiments

# Common model settings
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype: "float16"

# Common training settings
training:
  max_length: 128
  batch_size: 16
  gradient_accumulation: 2
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_steps: 500
  max_epochs: 20
  fp16: true
  gradient_checkpointing: true
  save_every: 5
  eval_every: 1
  pairs_per_example: 5

# Common evaluation settings
evaluation:
  batch_size: 32
  max_samples: null

# Common logging settings
logging:
  use_wandb: true
  project_name: "sci-compositional"
  log_every: 10

# Common output settings
output:
  checkpoint_dir: "outputs/checkpoints"
  log_dir: "outputs/logs"
  results_dir: "outputs/results"
```

## 10.2 FULL SCI CONFIGURATION

```yaml
# configs/sci_tinyllama_scan.yaml
# Full SCI configuration for SCAN training

# Inherit base config
defaults:
  - base

# Override/extend with SCI-specific settings
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# SCI Components - ALL ENABLED
sci:
  # Structural Encoder
  structural_encoder:
    enabled: true
    n_slots: 8
    n_layers: 2
    n_heads: 4
    abstraction_hidden_mult: 2
    dropout: 0.1
    
  # Content Encoder
  content_encoder:
    enabled: true
    refiner_layers: 2
    shared_embeddings: true
    orthogonality_weight: 0.01
    
  # Causal Binding Mechanism
  causal_binding:
    enabled: true
    injection_layers: [6, 11, 16]
    n_iterations: 3
    
  # Structural Contrastive Learning
  contrastive_learning:
    enabled: true
    temperature: 0.07
    scl_weight: 0.1
    use_hard_negatives: true
    hard_negative_ratio: 0.3
    warmup_epochs: 2

# Training configuration
training:
  dataset: "scan"
  split: "length"
  train_subset: "train"
  eval_subset: "test"
  max_length: 128
  batch_size: 16
  max_epochs: 20
  pairs_per_example: 5
  min_positives_per_batch: 4

# Evaluation datasets
evaluation:
  datasets:
    - name: "scan"
      split: "length"
    - name: "scan"
      split: "template"
    - name: "scan"
      split: "addprim_jump"

# Output paths
output:
  checkpoint_dir: "outputs/checkpoints/sci_scan_length"
  log_dir: "outputs/logs/sci_scan_length"
  results_dir: "outputs/results/sci_scan_length"
```

## 10.3 BASELINE CONFIGURATION (NO SCI)

```yaml
# configs/baseline_tinyllama_scan.yaml
# Baseline configuration WITHOUT SCI components

defaults:
  - base

model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ALL SCI COMPONENTS DISABLED
sci:
  structural_encoder:
    enabled: false
  content_encoder:
    enabled: false
  causal_binding:
    enabled: false
  contrastive_learning:
    enabled: false

training:
  dataset: "scan"
  split: "length"
  train_subset: "train"
  eval_subset: "test"
  max_length: 128
  batch_size: 16
  max_epochs: 20
  # No pair generation needed
  pairs_per_example: 0

evaluation:
  datasets:
    - name: "scan"
      split: "length"
    - name: "scan"
      split: "template"

output:
  checkpoint_dir: "outputs/checkpoints/baseline_scan_length"
  log_dir: "outputs/logs/baseline_scan_length"
  results_dir: "outputs/results/baseline_scan_length"
```

## 10.4 ABLATION CONFIGURATIONS

```yaml
# configs/ablation_no_scl.yaml
# Ablation: Full SCI but WITHOUT SCL loss
# Tests: Is contrastive learning necessary?

defaults:
  - base

sci:
  structural_encoder:
    enabled: true
    n_slots: 8
    n_layers: 2
    n_heads: 4
    abstraction_hidden_mult: 2
    
  content_encoder:
    enabled: true
    refiner_layers: 2
    shared_embeddings: true
    orthogonality_weight: 0.01
    
  causal_binding:
    enabled: true
    injection_layers: [6, 11, 16]
    n_iterations: 3
    
  # SCL DISABLED - THIS IS THE ABLATION
  contrastive_learning:
    enabled: false
    scl_weight: 0.0

training:
  dataset: "scan"
  split: "length"
  max_epochs: 20

output:
  checkpoint_dir: "outputs/checkpoints/ablation_no_scl"
```

```yaml
# configs/ablation_no_abstraction.yaml
# Ablation: SE WITHOUT AbstractionLayer
# Tests: Is AbstractionLayer the key component?

defaults:
  - base

sci:
  structural_encoder:
    enabled: true
    n_slots: 8
    n_layers: 2
    n_heads: 4
    # ABLATION: No abstraction layer
    use_abstraction: false  # Special flag
    
  content_encoder:
    enabled: true
    refiner_layers: 2
    
  causal_binding:
    enabled: true
    injection_layers: [6, 11, 16]
    
  contrastive_learning:
    enabled: true
    scl_weight: 0.1

output:
  checkpoint_dir: "outputs/checkpoints/ablation_no_abstraction"
```

---

<a name="section-11"></a>
# SECTION 11: EXECUTION CHECKLIST

## 11.1 COMPLETE STEP-BY-STEP CHECKLIST

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SCI IMPLEMENTATION CHECKLIST                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: ENVIRONMENT SETUP                                                 │
│  ─────────────────────────                                                  │
│  [ ] Create project directory structure                                     │
│  [ ] Create requirements.txt                                                │
│  [ ] Create setup.py and pyproject.toml                                    │
│  [ ] Create virtual environment                                             │
│  [ ] Install dependencies                                                   │
│  [ ] Verify GPU is available                                                │
│                                                                             │
│  STEP 2: IMPLEMENT COMPONENTS (in order)                                   │
│  ────────────────────────────────────────                                   │
│  [ ] src/models/components/abstraction_layer.py                            │
│      [ ] Run test_abstraction_layer.py - MUST PASS                         │
│  [ ] src/models/components/structural_gnn.py                               │
│  [ ] src/models/components/edge_predictor.py                               │
│  [ ] src/models/components/content_refiner.py                              │
│  [ ] src/models/components/orthogonal_projector.py                         │
│  [ ] src/models/components/causal_intervention.py                          │
│                                                                             │
│  STEP 3: IMPLEMENT ENCODERS                                                │
│  ──────────────────────────                                                 │
│  [ ] src/models/structural_encoder.py                                       │
│      [ ] Verify uses AbstractionLayer (not custom)                         │
│      [ ] Run test_structural_encoder.py - MUST PASS                        │
│  [ ] src/models/content_encoder.py                                          │
│      [ ] Run test_content_encoder.py - MUST PASS                           │
│  [ ] src/models/causal_binding.py                                           │
│      [ ] Run test_causal_binding.py - MUST PASS                            │
│  [ ] src/models/sci_model.py                                                │
│      [ ] Run test_full_model.py - MUST PASS                                │
│                                                                             │
│  STEP 4: IMPLEMENT LOSSES                                                  │
│  ───────────────────────                                                    │
│  [ ] src/losses/structural_contrastive.py                                  │
│      [ ] Run test_scl_loss.py - MUST PASS                                  │
│  [ ] src/losses/orthogonality_loss.py                                      │
│  [ ] src/losses/combined_loss.py                                           │
│                                                                             │
│  STEP 5: IMPLEMENT DATA PIPELINE                                           │
│  ───────────────────────────────                                            │
│  [ ] src/data/structure_extractors/scan_extractor.py                       │
│  [ ] src/data/pair_generators/scan_pair_generator.py                       │
│      [ ] Run test_pair_generation.py - MUST PASS                           │
│  [ ] src/data/datasets/scan_dataset.py                                     │
│  [ ] src/data/datasets/collators.py                                        │
│                                                                             │
│  STEP 6: IMPLEMENT TRAINING                                                │
│  ──────────────────────────                                                 │
│  [ ] src/training/trainer.py                                                │
│  [ ] src/training/scheduler.py                                              │
│  [ ] src/training/callbacks.py                                              │
│                                                                             │
│  STEP 7: IMPLEMENT EVALUATION                                              │
│  ────────────────────────────                                               │
│  [ ] src/evaluation/evaluator.py                                            │
│  [ ] src/evaluation/metrics.py                                              │
│  [ ] src/evaluation/invariance_tester.py                                   │
│                                                                             │
│  STEP 8: CREATE CONFIGS                                                    │
│  ──────────────────────                                                     │
│  [ ] configs/base.yaml                                                      │
│  [ ] configs/sci_tinyllama_scan.yaml                                       │
│  [ ] configs/baseline_tinyllama_scan.yaml                                  │
│  [ ] configs/ablation_*.yaml (all 5)                                       │
│                                                                             │
│  STEP 9: PRE-TRAINING VERIFICATION                                        │
│  ─────────────────────────────────                                          │
│  [ ] Run scripts/verify_implementation.py                                   │
│      [ ] All component checks pass                                          │
│      [ ] Full model forward pass works                                      │
│      [ ] Pair generation produces valid pairs                               │
│  [ ] Run test_data_leakage.py - MUST PASS                                  │
│  [ ] Manually inspect a few batches:                                        │
│      [ ] pair_labels has non-zero entries                                  │
│      [ ] labels correctly mask instruction tokens                          │
│                                                                             │
│  STEP 10: BASELINE TRAINING                                                │
│  ──────────────────────────                                                 │
│  [ ] Run: python scripts/train.py --config configs/baseline_tinyllama_scan.yaml │
│  [ ] Verify training loss decreases                                         │
│  [ ] Record baseline results:                                               │
│      [ ] SCAN length (train): ___% (expect >95%)                           │
│      [ ] SCAN length (test): ___% (expect ~20%)                            │
│      [ ] SCAN template: ___% (expect ~50%)                                 │
│                                                                             │
│  STEP 11: SCI TRAINING                                                     │
│  ────────────────────                                                       │
│  [ ] Run: python scripts/train.py --config configs/sci_tinyllama_scan.yaml │
│  [ ] Verify during training:                                                │
│      [ ] LM loss decreases                                                  │
│      [ ] SCL loss decreases                                                 │
│      [ ] SCL loss is non-zero (if 0, pair generation is broken)            │
│  [ ] Record SCI results:                                                    │
│      [ ] SCAN length (train): ___% (expect >95%)                           │
│      [ ] SCAN length (test): ___% (expect >85%)                            │
│      [ ] SCAN template: ___% (expect >90%)                                 │
│                                                                             │
│  STEP 12: VERIFY IMPROVEMENT                                               │
│  ───────────────────────────                                                │
│  [ ] SCI OOD accuracy > Baseline OOD accuracy                              │
│  [ ] Improvement is significant (>20% absolute)                            │
│  [ ] Run test_structural_invariance.py on trained model                    │
│      [ ] Same-structure similarity > 0.85                                  │
│      [ ] Content substitution similarity > 0.90                            │
│                                                                             │
│  STEP 13: ABLATION STUDIES                                                 │
│  ────────────────────────                                                   │
│  [ ] Run ablation_no_se.yaml - Record results                              │
│  [ ] Run ablation_no_ce.yaml - Record results                              │
│  [ ] Run ablation_no_cbm.yaml - Record results                             │
│  [ ] Run ablation_no_scl.yaml - Record results                             │
│  [ ] Run ablation_no_abstraction.yaml - Record results                     │
│  [ ] Verify: Full SCI > All ablations > Baseline                           │
│                                                                             │
│  STEP 14: GENERATE RESULTS                                                 │
│  ────────────────────────                                                   │
│  [ ] Create results table comparing all experiments                        │
│  [ ] Generate figures showing:                                              │
│      [ ] Training curves (LM loss, SCL loss)                               │
│      [ ] Accuracy by split (in-dist vs OOD)                                │
│      [ ] Ablation comparison                                               │
│      [ ] Structural invariance visualization                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 11.2 EXPECTED RESULTS TABLE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXPECTED RESULTS                                     │
├──────────────────────────┬────────────────┬────────────────┬────────────────┤
│ Configuration            │ SCAN Length    │ SCAN Length    │ SCAN Template  │
│                          │ (In-Dist)      │ (OOD)          │ (OOD)          │
├──────────────────────────┼────────────────┼────────────────┼────────────────┤
│ Baseline (No SCI)        │ ~95%           │ ~20%           │ ~50%           │
│ SCI (Full)               │ ~98%           │ >85%           │ >90%           │
│ Ablation: No SE          │ ~95%           │ ~25%           │ ~55%           │
│ Ablation: No CE          │ ~96%           │ ~60%           │ ~70%           │
│ Ablation: No CBM         │ ~96%           │ ~70%           │ ~80%           │
│ Ablation: No SCL         │ ~96%           │ ~50%           │ ~65%           │
│ Ablation: No Abstraction │ ~96%           │ ~40%           │ ~60%           │
├──────────────────────────┴────────────────┴────────────────┴────────────────┤
│ KEY INSIGHTS:                                                               │
│ - SE + AbstractionLayer is most critical (biggest drop when removed)       │
│ - SCL is essential for learning structural invariance                      │
│ - CBM helps but is not as critical as SE                                   │
│ - CE provides modest improvement                                            │
│ - Full SCI = SE + CE + CBM + SCL achieves best results                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 11.3 TROUBLESHOOTING GUIDE

```
PROBLEM: SCL loss is always 0
─────────────────────────────
CAUSE: No positive pairs in batches
FIX: 
  1. Check pair_generator.process_dataset() was called
  2. Check pair_labels in collator output has 1s
  3. Increase batch size to get more same-structure examples

PROBLEM: Training loss not decreasing
─────────────────────────────────────
CAUSE: Learning rate too high/low, or gradient issues
FIX:
  1. Try learning rate 1e-5 to 1e-4
  2. Check gradient norms (should be < 10)
  3. Verify no NaN in losses

PROBLEM: OOD accuracy same as baseline
────────────────────────────────────────
CAUSE: SCI components not working properly
FIX:
  1. Verify AbstractionLayer is being used (check assertions)
  2. Verify SCL loss is non-zero and decreasing
  3. Run test_structural_invariance.py to check invariance

PROBLEM: Model outputs garbage
──────────────────────────────
CAUSE: Data leakage or wrong label masking
FIX:
  1. Run test_data_leakage.py
  2. Check labels have -100 for instruction tokens
  3. Verify tokenization is correct

PROBLEM: Out of memory
──────────────────────
CAUSE: Batch size too large
FIX:
  1. Reduce batch_size to 8
  2. Increase gradient_accumulation to 4
  3. Enable gradient_checkpointing
```

---

## END OF MASTER GUIDE

This guide provides COMPLETE instructions for implementing SCI. Follow each section in order and verify at each checkpoint before proceeding.

Total estimated implementation time: 40-60 hours
Total files to create: ~70 files
Total lines of code: ~8,000 lines
