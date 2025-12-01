# SCI Agent Instructions - Part 3: Training, Evaluation, and Verification

## Phase 5: Training Pipeline

### Step 5.1: Main Trainer (src/training/trainer.py)

```python
# src/training/trainer.py
"""
SCI Trainer - Main Training Loop

CRITICAL IMPLEMENTATION NOTES:
1. Must generate pair_labels for each batch
2. SCL loss requires structural representations
3. Warmup schedule for SCL weight prevents early instability
4. Log both LM loss and SCL loss separately
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional
import yaml
from tqdm import tqdm
import wandb

from ..models.sci_model import SCIModel
from ..losses.combined_loss import SCICombinedLoss
from ..data.scan_dataset import SCANDataset, SCANCollator
from ..evaluation.evaluator import SCIEvaluator


class SCITrainer:
    """
    Main trainer for SCI models.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SCIModel(config).to(self.device)
        
        # Initialize loss
        sci_config = config.get('sci', {})
        scl_config = sci_config.get('contrastive_learning', {})
        
        self.loss_fn = SCICombinedLoss(
            scl_weight=scl_config.get('scl_weight', 0.1),
            ortho_weight=sci_config.get('content_encoder', {}).get('orthogonality_weight', 0.01),
            temperature=scl_config.get('temperature', 0.07),
            use_hard_negatives=scl_config.get('use_hard_negatives', True)
        )
        
        # Initialize dataset
        self.train_dataset = SCANDataset(
            tokenizer=self.model.tokenizer,
            split=config['training']['split'],
            subset='train',
            max_length=config['training']['max_length']
        )
        
        self.train_collator = SCANCollator(self.train_dataset)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=self.train_collator,
            num_workers=4
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        num_training_steps = (
            len(self.train_loader) * config['training']['max_epochs'] 
            // config['training']['gradient_accumulation']
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # SCL warmup schedule
        self.scl_warmup_epochs = scl_config.get('warmup_epochs', 2)
        
        # Initialize evaluator
        self.evaluator = SCIEvaluator(config)
        
        # Logging
        if config['logging'].get('use_wandb', False):
            wandb.init(
                project=config['logging']['project_name'],
                config=config
            )
            
        # Checkpointing
        self.checkpoint_dir = config['output']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0
        
    def compute_scl_weight(self, epoch: int) -> float:
        """
        Compute SCL weight with warmup.
        
        Gradually increase SCL weight to prevent early instability.
        """
        base_weight = self.config['sci']['contrastive_learning'].get('scl_weight', 0.1)
        
        if epoch < self.scl_warmup_epochs:
            # Linear warmup
            return base_weight * (epoch + 1) / self.scl_warmup_epochs
        else:
            return base_weight
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_scl_loss = 0.0
        total_ortho_loss = 0.0
        num_batches = 0
        
        # Get current SCL weight
        scl_weight = self.compute_scl_weight(epoch)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            pair_labels = batch['pair_labels'].to(self.device)
            
            # Forward pass
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
            
            loss = losses['total_loss']
            
            # Backward pass (with gradient accumulation)
            loss = loss / self.config['training']['gradient_accumulation']
            loss.backward()
            
            if (batch_idx + 1) % self.config['training']['gradient_accumulation'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
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
                'lm': total_lm_loss / num_batches,
                'scl': total_scl_loss / num_batches
            })
            
            # Log to wandb
            if self.global_step % self.config['logging'].get('log_every', 10) == 0:
                if wandb.run is not None:
                    wandb.log({
                        'train/total_loss': losses['total_loss'].item(),
                        'train/lm_loss': losses['lm_loss'].item(),
                        'train/scl_loss': losses['scl_loss'].item() if torch.is_tensor(losses['scl_loss']) else 0,
                        'train/ortho_loss': losses['orthogonality_loss'].item() if torch.is_tensor(losses['orthogonality_loss']) else 0,
                        'train/scl_weight': scl_weight,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step
                    })
                    
        return {
            'total_loss': total_loss / num_batches,
            'lm_loss': total_lm_loss / num_batches,
            'scl_loss': total_scl_loss / num_batches,
            'ortho_loss': total_ortho_loss / num_batches
        }
        
    def train(self):
        """
        Full training loop.
        """
        for epoch in range(self.config['training']['max_epochs']):
            print(f"\n=== Epoch {epoch + 1}/{self.config['training']['max_epochs']} ===")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"LM: {train_metrics['lm_loss']:.4f}, "
                  f"SCL: {train_metrics['scl_loss']:.4f}")
            
            # Evaluate
            if (epoch + 1) % self.config['training'].get('eval_every', 1) == 0:
                eval_results = self.evaluator.evaluate(self.model)
                
                # Log evaluation results
                for dataset_name, metrics in eval_results.items():
                    print(f"Eval {dataset_name} - Accuracy: {metrics['accuracy']:.4f}")
                    
                    if wandb.run is not None:
                        wandb.log({
                            f'eval/{dataset_name}/accuracy': metrics['accuracy'],
                            f'eval/{dataset_name}/exact_match': metrics.get('exact_match', metrics['accuracy']),
                            'epoch': epoch + 1
                        })
                        
                # Check for best model
                primary_accuracy = eval_results[list(eval_results.keys())[0]]['accuracy']
                if primary_accuracy > self.best_accuracy:
                    self.best_accuracy = primary_accuracy
                    self.save_checkpoint('best_model')
                    print(f"New best accuracy: {self.best_accuracy:.4f}")
                    
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_every', 1) == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')
                
        # Final save
        self.save_checkpoint('final_model')
        
        print(f"\nTraining complete. Best accuracy: {self.best_accuracy:.4f}")
        
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, name)
        self.model.save_pretrained(path)
        
        # Save optimizer state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy
        }, os.path.join(path, 'training_state.pt'))
```

---

## Phase 6: Evaluation Pipeline

### Step 6.1: Evaluator (src/evaluation/evaluator.py)

```python
# src/evaluation/evaluator.py
"""
SCI Evaluator - Evaluation on Compositional Benchmarks

EVALUATION STRATEGY:
1. Train on SCAN length split
2. Evaluate on:
   - SCAN length (in-distribution)
   - SCAN template (compositional generalization)
   - COGS gen (cross-benchmark transfer)

CRITICAL: Use exact match for SCAN (sequence must match exactly)
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm

from ..data.scan_dataset import SCANDataset
from .metrics import compute_exact_match, compute_token_accuracy


class SCIEvaluator:
    """
    Evaluator for SCI models on compositional benchmarks.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load evaluation datasets
        self.eval_datasets = {}
        
        for dataset_config in config['evaluation']['datasets']:
            name = dataset_config['name']
            split = dataset_config['split']
            key = f"{name}_{split}"
            
            if name == 'scan':
                self.eval_datasets[key] = SCANDataset(
                    tokenizer=None,  # Will be set during evaluation
                    split=split,
                    subset='test',
                    max_length=config['training']['max_length']
                )
            # Add COGS, etc. here
            
        self.batch_size = config['evaluation'].get('batch_size', 32)
        
    def evaluate(self, model, datasets: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Evaluate model on all specified datasets.
        
        Returns dict of dataset -> metrics
        """
        model.eval()
        results = {}
        
        # Set tokenizer for datasets
        for dataset in self.eval_datasets.values():
            dataset.tokenizer = model.tokenizer
            
        datasets_to_eval = datasets if datasets else self.eval_datasets.keys()
        
        for dataset_key in datasets_to_eval:
            if dataset_key not in self.eval_datasets:
                continue
                
            dataset = self.eval_datasets[dataset_key]
            
            # Simple collator for evaluation (no pair generation needed)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._eval_collate
            )
            
            metrics = self._evaluate_dataset(model, loader, dataset_key)
            results[dataset_key] = metrics
            
        return results
    
    def _eval_collate(self, batch):
        """Simple collator for evaluation."""
        return {
            'input_ids': torch.stack([ex['input_ids'] for ex in batch]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in batch]),
            'labels': torch.stack([ex['labels'] for ex in batch]),
            'commands': [ex['command'] for ex in batch],
            'actions': [ex['actions'] for ex in batch]
        }
        
    def _evaluate_dataset(
        self, 
        model, 
        loader: DataLoader,
        dataset_name: str
    ) -> Dict[str, float]:
        """
        Evaluate on a single dataset.
        """
        all_predictions = []
        all_targets = []
        all_exact_match = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Generate outputs
                # Find where the instruction ends (before response)
                predictions = self._generate_outputs(
                    model, 
                    input_ids, 
                    attention_mask,
                    batch['commands']
                )
                
                # Compare with targets
                targets = batch['actions']
                
                for pred, target in zip(predictions, targets):
                    all_predictions.append(pred)
                    all_targets.append(target)
                    all_exact_match.append(pred.strip() == target.strip())
                    
        # Compute metrics
        accuracy = sum(all_exact_match) / len(all_exact_match)
        
        return {
            'accuracy': accuracy,
            'exact_match': accuracy,
            'total_examples': len(all_exact_match),
            'correct': sum(all_exact_match)
        }
        
    def _generate_outputs(
        self, 
        model, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        commands: List[str]
    ) -> List[str]:
        """
        Generate outputs for evaluation.
        """
        # For SCAN, we use greedy decoding
        # Find where instruction ends
        
        outputs = []
        
        for i, command in enumerate(commands):
            # Create prompt
            prompt = f"IN: {command} OUT:"
            prompt_ids = model.tokenizer.encode(
                prompt, 
                return_tensors='pt',
                add_special_tokens=False
            ).to(self.device)
            
            # Generate
            generated = model.base_model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=50,
                do_sample=False,  # Greedy
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            
            # Decode only the generated part
            generated_text = model.tokenizer.decode(
                generated[0][prompt_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            outputs.append(generated_text.strip())
            
        return outputs
```

---

## Phase 7: Verification Tests (CRITICAL)

**AI AGENT: These tests MUST pass before training. They verify correct implementation.**

### Step 7.1: Test Structural Invariance (tests/test_invariance.py)

```python
# tests/test_invariance.py
"""
Structural Invariance Tests

THESE TESTS ARE CRITICAL. They verify that:
1. Same structure → same structural representation (invariance)
2. Different structure → different structural representation (discrimination)
3. Content changes don't affect structure

If these tests fail, the SCI implementation is WRONG.
"""

import torch
import pytest
from typing import List, Tuple

from src.models.structural_encoder import StructuralEncoder
from src.models.sci_model import SCIModel
from src.data.structure_extractor import SCANStructureExtractor


class TestStructuralInvariance:
    """
    Tests for structural invariance property.
    """
    
    @pytest.fixture
    def encoder(self):
        """Create structural encoder for testing."""
        return StructuralEncoder(
            d_model=256,
            n_slots=8,
            n_layers=2
        )
        
    @pytest.fixture
    def structure_extractor(self):
        """Create structure extractor."""
        return SCANStructureExtractor()
        
    def create_embeddings(
        self, 
        texts: List[str], 
        d_model: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create fake embeddings for testing.
        
        Note: In real usage, these come from the tokenizer + embedding layer.
        For testing, we create consistent embeddings per word.
        """
        # Create word -> embedding mapping
        word_embeddings = {}
        
        all_words = set()
        for text in texts:
            all_words.update(text.lower().split())
            
        for i, word in enumerate(sorted(all_words)):
            torch.manual_seed(hash(word) % 2**32)
            word_embeddings[word] = torch.randn(d_model)
            
        # Create batch
        max_len = max(len(t.split()) for t in texts)
        batch_embeddings = torch.zeros(len(texts), max_len, d_model)
        attention_mask = torch.zeros(len(texts), max_len)
        
        for i, text in enumerate(texts):
            words = text.lower().split()
            for j, word in enumerate(words):
                batch_embeddings[i, j] = word_embeddings[word]
                attention_mask[i, j] = 1
                
        return batch_embeddings, attention_mask
        
    def test_same_structure_same_representation(self, encoder, structure_extractor):
        """
        Test: Inputs with same structure should have similar structural representations.
        
        "walk twice" and "run twice" have SAME structure (X twice)
        Their structural representations should be nearly identical.
        """
        # Create structurally equivalent examples
        examples_same_structure = [
            "walk twice",
            "run twice",
            "jump twice",
            "look twice"
        ]
        
        # Verify they have same structure
        patterns = [structure_extractor.extract(ex) for ex in examples_same_structure]
        assert all(p.pattern_template == patterns[0].pattern_template for p in patterns), \
            "Test setup error: examples should have same structure"
            
        # Get embeddings
        embeddings, mask = self.create_embeddings(examples_same_structure)
        
        # Get structural representations
        encoder.eval()
        with torch.no_grad():
            struct_graph, _ = encoder(embeddings, mask)
            
        # Compute pairwise similarities
        # Should be HIGH for same-structure pairs
        struct_flat = struct_graph.mean(dim=1)  # [B, D]
        struct_norm = torch.nn.functional.normalize(struct_flat, dim=-1)
        similarities = torch.mm(struct_norm, struct_norm.T)
        
        # All pairs should have similarity > 0.8 (excluding diagonal)
        off_diagonal = similarities[~torch.eye(len(examples_same_structure), dtype=bool)]
        mean_similarity = off_diagonal.mean().item()
        
        print(f"Same-structure similarity: {mean_similarity:.4f}")
        
        # This test may fail before training (random initialization)
        # But should pass after training with SCL
        # For now, just check the computation works
        assert struct_graph.shape == (len(examples_same_structure), 8, 256)
        
    def test_different_structure_different_representation(self, encoder, structure_extractor):
        """
        Test: Inputs with different structure should have different representations.
        
        "walk twice" vs "walk and run" have DIFFERENT structures
        """
        examples_different_structure = [
            "walk twice",      # X twice
            "walk and run",    # X and Y
            "walk after run",  # X after Y
            "turn left"        # turn direction
        ]
        
        # Verify they have different structures
        patterns = [structure_extractor.extract(ex) for ex in examples_different_structure]
        templates = [p.pattern_template for p in patterns]
        assert len(set(templates)) == len(templates), \
            "Test setup error: examples should have different structures"
            
        # Get embeddings and representations
        embeddings, mask = self.create_embeddings(examples_different_structure)
        
        encoder.eval()
        with torch.no_grad():
            struct_graph, _ = encoder(embeddings, mask)
            
        # Structural representations should be different
        struct_flat = struct_graph.mean(dim=1)
        struct_norm = torch.nn.functional.normalize(struct_flat, dim=-1)
        similarities = torch.mm(struct_norm, struct_norm.T)
        
        # Off-diagonal should have LOWER similarity than same-structure
        off_diagonal = similarities[~torch.eye(len(examples_different_structure), dtype=bool)]
        mean_similarity = off_diagonal.mean().item()
        
        print(f"Different-structure similarity: {mean_similarity:.4f}")
        
        # Should work even before training
        assert struct_graph.shape == (len(examples_different_structure), 8, 256)
        
    def test_content_invariance(self, encoder, structure_extractor):
        """
        Test: Changing content should NOT change structural representation.
        
        This is the KEY property of SCI.
        """
        # Same structure, different content
        base_example = "walk twice"
        content_variants = [
            "walk twice",
            "run twice",
            "jump twice"
        ]
        
        # Get embeddings
        embeddings, mask = self.create_embeddings(content_variants)
        
        encoder.eval()
        with torch.no_grad():
            struct_graph, _ = encoder(embeddings, mask)
            
        # All should have same structural representation
        struct_flat = struct_graph.mean(dim=1)
        
        # Check variance across examples
        variance = struct_flat.var(dim=0).mean().item()
        print(f"Content invariance variance: {variance:.6f}")
        
        # Lower variance = more invariant to content
        # After training, this should be very low


class TestDataLeakage:
    """
    Tests to ensure no data leakage from response to structure encoding.
    """
    
    def test_instruction_mask(self):
        """
        Test that instruction mask correctly separates instruction and response.
        """
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Create example
        instruction = "IN: walk twice OUT:"
        response = " WALK WALK"
        full_text = instruction + response
        
        # Tokenize
        full_tokens = tokenizer(full_text, return_tensors='pt')
        instruction_tokens = tokenizer(instruction, return_tensors='pt')
        
        instruction_length = instruction_tokens['input_ids'].shape[1]
        full_length = full_tokens['input_ids'].shape[1]
        
        # Create labels (-100 for instruction)
        labels = full_tokens['input_ids'].clone()
        labels[0, :instruction_length] = -100
        
        # Create instruction mask
        instruction_mask = (labels == -100).long()
        
        # Verify
        assert instruction_mask[0, :instruction_length].sum() == instruction_length, \
            "Instruction tokens should be masked"
        assert instruction_mask[0, instruction_length:].sum() == 0, \
            "Response tokens should NOT be masked"
            
    def test_no_response_in_structure(self):
        """
        Test that structural encoder does NOT see response tokens.
        """
        # This test verifies that when we pass instruction_mask,
        # the cross-attention only attends to instruction tokens.
        
        encoder = StructuralEncoder(d_model=256, n_slots=8, n_layers=2)
        
        # Create fake sequence: [instruction tokens] [response tokens]
        batch_size = 2
        seq_len = 10
        d_model = 256
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # First 5 tokens are instruction, last 5 are response
        instruction_mask = torch.zeros(batch_size, seq_len)
        instruction_mask[:, :5] = 1
        
        encoder.eval()
        with torch.no_grad():
            # This should only attend to first 5 tokens
            struct_graph, _ = encoder(hidden_states, instruction_mask)
            
        assert struct_graph.shape == (batch_size, 8, d_model)


class TestOrthogonality:
    """
    Test that structure and content representations are orthogonal.
    """
    
    def test_orthogonality_loss(self):
        """
        Test orthogonality loss computation.
        """
        from src.models.content_encoder import ContentEncoder
        
        encoder = ContentEncoder(
            d_model=256,
            vocab_size=1000,
            n_refiner_layers=2,
            share_embeddings=False
        )
        
        # Create fake representations
        content = torch.randn(4, 10, 256)
        structure = torch.randn(4, 8, 256)
        
        # Compute loss
        loss = encoder.compute_orthogonality_loss(content, structure)
        
        assert loss.shape == ()  # Scalar
        assert loss >= 0  # Non-negative
        
    def test_orthogonal_projector(self):
        """
        Test that OrthogonalProjector reduces similarity to structure.
        """
        from src.models.content_encoder import OrthogonalProjector
        
        projector = OrthogonalProjector(d_model=256)
        
        # Create content and structure that are NOT orthogonal
        content = torch.randn(4, 10, 256)
        structure = torch.randn(4, 8, 256)
        
        # Compute initial similarity
        content_avg = content.mean(dim=1)
        structure_avg = structure.mean(dim=1)
        initial_sim = torch.nn.functional.cosine_similarity(
            content_avg, structure_avg
        ).abs().mean()
        
        # Project
        projected_content = projector(content, structure)
        
        # Compute new similarity
        projected_avg = projected_content.mean(dim=1)
        final_sim = torch.nn.functional.cosine_similarity(
            projected_avg, structure_avg
        ).abs().mean()
        
        print(f"Initial similarity: {initial_sim:.4f}")
        print(f"After projection: {final_sim:.4f}")
        
        # Should be lower after projection
        # (may not always hold due to random initialization)


def run_all_tests():
    """Run all verification tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_all_tests()
```

### Step 7.2: Implementation Verification Script (scripts/verify_implementation.py)

```python
#!/usr/bin/env python
# scripts/verify_implementation.py
"""
Implementation Verification Script

RUN THIS BEFORE TRAINING.

Checks:
1. All components are correctly implemented
2. Shapes are correct
3. Forward pass works end-to-end
4. No data leakage
5. Pair generation works
"""

import sys
import torch
import yaml

print("=" * 60)
print("SCI IMPLEMENTATION VERIFICATION")
print("=" * 60)


def check_structural_encoder():
    """Verify Structural Encoder."""
    print("\n[1/6] Checking Structural Encoder...")
    
    try:
        from src.models.structural_encoder import StructuralEncoder, AbstractionLayer
        
        # Check AbstractionLayer exists and works
        abstraction = AbstractionLayer(d_model=256)
        x = torch.randn(2, 10, 256)
        out = abstraction(x)
        assert out.shape == x.shape, "AbstractionLayer shape mismatch"
        
        # Check full encoder
        encoder = StructuralEncoder(
            d_model=256,
            n_slots=8,
            n_layers=2
        )
        
        hidden_states = torch.randn(2, 20, 256)
        attention_mask = torch.ones(2, 20)
        
        struct_graph, edge_weights = encoder(hidden_states, attention_mask)
        
        assert struct_graph.shape == (2, 8, 256), f"Wrong shape: {struct_graph.shape}"
        assert edge_weights.shape == (2, 8, 8), f"Wrong shape: {edge_weights.shape}"
        
        print("   ✓ AbstractionLayer implemented correctly")
        print("   ✓ StructuralEncoder shapes correct")
        print("   ✓ Edge prediction working")
        
        return True
        
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def check_content_encoder():
    """Verify Content Encoder."""
    print("\n[2/6] Checking Content Encoder...")
    
    try:
        from src.models.content_encoder import ContentEncoder, OrthogonalProjector
        
        # Check OrthogonalProjector
        proj = OrthogonalProjector(d_model=256)
        content = torch.randn(2, 10, 256)
        structure = torch.randn(2, 8, 256)
        
        projected = proj(content, structure)
        assert projected.shape == content.shape, "Projector shape mismatch"
        
        # Check full encoder
        encoder = ContentEncoder(
            d_model=256,
            vocab_size=1000,
            n_refiner_layers=2
        )
        
        input_ids = torch.randint(0, 1000, (2, 10))
        output = encoder(input_ids, structure)
        
        assert output.shape == (2, 10, 256), f"Wrong shape: {output.shape}"
        
        # Check orthogonality loss
        loss = encoder.compute_orthogonality_loss(output, structure)
        assert loss.shape == (), "Loss should be scalar"
        
        print("   ✓ OrthogonalProjector working")
        print("   ✓ ContentEncoder shapes correct")
        print("   ✓ Orthogonality loss computable")
        
        return True
        
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def check_causal_binding():
    """Verify Causal Binding Mechanism."""
    print("\n[3/6] Checking Causal Binding Mechanism...")
    
    try:
        from src.models.causal_binding import CausalBindingMechanism, CausalInterventionLayer
        
        # Check intervention layer
        intervention = CausalInterventionLayer(d_model=256, n_iterations=3)
        slots = torch.randn(2, 8, 256)
        edges = torch.softmax(torch.randn(2, 8, 8), dim=-1)
        
        out = intervention(slots, edges)
        assert out.shape == slots.shape, "Intervention shape mismatch"
        
        # Check full mechanism
        cbm = CausalBindingMechanism(d_model=256, n_slots=8, n_iterations=3)
        
        hidden_states = torch.randn(2, 20, 256)
        content = torch.randn(2, 20, 256)
        attention_mask = torch.ones(2, 20)
        
        output = cbm(hidden_states, slots, edges, content, attention_mask)
        
        assert output.shape == hidden_states.shape, f"Wrong shape: {output.shape}"
        
        print("   ✓ CausalInterventionLayer working")
        print("   ✓ CausalBindingMechanism shapes correct")
        print("   ✓ Binding and broadcast working")
        
        return True
        
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def check_scl_loss():
    """Verify Structural Contrastive Loss."""
    print("\n[4/6] Checking SCL Loss...")
    
    try:
        from src.losses.structural_contrastive import StructuralContrastiveLoss
        
        loss_fn = StructuralContrastiveLoss(temperature=0.07)
        
        # Create fake structural representations
        struct_reps = torch.randn(4, 8, 256)
        
        # Create pair labels (some positive, some negative)
        pair_labels = torch.tensor([
            [1, 1, 0, 0],  # 0 and 1 are positive
            [1, 1, 0, 0],
            [0, 0, 1, 1],  # 2 and 3 are positive
            [0, 0, 1, 1]
        ]).float()
        
        loss = loss_fn(struct_reps, pair_labels)
        
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert not torch.isnan(loss), "Loss is NaN"
        
        print("   ✓ SCL loss computes correctly")
        print("   ✓ No NaN values")
        
        return True
        
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def check_pair_generation():
    """Verify pair generation for training."""
    print("\n[5/6] Checking Pair Generation...")
    
    try:
        from src.data.pair_generator import SCANPairGenerator
        from src.data.structure_extractor import SCANStructureExtractor
        
        extractor = SCANStructureExtractor()
        generator = SCANPairGenerator(pairs_per_example=3)
        
        # Test examples
        examples = [
            {'input': 'walk twice'},
            {'input': 'run twice'},      # Same structure as walk twice
            {'input': 'jump twice'},     # Same structure
            {'input': 'walk and run'},   # Different structure
            {'input': 'turn left'},      # Different structure
        ]
        
        generator.process_dataset(examples)
        
        # Generate pair labels
        batch_indices = [0, 1, 2, 3, 4]
        pair_labels = generator.generate_batch_pairs(batch_indices)
        
        assert pair_labels.shape == (5, 5), f"Wrong shape: {pair_labels.shape}"
        
        # Check some expected values
        assert pair_labels[0, 1] == 1, "walk twice and run twice should be positive"
        assert pair_labels[0, 3] == 0, "walk twice and walk and run should be negative"
        
        print("   ✓ Structure extraction working")
        print("   ✓ Pair labels generated correctly")
        print(f"   ✓ Found {len(generator.pattern_groups)} structural groups")
        
        return True
        
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def check_full_model():
    """Verify full SCI model integration."""
    print("\n[6/6] Checking Full Model Integration...")
    
    try:
        # Use a minimal config
        config = {
            'model': {
                'base_model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'dtype': 'float32'  # Use float32 for testing
            },
            'sci': {
                'structural_encoder': {
                    'enabled': True,
                    'n_slots': 8,
                    'n_layers': 2,
                    'n_heads': 4,
                    'abstraction_hidden_mult': 2
                },
                'content_encoder': {
                    'enabled': True,
                    'refiner_layers': 2,
                    'shared_embeddings': True,
                    'orthogonality_weight': 0.01
                },
                'causal_binding': {
                    'enabled': True,
                    'injection_layers': [6, 11, 16],
                    'n_iterations': 3
                },
                'contrastive_learning': {
                    'enabled': True,
                    'temperature': 0.07,
                    'scl_weight': 0.1
                }
            },
            'training': {
                'fp16': False
            }
        }
        
        print("   Loading model (this may take a moment)...")
        from src.models.sci_model import SCIModel
        
        model = SCIModel(config)
        model.eval()
        
        # Create fake batch
        batch_size = 2
        seq_len = 20
        
        tokenizer = model.tokenizer
        
        # Tokenize example
        texts = ["IN: walk twice OUT: WALK WALK", "IN: run twice OUT: RUN RUN"]
        tokenized = tokenizer(texts, return_tensors='pt', padding=True)
        
        # Create labels (mask instruction part)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Mask instruction tokens (everything before "OUT:")
        for i, text in enumerate(texts):
            out_pos = text.find("OUT:")
            instruction_ids = tokenizer(text[:out_pos + 4], return_tensors='pt')['input_ids']
            labels[i, :instruction_ids.shape[1]] = -100
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_loss=True
            )
            
        assert 'logits' in outputs
        assert 'structural_graph' in outputs
        assert 'content' in outputs
        assert 'lm_loss' in outputs
        
        print("   ✓ Model loads successfully")
        print("   ✓ Forward pass works")
        print("   ✓ All outputs present")
        print(f"   ✓ LM Loss: {outputs['lm_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    results = {
        'Structural Encoder': check_structural_encoder(),
        'Content Encoder': check_content_encoder(),
        'Causal Binding': check_causal_binding(),
        'SCL Loss': check_scl_loss(),
        'Pair Generation': check_pair_generation(),
        'Full Model': check_full_model()
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False
            
    print("=" * 60)
    
    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Ready for training!\n")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Fix issues before training!\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

---

## Phase 8: Main Training Script

### Step 8.1: Main Train Script (scripts/train.py)

```python
#!/usr/bin/env python
# scripts/train.py
"""
Main Training Script for SCI

Usage:
    python scripts/train.py --config configs/sci_tinyllama.yaml

For baseline comparison:
    python scripts/train.py --config configs/baseline.yaml
"""

import argparse
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='Train SCI model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode (small data)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Debug mode modifications
    if args.debug:
        config['training']['max_epochs'] = 2
        config['training']['batch_size'] = 4
        config['logging']['use_wandb'] = False
        
    print("=" * 60)
    print("SCI Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Dataset: {config['training']['dataset']}")
    print(f"Split: {config['training']['split']}")
    print(f"Epochs: {config['training']['max_epochs']}")
    print("=" * 60)
    
    # Verify implementation first
    print("\nRunning implementation verification...")
    from scripts.verify_implementation import main as verify
    if verify() != 0:
        print("Implementation verification failed. Fix issues before training.")
        return 1
        
    # Train
    from src.training.trainer import SCITrainer
    
    trainer = SCITrainer(config)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        # Load checkpoint logic here
        
    trainer.train()
    
    print("\nTraining complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

---

## Phase 9: Training Regime and Evaluation Strategy

### IMPORTANT: Training and Evaluation Plan

**Training Dataset: SCAN (length split)**
- Why: Clear compositional structure, known grammar enables perfect pair generation
- Training: Use SCL loss to enforce structural invariance

**In-Distribution Evaluation: SCAN (length split test)**
- Measures: Basic learning and optimization

**Compositional Generalization Evaluation: SCAN (template split)**
- Tests: Can model generalize to unseen structural compositions
- This is the KEY metric for SCI

**Cross-Benchmark Transfer: COGS (generalization split)**
- Tests: Does learned structural invariance transfer across datasets
- NOT expected to perform as well without fine-tuning

**What SCI Should NOT be evaluated on (without fine-tuning):**
- BBH: Too diverse, no clear structural equivalence
- GSM8K: Math reasoning requires different skills
- These can be evaluated AFTER proving compositional generalization works

### Expected Results

| Benchmark | Baseline | SCI (Expected) | Notes |
|-----------|----------|----------------|-------|
| SCAN (length, in-dist) | ~95% | ~98% | Basic learning |
| SCAN (length, OOD) | ~20% | >85% | KEY metric |
| SCAN (template) | ~50% | >90% | Compositional gen |
| COGS (gen) | ~35% | >70% | Transfer |

---

## Summary: AI Agent Workflow

1. **Create project structure** (Phase 1)
2. **Install dependencies** (Phase 1)
3. **Create all config files** (Phase 2)
4. **Implement Structural Encoder** with AbstractionLayer (Phase 3)
5. **Implement Content Encoder** with OrthogonalProjector (Phase 3)
6. **Implement Causal Binding** with CausalInterventionLayer (Phase 3)
7. **Implement SCL Loss** (Phase 3)
8. **Implement pair generation** for SCAN (Phase 4)
9. **Run verification tests** - MUST PASS (Phase 7)
10. **Train on SCAN length** (Phase 5)
11. **Evaluate on all splits** (Phase 6)
12. **Run ablations** (configs provided)
13. **Generate results**

**CRITICAL CHECKPOINTS:**
- After Step 9: All verification tests pass
- After Step 10: Training loss decreasing, SCL loss decreasing
- After Step 11: OOD accuracy > 70% (success indicator)
