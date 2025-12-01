# SCI Engineering Standards & Implementation Checklist

## Document Purpose
This document provides mandatory engineering standards, testing requirements, and implementation checklists for the AI agent implementing Structural Causal Invariance (SCI) in Python/PyTorch. Every section contains verification checkboxes that MUST be completed before the task is considered done.

**CRITICAL**: The AI agent must refer to this checklist at each implementation stage and explicitly confirm completion of each item.

---

# PART 1: FAIRNESS & EXPERIMENTAL INTEGRITY

## 1.1 Fair Comparison Protocol

### Training Budget Parity
```yaml
fairness_requirements:
  same_training_steps: true
  same_effective_batch_size: true
  same_optimizer_type: true  # Both use AdamW
  same_base_learning_rate: true  # Same base LR, SCI modules can have separate LR
  same_warmup_steps: true
  same_total_epochs: true
  same_gradient_accumulation: true
  same_mixed_precision: true  # Both FP16 or both FP32
  same_random_seeds: [42, 123, 456]  # Run both with same 3 seeds
```

### Checklist: Training Fairness
- [ ] Baseline and SCI use identical number of training steps
- [ ] Baseline and SCI use identical effective batch size
- [ ] Baseline and SCI use identical warmup schedule
- [ ] Baseline and SCI use identical weight decay (except SCI-specific modules)
- [ ] Baseline and SCI trained with same 3 random seeds for statistical significance
- [ ] Training time logged for both (SCI overhead should be <15%)
- [ ] Parameter count logged for both (SCI overhead should be <12%)

### Evaluation Parity
```yaml
evaluation_requirements:
  same_generation_config:
    max_new_tokens: 300  # Must accommodate longest SCAN output (288 tokens)
    temperature: 1.0  # Greedy decoding for reproducibility
    do_sample: false
    num_beams: 1  # No beam search for fair comparison
    repetition_penalty: 1.0  # Same for both OR same custom value
    length_penalty: 1.0  # Same for both
    eos_token_id: must_be_identical
    pad_token_id: must_be_identical
  same_tokenizer: true
  same_evaluation_script: true
  same_metrics: [exact_match, token_accuracy, sequence_accuracy]
```

### Checklist: Evaluation Fairness
- [ ] Generation config is IDENTICAL between baseline and SCI
- [ ] Same tokenizer used for both models
- [ ] Same evaluation script processes both outputs
- [ ] Same post-processing applied to both outputs
- [ ] Same metrics computed for both
- [ ] Evaluation runs on same hardware
- [ ] Results logged with timestamps and git commit hash

---

## 1.2 Anti-Cheating & Data Leakage Prevention

### Data Isolation Requirements
```python
# MANDATORY: Data split verification
class DataLeakageChecker:
    """Must be run before any training starts."""
    
    def verify_no_leakage(self, train_data, val_data, test_data):
        """
        Verifies:
        1. No overlap between train/val/test inputs
        2. No overlap between train/val/test outputs  
        3. For SCAN length split: train has sequences ≤22 actions, test has >22
        4. For template split: held-out primitives not in train
        """
        train_inputs = set(d['input'] for d in train_data)
        val_inputs = set(d['input'] for d in val_data)
        test_inputs = set(d['input'] for d in test_data)
        
        assert len(train_inputs & val_inputs) == 0, "Train/Val input overlap!"
        assert len(train_inputs & test_inputs) == 0, "Train/Test input overlap!"
        assert len(val_inputs & test_inputs) == 0, "Val/Test input overlap!"
        
        # For SCAN length split
        if self.split_type == 'length':
            train_max_actions = max(self._count_actions(d['output']) for d in train_data)
            test_min_actions = min(self._count_actions(d['output']) for d in test_data)
            assert train_max_actions <= 22, f"Train has sequences > 22 actions: {train_max_actions}"
            assert test_min_actions > 22, f"Test has sequences ≤ 22 actions: {test_min_actions}"
```

### Architecture-Level Leakage Prevention
```python
# MANDATORY: Attention mask verification
def verify_instruction_only_encoding(model, sample_batch):
    """
    Verifies that SE and CE encoders ONLY attend to instruction tokens,
    never to response tokens during encoding phase.
    """
    model.eval()
    with torch.no_grad():
        # Get attention weights from SE cross-attention
        _, se_attn_weights = model.structural_encoder(
            sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            instruction_mask=sample_batch['instruction_mask'],
            return_attention=True
        )
        
        # Verify zero attention on response tokens
        response_positions = ~sample_batch['instruction_mask']
        response_attention = se_attn_weights[:, :, :, response_positions]
        
        assert response_attention.abs().max() < 1e-9, \
            f"Data leakage detected! SE attending to response tokens: {response_attention.abs().max()}"
```

### Checklist: Anti-Cheating
- [ ] DataLeakageChecker runs before every training
- [ ] Train/val/test splits verified with zero overlap
- [ ] SCAN length split verified: train ≤22 actions, test >22 actions
- [ ] Instruction mask properly separates input from output
- [ ] SE attention weights verified to be 0 on response tokens
- [ ] CE attention weights verified to be 0 on response tokens
- [ ] No teacher forcing during evaluation
- [ ] No access to ground truth during inference
- [ ] Autoregressive generation used (no parallel decoding of answer)

---

# PART 2: MODULE TESTING REQUIREMENTS

## 2.1 Test Script Structure

Every SCI module MUST have a corresponding test file:

```
tests/
├── test_structural_encoder.py
├── test_content_encoder.py
├── test_abstraction_layer.py
├── test_causal_binding_mechanism.py
├── test_structural_gnn.py
├── test_scl_loss.py
├── test_orthogonality_loss.py
├── test_hook_registration.py
├── test_hook_activation.py
├── test_data_preparation.py
├── test_pair_generation.py
├── test_evaluation_metrics.py
├── test_generation_pipeline.py
├── test_checkpoint_resume.py
└── test_integration.py
```

## 2.2 Structural Encoder Tests

```python
# tests/test_structural_encoder.py
import pytest
import torch
from sci.modules import StructuralEncoder, AbstractionLayer

class TestAbstractionLayer:
    """Tests for AbstractionLayer module."""
    
    def test_output_shape(self):
        """Verify output shape matches input shape."""
        layer = AbstractionLayer(hidden_dim=768, num_features=8)
        x = torch.randn(4, 32, 768)  # [batch, seq, hidden]
        output, scores = layer(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        assert scores.shape == (4, 32), f"Scores shape wrong: {scores.shape}"
    
    def test_scores_range(self):
        """Verify structuralness scores are in [0, 1]."""
        layer = AbstractionLayer(hidden_dim=768, num_features=8)
        x = torch.randn(4, 32, 768)
        _, scores = layer(x)
        
        assert scores.min() >= 0.0, f"Scores below 0: {scores.min()}"
        assert scores.max() <= 1.0, f"Scores above 1: {scores.max()}"
    
    def test_gradient_flow(self):
        """Verify gradients flow through the layer."""
        layer = AbstractionLayer(hidden_dim=768, num_features=8)
        x = torch.randn(4, 32, 768, requires_grad=True)
        output, _ = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "No gradient on input"
        assert x.grad.abs().sum() > 0, "Zero gradients"
    
    def test_content_suppression_after_training(self):
        """After training, content words should have low scores."""
        # This test runs after a few training steps
        layer = AbstractionLayer(hidden_dim=768, num_features=8)
        # Load trained weights...
        # Verify content words (walk, jump, run) get scores < 0.3
        # Verify structure words (twice, and, left) get scores > 0.7
        pass


class TestStructuralEncoder:
    """Tests for full Structural Encoder."""
    
    def test_output_shapes(self):
        """Verify all output shapes are correct."""
        encoder = StructuralEncoder(
            hidden_dim=768,
            num_slots=8,
            num_gnn_layers=2
        )
        x = torch.randn(4, 32, 768)
        mask = torch.ones(4, 32).bool()
        
        slots, edge_weights, attn_weights = encoder(x, mask, return_all=True)
        
        assert slots.shape == (4, 8, 768), f"Slots shape: {slots.shape}"
        assert edge_weights.shape == (4, 8, 8), f"Edges shape: {edge_weights.shape}"
        assert attn_weights.shape == (4, 8, 32), f"Attn shape: {attn_weights.shape}"
    
    def test_slot_queries_learnable(self):
        """Verify slot queries are learnable parameters."""
        encoder = StructuralEncoder(hidden_dim=768, num_slots=8)
        
        assert hasattr(encoder, 'slot_queries'), "Missing slot_queries"
        assert encoder.slot_queries.requires_grad, "Slot queries not learnable"
        assert encoder.slot_queries.shape == (8, 768), "Wrong slot query shape"
    
    def test_instruction_only_attention(self):
        """Verify encoder only attends to instruction tokens."""
        encoder = StructuralEncoder(hidden_dim=768, num_slots=8)
        x = torch.randn(2, 20, 768)
        
        # First 10 tokens are instruction, last 10 are response
        instruction_mask = torch.zeros(2, 20).bool()
        instruction_mask[:, :10] = True
        
        _, _, attn_weights = encoder(x, instruction_mask, return_all=True)
        
        # Attention to response tokens (positions 10-19) should be 0
        response_attn = attn_weights[:, :, 10:]
        assert response_attn.abs().max() < 1e-9, \
            f"Attending to response tokens: {response_attn.abs().max()}"
```

## 2.3 Content Encoder Tests

```python
# tests/test_content_encoder.py
class TestContentEncoder:
    """Tests for Content Encoder module."""
    
    def test_output_shape(self):
        """Verify output shape."""
        encoder = ContentEncoder(hidden_dim=768)
        x = torch.randn(4, 32, 768)
        mask = torch.ones(4, 32).bool()
        
        output = encoder(x, mask)
        assert output.shape == (4, 32, 768), f"Shape: {output.shape}"
    
    def test_orthogonality_to_structure(self):
        """Verify content representations are orthogonal to structure."""
        se = StructuralEncoder(hidden_dim=768, num_slots=8)
        ce = ContentEncoder(hidden_dim=768)
        
        x = torch.randn(4, 32, 768)
        mask = torch.ones(4, 32).bool()
        
        struct_out, _, _ = se(x, mask, return_all=True)
        content_out = ce(x, mask)
        
        # Compute cosine similarity
        struct_flat = struct_out.mean(dim=1)  # [batch, hidden]
        content_flat = content_out.mean(dim=1)
        
        cosine_sim = F.cosine_similarity(struct_flat, content_flat, dim=-1)
        
        # After training, this should be close to 0
        # During init, just verify computation works
        assert cosine_sim.shape == (4,), f"Cosine shape: {cosine_sim.shape}"
    
    def test_shared_embeddings(self):
        """Verify CE uses shared embeddings from base model."""
        base_model = load_base_model()
        ce = ContentEncoder(hidden_dim=768, base_embeddings=base_model.embed_tokens)
        
        assert ce.embeddings is base_model.embed_tokens, "Embeddings not shared"
```

## 2.4 Causal Binding Mechanism Tests

```python
# tests/test_causal_binding_mechanism.py
class TestCausalBindingMechanism:
    """Tests for CBM module."""
    
    def test_binding_attention_shape(self):
        """Verify binding attention produces correct shapes."""
        cbm = CausalBindingMechanism(hidden_dim=768, num_slots=8)
        
        slots = torch.randn(4, 8, 768)  # Structure
        content = torch.randn(4, 32, 768)  # Content
        edge_weights = torch.randn(4, 8, 8)
        
        bound, attn = cbm(slots, content, edge_weights, return_attention=True)
        
        assert bound.shape == (4, 8, 768), f"Bound shape: {bound.shape}"
        assert attn.shape == (4, 8, 32), f"Attn shape: {attn.shape}"
    
    def test_causal_intervention(self):
        """Verify causal intervention uses edge weights correctly."""
        cbm = CausalBindingMechanism(hidden_dim=768, num_slots=8)
        
        slots = torch.randn(4, 8, 768)
        content = torch.randn(4, 32, 768)
        
        # Zero edge weights = no message passing
        zero_edges = torch.zeros(4, 8, 8)
        bound_no_edges = cbm(slots, content, zero_edges)
        
        # Full edge weights
        full_edges = torch.ones(4, 8, 8)
        bound_full_edges = cbm(slots, content, full_edges)
        
        # Results should differ
        assert not torch.allclose(bound_no_edges, bound_full_edges), \
            "Edge weights have no effect on binding"
    
    def test_broadcast_to_sequence(self):
        """Verify broadcast operation maps slots back to sequence."""
        cbm = CausalBindingMechanism(hidden_dim=768, num_slots=8)
        
        bound_slots = torch.randn(4, 8, 768)
        seq_len = 32
        
        broadcast = cbm.broadcast(bound_slots, seq_len)
        
        assert broadcast.shape == (4, 32, 768), f"Broadcast shape: {broadcast.shape}"
```

## 2.5 Hook Registration & Activation Tests

```python
# tests/test_hook_registration.py
class TestHookRegistration:
    """Tests for hook registration and activation."""
    
    def test_hooks_registered_at_correct_layers(self):
        """Verify hooks are registered at specified layers."""
        config = SCIConfig(injection_layers=[3, 6, 9, 12])
        model = SCIModel(base_model, config)
        
        registered_layers = [h.layer_idx for h in model._sci_hooks]
        assert registered_layers == [3, 6, 9, 12], \
            f"Wrong layers: {registered_layers}"
    
    def test_hooks_activated_during_forward(self):
        """Verify hooks are called during forward pass."""
        config = SCIConfig(injection_layers=[3, 6, 9])
        model = SCIModel(base_model, config)
        
        # Add activation counter
        activation_counts = {3: 0, 6: 0, 9: 0}
        
        def count_hook(layer_idx):
            def hook(module, input, output):
                activation_counts[layer_idx] += 1
                return output
            return hook
        
        for layer_idx in [3, 6, 9]:
            model.base_model.layers[layer_idx].register_forward_hook(
                count_hook(layer_idx)
            )
        
        # Forward pass
        x = torch.randint(0, 1000, (2, 32))
        model(x)
        
        for layer_idx, count in activation_counts.items():
            assert count == 1, f"Layer {layer_idx} activated {count} times"
    
    def test_hooks_modify_hidden_states(self):
        """Verify hooks actually modify hidden states."""
        config = SCIConfig(injection_layers=[6])
        model = SCIModel(base_model, config)
        
        x = torch.randint(0, 1000, (2, 32))
        
        # Get hidden states without SCI
        model.disable_sci()
        _, hidden_no_sci = model(x, output_hidden_states=True)
        
        # Get hidden states with SCI
        model.enable_sci()
        _, hidden_with_sci = model(x, output_hidden_states=True)
        
        # Layer 6 hidden states should differ
        layer_6_diff = (hidden_with_sci[6] - hidden_no_sci[6]).abs().mean()
        assert layer_6_diff > 0.01, f"SCI not modifying layer 6: diff={layer_6_diff}"
    
    def test_hooks_work_during_generation(self):
        """Verify hooks are active during autoregressive generation."""
        config = SCIConfig(injection_layers=[3, 6, 9])
        model = SCIModel(base_model, config)
        
        hook_activations = []
        
        def tracking_hook(module, input, output):
            hook_activations.append(len(hook_activations))
            return output
        
        model.base_model.layers[6].register_forward_hook(tracking_hook)
        
        # Generate 10 tokens
        x = torch.randint(0, 1000, (1, 10))
        model.generate(x, max_new_tokens=10)
        
        # Hook should be called multiple times (once per generated token)
        assert len(hook_activations) >= 10, \
            f"Hook called only {len(hook_activations)} times during generation"


# tests/test_hook_activation.py
class TestHookActivationModes:
    """Tests for hook behavior in different modes."""
    
    def test_training_mode_hooks(self):
        """Verify hooks work correctly in training mode."""
        model = SCIModel(base_model, config)
        model.train()
        
        x = torch.randint(0, 1000, (4, 32))
        labels = torch.randint(0, 1000, (4, 32))
        
        loss = model(x, labels=labels).loss
        loss.backward()
        
        # Verify SCI parameters have gradients
        for name, param in model.named_parameters():
            if 'sci' in name.lower():
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_eval_mode_hooks(self):
        """Verify hooks work correctly in eval mode."""
        model = SCIModel(base_model, config)
        model.eval()
        
        with torch.no_grad():
            x = torch.randint(0, 1000, (4, 32))
            output = model(x)
        
        assert output.logits.shape == (4, 32, vocab_size)
    
    def test_inference_mode_generation(self):
        """Verify hooks work during inference generation."""
        model = SCIModel(base_model, config)
        model.eval()
        
        x = torch.randint(0, 1000, (1, 10))
        
        with torch.inference_mode():
            output = model.generate(
                x,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        
        assert output.shape[1] > 10, "No tokens generated"
```

## 2.6 SCL Loss & Pair Generation Tests

```python
# tests/test_scl_loss.py
class TestSCLLoss:
    """Tests for Structural Contrastive Learning loss."""
    
    def test_loss_shape(self):
        """Verify loss is scalar."""
        scl = SCLLoss(temperature=0.1)
        
        z_anchor = torch.randn(32, 768)
        z_positive = torch.randn(32, 768)
        z_negatives = torch.randn(32, 5, 768)  # 5 negatives per anchor
        
        loss = scl(z_anchor, z_positive, z_negatives)
        
        assert loss.dim() == 0, f"Loss not scalar: {loss.shape}"
    
    def test_loss_decreases_for_similar_pairs(self):
        """Verify loss is lower when positive pairs are similar."""
        scl = SCLLoss(temperature=0.1)
        
        z_anchor = torch.randn(32, 768)
        z_anchor_normalized = F.normalize(z_anchor, dim=-1)
        
        # Very similar positive
        z_pos_similar = z_anchor_normalized + 0.01 * torch.randn_like(z_anchor)
        z_pos_similar = F.normalize(z_pos_similar, dim=-1)
        
        # Dissimilar positive
        z_pos_dissimilar = torch.randn(32, 768)
        z_pos_dissimilar = F.normalize(z_pos_dissimilar, dim=-1)
        
        z_neg = torch.randn(32, 5, 768)
        
        loss_similar = scl(z_anchor_normalized, z_pos_similar, z_neg)
        loss_dissimilar = scl(z_anchor_normalized, z_pos_dissimilar, z_neg)
        
        assert loss_similar < loss_dissimilar, \
            f"Loss not lower for similar pairs: {loss_similar} vs {loss_dissimilar}"
    
    def test_temperature_effect(self):
        """Verify temperature affects loss magnitude."""
        z_anchor = torch.randn(32, 768)
        z_positive = torch.randn(32, 768)
        z_negatives = torch.randn(32, 5, 768)
        
        scl_low_temp = SCLLoss(temperature=0.05)
        scl_high_temp = SCLLoss(temperature=0.5)
        
        loss_low = scl_low_temp(z_anchor, z_positive, z_negatives)
        loss_high = scl_high_temp(z_anchor, z_positive, z_negatives)
        
        # Lower temperature should give sharper distribution
        # This is a sanity check that temperature is being used
        assert loss_low != loss_high, "Temperature has no effect"


# tests/test_pair_generation.py
class TestPairGeneration:
    """Tests for automated pair generation."""
    
    def test_positive_pairs_have_same_structure(self):
        """Verify positive pairs share structural template."""
        generator = SCLPairGenerator(grammar='scan')
        
        pairs = generator.generate_positive_pairs(n=100)
        
        for anchor, positive in pairs:
            anchor_template = generator.extract_template(anchor)
            positive_template = generator.extract_template(positive)
            
            assert anchor_template == positive_template, \
                f"Template mismatch: {anchor} ({anchor_template}) vs {positive} ({positive_template})"
    
    def test_negative_pairs_have_different_structure(self):
        """Verify negative pairs have different templates."""
        generator = SCLPairGenerator(grammar='scan')
        
        negatives = generator.generate_negative_pairs(n=100)
        
        for anchor, negative in negatives:
            anchor_template = generator.extract_template(anchor)
            negative_template = generator.extract_template(negative)
            
            assert anchor_template != negative_template, \
                f"Same template for negative: {anchor} ({anchor_template}) vs {negative}"
    
    def test_template_extraction_accuracy(self):
        """Verify template extraction is correct."""
        generator = SCLPairGenerator(grammar='scan')
        
        test_cases = [
            ("walk twice", "[ACTION] twice"),
            ("run and jump", "[ACTION] and [ACTION]"),
            ("jump left twice", "[ACTION] [DIRECTION] twice"),
            ("walk opposite left", "[ACTION] opposite [DIRECTION]"),
            ("run around right thrice", "[ACTION] around [DIRECTION] thrice"),
        ]
        
        for input_text, expected_template in test_cases:
            actual_template = generator.extract_template(input_text)
            assert actual_template == expected_template, \
                f"Template extraction failed: {input_text} -> {actual_template} (expected {expected_template})"
    
    def test_pair_batch_diversity(self):
        """Verify pair batches have diverse structures."""
        generator = SCLPairGenerator(grammar='scan')
        
        batch = generator.generate_batch(batch_size=64, num_negatives=5)
        
        anchor_templates = [generator.extract_template(a) for a in batch['anchors']]
        unique_templates = set(anchor_templates)
        
        # Should have at least 5 different templates in a batch of 64
        assert len(unique_templates) >= 5, \
            f"Low diversity: only {len(unique_templates)} unique templates"
```

---

# PART 3: TRAINING CONFIGURATION

## 3.1 Optimal Hyperparameter Configuration

```yaml
# configs/sci_optimal.yaml
# CRITICAL: These hyperparameters have been tuned for stable training
# and optimal compositional generalization on SCAN length split.

model:
  base_model: "google/t5-small"  # Or your chosen base
  hidden_dim: 768
  
sci_modules:
  # Structural Encoder
  structural_encoder:
    num_slots: 8  # Number of structural slot queries
    num_gnn_layers: 2  # GNN depth for causal reasoning
    gnn_hidden_dim: 768
    dropout: 0.1
    
  # Abstraction Layer  
  abstraction_layer:
    num_features: 8  # Feature dimensions for structuralness scoring
    temperature: 1.0  # Softmax temperature (higher = softer masking)
    learned_temperature: true  # Allow temperature to be learned
    
  # Content Encoder
  content_encoder:
    num_layers: 2  # Lightweight transformer layers
    use_shared_embeddings: true
    orthogonality_weight: 0.1  # Loss weight for orthogonality constraint
    
  # Causal Binding Mechanism
  cbm:
    num_attention_heads: 8
    binding_dropout: 0.1
    use_causal_intervention: true
    intervention_strength: 1.0
    
  # Injection Configuration
  injection:
    layers: [3, 6, 9, 12]  # Layers to inject CBM output
    # REASONING: Early layers (3) capture syntax, middle (6,9) capture semantics,
    # late (12) captures task-specific patterns. Multi-layer injection ensures
    # structural information propagates throughout the network.
    injection_method: "residual_add"  # Options: residual_add, gated_add, replace
    injection_scale: 0.5  # Scale factor for residual addition

training:
  # Optimizer
  optimizer: "adamw"
  base_lr: 2e-5  # For base model parameters
  sci_lr: 5e-5  # Higher LR for SCI modules (they need to learn faster)
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
  
  # Learning Rate Schedule
  scheduler: "cosine_with_warmup"
  warmup_ratio: 0.1  # 10% of total steps for warmup
  min_lr_ratio: 0.01  # Minimum LR = 1% of peak LR
  
  # Batch Configuration
  batch_size: 8
  gradient_accumulation_steps: 4
  effective_batch_size: 32  # batch_size * gradient_accumulation
  
  # Training Duration
  num_epochs: 20
  max_steps: null  # If set, overrides num_epochs
  
  # Regularization
  max_grad_norm: 1.0
  dropout: 0.1
  label_smoothing: 0.0
  
  # Mixed Precision
  fp16: true
  bf16: false  # Use if available (A100/H100)
  
  # SCL Training
  scl:
    enabled: true
    weight: 0.3  # Weight of SCL loss in total loss
    temperature: 0.1  # Contrastive loss temperature
    num_negatives: 5  # Negatives per anchor
    hard_negative_mining: true  # Mine hard negatives from batch
    warmup_epochs: 2  # Don't apply SCL for first 2 epochs
    
  # Orthogonality Loss
  orthogonality:
    enabled: true
    weight: 0.1
    
  # Early Stopping
  early_stopping:
    enabled: true
    patience: 5  # Stop after 5 epochs without improvement
    min_delta: 0.001  # Minimum improvement to count
    metric: "val_exact_match"  # Metric to monitor
    mode: "max"  # Higher is better

evaluation:
  # Run evaluation every N steps
  eval_steps: 500
  
  # Generation Config for Evaluation
  generation:
    max_new_tokens: 300  # Must be >= max SCAN output length (288)
    min_new_tokens: 1
    do_sample: false  # Greedy decoding
    temperature: 1.0
    num_beams: 1
    repetition_penalty: 1.0  # No penalty (fair comparison)
    length_penalty: 1.0  # No penalty
    early_stopping: true  # Stop at EOS
    
  # Metrics
  metrics:
    - exact_match
    - token_accuracy
    - sequence_accuracy
    - length_accuracy  # Did model predict correct length?

checkpointing:
  save_strategy: "steps"
  save_steps: 1000
  save_total_limit: 3  # Keep only last 3 checkpoints
  save_on_each_node: false
  resume_from_checkpoint: null  # Set to path to resume

logging:
  log_level: "info"
  log_steps: 10
  log_to_file: true
  log_dir: "logs/"
  tensorboard: true
  wandb:
    enabled: true
    project: "sci-compositional"
    entity: null

seeds:
  training_seed: 42
  data_seed: 42
  model_seed: 42
```

## 3.2 Hyperparameter Reasoning for Stability

### Why These Learning Rates?
```python
"""
CRITICAL: Learning rate configuration for stable SCI training.

BASE MODEL LR = 2e-5:
- Standard for fine-tuning pretrained transformers
- Lower to avoid catastrophic forgetting of base knowledge
- Allows gradual adaptation to SCAN task

SCI MODULE LR = 5e-5 (2.5x higher):
- SCI modules are randomly initialized (no pretrained weights)
- Need faster learning to catch up with pretrained base
- Higher LR helps AbstractionLayer learn structuralness quickly
- Prevents SCI from being a bottleneck

WARMUP (10% of steps):
- Prevents early instability from large gradients
- Critical for SCL loss which can be noisy initially
- Allows model to find stable training trajectory

COSINE SCHEDULE:
- Smooth decay prevents training instability
- Reaches near-zero LR at end for fine-grained convergence
- Better than step decay for compositional generalization
"""

def get_optimizer_groups(model, config):
    """Create parameter groups with different learning rates."""
    
    # Base model parameters
    base_params = []
    # SCI module parameters  
    sci_params = []
    # No decay parameters (biases, layer norms)
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(nd in name for nd in ['bias', 'LayerNorm', 'layer_norm']):
            no_decay_params.append(param)
        elif 'sci' in name.lower() or any(m in name for m in 
            ['structural_encoder', 'content_encoder', 'cbm', 'abstraction']):
            sci_params.append(param)
        else:
            base_params.append(param)
    
    return [
        {'params': base_params, 'lr': config.base_lr, 'weight_decay': config.weight_decay},
        {'params': sci_params, 'lr': config.sci_lr, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'lr': config.base_lr, 'weight_decay': 0.0},
    ]
```

### Handling Long Sequences (SCAN Length Challenge)
```yaml
# CRITICAL: Configuration for SCAN length split (up to 288 output tokens)

long_sequence_handling:
  # Positional Encoding
  max_position_embeddings: 512  # Must exceed max seq length
  position_encoding: "rotary"  # RoPE for better length generalization
  
  # Attention Configuration
  attention_type: "flash_attention"  # Memory efficient for long sequences
  sliding_window: null  # Full attention needed for structural patterns
  
  # Structural Slot Configuration
  num_slots: 8  # 8 slots can represent most SCAN structures
  # REASONING: SCAN has limited structural complexity
  # "X around Y twice and Z opposite W" needs ~6-7 slots max
  
  # Length Prediction
  length_prediction:
    enabled: true
    auxiliary_head: true  # Predict output length from structure
    loss_weight: 0.05
    
  # EOS Token Handling (CRITICAL for exact match)
  eos_handling:
    add_eos_to_targets: true  # Ensure EOS in training targets
    eos_loss_weight: 2.0  # Higher weight on EOS prediction
    # REASONING: Model must learn when to stop generating
    # Higher EOS loss weight encourages correct length prediction
```

### Preventing Local Minima
```python
"""
STRATEGIES TO AVOID LOCAL MINIMA IN SCI TRAINING:

1. WARMUP PERIOD:
   - First 10% of training uses warmup
   - SCL loss not applied in first 2 epochs
   - Allows base model to stabilize before SCI constraints

2. LEARNING RATE CONFIGURATION:
   - Cosine schedule prevents premature convergence
   - Higher SCI LR ensures modules learn quickly
   - Separate LR for different module types

3. GRADIENT CLIPPING:
   - max_grad_norm = 1.0 prevents explosion
   - Especially important for SCL loss gradients

4. LOSS WEIGHTING SCHEDULE:
   - SCL weight starts at 0, increases to 0.3 over 2 epochs
   - Orthogonality weight constant at 0.1
   - Task loss always has weight 1.0

5. BATCH DIVERSITY:
   - Ensure each batch has diverse structural templates
   - Hard negative mining after warmup
   - Mix of simple and complex structures

6. REGULARIZATION:
   - Dropout 0.1 on all SCI modules
   - Weight decay 0.01 on all parameters
   - Label smoothing 0.0 (exact match needed)
"""

class SCITrainer:
    def compute_loss_weights(self, epoch, step):
        """Dynamic loss weighting to avoid local minima."""
        
        # Task loss always weight 1.0
        task_weight = 1.0
        
        # SCL weight warmup
        if epoch < self.config.scl_warmup_epochs:
            scl_weight = 0.0
        else:
            # Linear increase from 0 to target over 2 epochs
            warmup_progress = (epoch - self.config.scl_warmup_epochs) / 2.0
            scl_weight = min(self.config.scl_weight, 
                           self.config.scl_weight * warmup_progress)
        
        # Orthogonality weight constant
        ortho_weight = self.config.orthogonality_weight
        
        return {
            'task': task_weight,
            'scl': scl_weight,
            'orthogonality': ortho_weight
        }
```

---

# PART 4: EVALUATION REGIME

## 4.1 Evaluation Order (MANDATORY)

```python
"""
MANDATORY EVALUATION ORDER:
1. In-distribution SCAN (sanity check)
2. SCAN Length Split (primary OOD benchmark)
3. SCAN Template Split (secondary OOD benchmark)
4. Zero-shot COGS (cross-benchmark generalization)

DO NOT skip steps or change order.
"""

EVALUATION_ORDER = [
    {
        "name": "scan_in_distribution",
        "dataset": "scan",
        "split": "simple",  # Or "train_val_split"
        "purpose": "Sanity check - model should achieve >95%",
        "threshold": 0.95,
        "required": True
    },
    {
        "name": "scan_length_ood", 
        "dataset": "scan",
        "split": "length",
        "purpose": "Primary benchmark - compositional length generalization",
        "threshold": 0.80,  # Target for publication
        "required": True
    },
    {
        "name": "scan_template_ood",
        "dataset": "scan",
        "split": "template",  # "addprim_jump" or "addprim_turn_left"
        "purpose": "Template generalization - held-out primitives",
        "threshold": 0.85,
        "required": True
    },
    {
        "name": "cogs_zero_shot",
        "dataset": "cogs",
        "split": "gen",  # Generalization split
        "purpose": "Cross-benchmark transfer - semantic compositionality",
        "threshold": 0.60,
        "required": False  # Nice to have
    }
]
```

## 4.2 Exact Match Requirements

```python
# CRITICAL: SCAN requires EXACT MATCH for success

class SCANEvaluator:
    """Evaluator for SCAN benchmark with exact match scoring."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def evaluate(self, model, test_dataloader):
        model.eval()
        
        results = {
            'exact_match': 0,
            'token_accuracy': 0,
            'length_correct': 0,
            'total': 0,
            'errors': []
        }
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                # Generate predictions
                outputs = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=300,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode predictions and targets
                predictions = self.tokenizer.batch_decode(
                    outputs[:, batch['input_ids'].shape[1]:],  # Remove input
                    skip_special_tokens=True
                )
                targets = self.tokenizer.batch_decode(
                    batch['labels'],
                    skip_special_tokens=True
                )
                
                for pred, target, input_ids in zip(predictions, targets, batch['input_ids']):
                    results['total'] += 1
                    
                    # Normalize whitespace
                    pred_normalized = ' '.join(pred.strip().split())
                    target_normalized = ' '.join(target.strip().split())
                    
                    # EXACT MATCH (most important)
                    if pred_normalized == target_normalized:
                        results['exact_match'] += 1
                    else:
                        # Log error for analysis
                        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                        results['errors'].append({
                            'input': input_text,
                            'prediction': pred_normalized,
                            'target': target_normalized
                        })
                    
                    # Token accuracy
                    pred_tokens = pred_normalized.split()
                    target_tokens = target_normalized.split()
                    
                    if len(pred_tokens) == len(target_tokens):
                        results['length_correct'] += 1
                        correct_tokens = sum(p == t for p, t in zip(pred_tokens, target_tokens))
                        results['token_accuracy'] += correct_tokens / len(target_tokens)
                    else:
                        # Length mismatch - partial credit for matching prefix
                        min_len = min(len(pred_tokens), len(target_tokens))
                        if min_len > 0:
                            correct_tokens = sum(p == t for p, t in 
                                               zip(pred_tokens[:min_len], target_tokens[:min_len]))
                            results['token_accuracy'] += correct_tokens / len(target_tokens)
        
        # Compute final metrics
        n = results['total']
        return {
            'exact_match': results['exact_match'] / n,
            'token_accuracy': results['token_accuracy'] / n,
            'length_accuracy': results['length_correct'] / n,
            'total_samples': n,
            'num_errors': len(results['errors']),
            'sample_errors': results['errors'][:10]  # First 10 errors for analysis
        }
```

## 4.3 EOS Token Handling (CRITICAL)

```python
"""
CRITICAL: Proper EOS handling is essential for exact match on SCAN length split.

The model MUST learn to generate EOS at the correct position.
Long sequences (up to 288 tokens) are particularly challenging.
"""

class SCANDataCollator:
    """Data collator with proper EOS handling."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure EOS token is set
        assert tokenizer.eos_token_id is not None, "EOS token not set!"
        
    def __call__(self, features):
        # Tokenize inputs
        inputs = [f['input'] for f in features]
        targets = [f['target'] for f in features]
        
        # Encode inputs
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode targets WITH EOS token
        target_encodings = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Ensure EOS is at the end of each target
        labels = target_encodings['input_ids'].clone()
        for i, target in enumerate(targets):
            # Find the last non-padding position
            non_pad_mask = labels[i] != self.tokenizer.pad_token_id
            if non_pad_mask.any():
                last_token_idx = non_pad_mask.nonzero()[-1].item()
                # Ensure EOS is there (add if not)
                if labels[i, last_token_idx] != self.tokenizer.eos_token_id:
                    if last_token_idx + 1 < self.max_length:
                        labels[i, last_token_idx + 1] = self.tokenizer.eos_token_id
        
        # Mask padding in labels with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
            'instruction_mask': self._create_instruction_mask(input_encodings, target_encodings)
        }
    
    def _create_instruction_mask(self, inputs, targets):
        """Create mask that is 1 for instruction tokens, 0 for response tokens."""
        # Combined sequence: [input tokens] [target tokens]
        batch_size = inputs['input_ids'].shape[0]
        input_len = inputs['attention_mask'].sum(dim=1)
        
        # Full sequence length
        total_len = inputs['input_ids'].shape[1] + targets['input_ids'].shape[1]
        
        mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
        for i in range(batch_size):
            mask[i, :input_len[i]] = True
            
        return mask
```

## 4.4 Repetition & Length Penalty Configuration

```yaml
# Evaluation generation config - MUST BE IDENTICAL for baseline and SCI

evaluation_generation:
  # IMPORTANT: Same settings for both models for fair comparison
  repetition_penalty: 1.0  # No penalty (let model learn naturally)
  length_penalty: 1.0  # No penalty
  
  # Alternative: If using penalties, must be same for both
  # repetition_penalty: 1.2
  # length_penalty: 0.8
  
  # WHY NO PENALTIES:
  # - SCAN has legitimate repetition ("WALK WALK WALK WALK")
  # - Repetition penalty would hurt legitimate sequences
  # - Length penalty would bias towards shorter outputs
  # - SCI should learn correct patterns without heuristics
  
  # If model generates repetitive garbage, that's a training problem
  # not something to fix with generation penalties
```

---

# PART 5: TRAINING LOGGING & REPRODUCIBILITY

## 5.1 Training Log Format

```python
# MANDATORY: Training log format for reproducibility

TRAINING_LOG_TEMPLATE = """
================================================================================
                         SCI TRAINING LOG
================================================================================

Experiment: {experiment_name}
Description: {description}
Start Time: {start_time}
Git Commit: {git_commit}
Random Seeds: train={train_seed}, data={data_seed}, model={model_seed}

================================================================================
                         CONFIGURATION
================================================================================

--- Model Parameters ---
Base model: {base_model_name}
Base model params: {base_params:,}
SCI added params: {sci_params:,}
Total trainable: {total_params:,}
Parameter overhead: {overhead:.2f}%

SCI Modules:
  Structural Encoder: {se_enabled} (slots={num_slots}, gnn_layers={gnn_layers})
  Content Encoder: {ce_enabled} (layers={ce_layers})
  Abstraction Layer: {al_enabled} (features={al_features})
  CBM: {cbm_enabled} (heads={cbm_heads})
  Injection layers: {injection_layers}

--- Training Config ---
Epochs: {num_epochs}
Batch size: {batch_size}
Gradient accumulation: {grad_accum}
Effective batch size: {effective_batch}
Base LR: {base_lr}
SCI LR: {sci_lr}
LR scheduler: {scheduler}
Warmup ratio: {warmup_ratio}
Weight decay: {weight_decay}
Max grad norm: {max_grad_norm}
Mixed precision: {mixed_precision}

--- Loss Weights ---
Task loss: 1.0
SCL loss: {scl_weight} (warmup epochs: {scl_warmup})
Orthogonality loss: {ortho_weight}
EOS loss weight: {eos_weight}

--- Dataset ---
Train samples: {train_samples:,}
Val samples: {val_samples:,}
Test samples: {test_samples:,}
Max input length: {max_input_len}
Max output length: {max_output_len}

================================================================================
                         TRAINING PROGRESS
================================================================================

  Epoch |   Step |     Loss |  Task L |   SCL L | Ortho L |    LR |   Time | GPU Mem
---------------------------------------------------------------------------------------
"""

EPOCH_SUMMARY_TEMPLATE = """
--- Epoch {epoch} Summary ---
Train Loss: {train_loss:.6f}
  - Task: {task_loss:.6f}
  - SCL: {scl_loss:.6f}
  - Orthogonality: {ortho_loss:.6f}
Val Loss: {val_loss:.6f}
Val Exact Match: {val_exact_match:.4f}
Val Token Accuracy: {val_token_acc:.4f}
Val Length Accuracy: {val_length_acc:.4f}
Learning Rate: {lr:.2e}
Epoch Time: {epoch_time:.1f}s
Best Val EM: {best_val_em:.4f} (Epoch {best_epoch})
Early Stopping Counter: {es_counter}/{patience}

AbstractionLayer Stats:
  - Content word avg score: {content_avg_score:.3f}
  - Structure word avg score: {struct_avg_score:.3f}
  - Separation ratio: {separation_ratio:.2f}

Structural Invariance:
  - Same-structure cosine sim: {same_struct_sim:.3f}
  - Diff-structure cosine sim: {diff_struct_sim:.3f}
  - Invariance ratio: {invariance_ratio:.2f}

"""

TRAINING_COMPLETE_TEMPLATE = """
================================================================================
                         TRAINING COMPLETE
================================================================================

End Time: {end_time}
Total Duration: {total_duration}
Final Epoch: {final_epoch}
Termination Reason: {termination_reason}

--- Best Model ---
Epoch: {best_epoch}
Val Exact Match: {best_val_em:.4f}
Val Token Accuracy: {best_val_token_acc:.4f}
Checkpoint: {best_checkpoint_path}

--- Final Test Results ---
Test Exact Match: {test_em:.4f}
Test Token Accuracy: {test_token_acc:.4f}
Test Length Accuracy: {test_length_acc:.4f}

--- Hardware ---
GPU: {gpu_name}
Peak GPU Memory: {peak_gpu_mem:.2f}GB
Training throughput: {throughput:.1f} samples/sec

================================================================================
"""
```

## 5.2 Checkpoint & Resume System

```python
# checkpointing/checkpoint_manager.py

import os
import json
import torch
from datetime import datetime

class CheckpointManager:
    """Manages checkpoints for training resumption."""
    
    def __init__(self, checkpoint_dir, save_total_limit=3):
        self.checkpoint_dir = checkpoint_dir
        self.save_total_limit = save_total_limit
        os.makedirs(checkpoint_dir, exist_ok=True)
        
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
            'best_val_metric': training_state['best_val_metric'],
            'best_epoch': training_state['best_epoch'],
            'early_stopping_counter': training_state['early_stopping_counter'],
            
            # Metrics history
            'metrics_history': metrics,
            
            # Configuration
            'config': config,
            
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest pointer
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")
        with open(latest_path, 'w') as f:
            json.dump({
                'checkpoint_name': checkpoint_name,
                'epoch': epoch,
                'step': step,
                'timestamp': checkpoint['timestamp']
            }, f, indent=2)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint for resumption."""
        
        if checkpoint_path is None:
            # Load latest
            latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")
            if not os.path.exists(latest_path):
                return None
            with open(latest_path, 'r') as f:
                latest_info = json.load(f)
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_info['checkpoint_name'])
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only save_total_limit."""
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("checkpoint_") and f.endswith(".pt"):
                path = os.path.join(self.checkpoint_dir, f)
                mtime = os.path.getmtime(path)
                checkpoints.append((mtime, path))
        
        # Sort by modification time (newest first)
        checkpoints.sort(reverse=True)
        
        # Remove old checkpoints
        for _, path in checkpoints[self.save_total_limit:]:
            os.remove(path)
            print(f"Removed old checkpoint: {path}")


class TrainingResumer:
    """Handles training resumption from checkpoint."""
    
    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager
    
    def resume_training(self, model, optimizer, scheduler, 
                       checkpoint_path=None, device='cuda'):
        """Resume training from checkpoint."""
        
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        if checkpoint is None:
            print("No checkpoint found, starting fresh training")
            return {
                'start_epoch': 0,
                'global_step': 0,
                'best_val_metric': 0.0,
                'best_epoch': 0,
                'early_stopping_counter': 0,
                'metrics_history': []
            }
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        # Load scheduler state
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'start_epoch': checkpoint['epoch'] + 1,  # Start from next epoch
            'global_step': checkpoint['global_step'],
            'best_val_metric': checkpoint['best_val_metric'],
            'best_epoch': checkpoint['best_epoch'],
            'early_stopping_counter': checkpoint['early_stopping_counter'],
            'metrics_history': checkpoint['metrics_history']
        }
```

## 5.3 Early Stopping & Overfitting Detection

```python
# training/early_stopping.py

class EarlyStopping:
    """Early stopping with patience and overfitting detection."""
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """Check if should stop training."""
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class OverfittingDetector:
    """Detects overfitting by comparing train/val losses."""
    
    def __init__(self, 
                 threshold_ratio=1.5,  # Val loss > 1.5x train loss
                 window_size=3,  # Average over 3 epochs
                 min_epochs=5):  # Don't detect before epoch 5
        self.threshold_ratio = threshold_ratio
        self.window_size = window_size
        self.min_epochs = min_epochs
        self.train_losses = []
        self.val_losses = []
        
    def update(self, train_loss, val_loss, epoch):
        """Update with new losses and check for overfitting."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if epoch < self.min_epochs:
            return False, None
        
        if len(self.train_losses) < self.window_size:
            return False, None
        
        # Compute windowed averages
        recent_train = sum(self.train_losses[-self.window_size:]) / self.window_size
        recent_val = sum(self.val_losses[-self.window_size:]) / self.window_size
        
        # Check ratio
        ratio = recent_val / (recent_train + 1e-8)
        
        is_overfitting = ratio > self.threshold_ratio
        
        return is_overfitting, ratio
```

---

# PART 6: LAYER SELECTION GUIDANCE

## 6.1 Optimal Injection Layer Analysis

```python
"""
LAYER SELECTION REASONING FOR CBM INJECTION:

For a 12-layer transformer (e.g., T5-small):

LAYER 3 (Early):
- Captures: Basic syntax, token-level patterns
- Good for: Learning basic structural markers (twice, and, left)
- Risk: Too much injection can disrupt basic language modeling

LAYER 6 (Middle):
- Captures: Semantic composition, phrase-level meaning
- Good for: Binding content to structural slots
- Optimal layer for CBM injection

LAYER 9 (Late-middle):
- Captures: Task-specific patterns, output formatting
- Good for: Reinforcing structural constraints before generation
- Important for maintaining generation coherence

LAYER 12 (Final):
- Captures: Output logits preparation
- Good for: Final structural guidance
- Risk: May be too late for meaningful structural influence

RECOMMENDED CONFIGURATION:
- For 12-layer model: [3, 6, 9]
- For 16-layer model: [4, 8, 12, 15]
- For 24-layer model: [4, 8, 12, 16, 20]

CRITICAL: Start with middle layers (6, 9) and add early/late based on results.
"""

def get_optimal_injection_layers(num_layers):
    """Get optimal injection layers based on model depth."""
    
    if num_layers <= 6:
        # Small model: inject at 1/3 and 2/3 points
        return [num_layers // 3, 2 * num_layers // 3]
    
    elif num_layers <= 12:
        # Medium model: inject at 1/4, 1/2, 3/4 points
        return [
            num_layers // 4,      # Early
            num_layers // 2,      # Middle
            3 * num_layers // 4   # Late
        ]
    
    elif num_layers <= 24:
        # Large model: inject at regular intervals
        return [
            num_layers // 6,      # Early
            num_layers // 3,      # Early-middle
            num_layers // 2,      # Middle
            2 * num_layers // 3,  # Late-middle
            5 * num_layers // 6   # Late
        ]
    
    else:
        # Very large model: inject at 5-6 points
        step = num_layers // 6
        return [step * i for i in range(1, 6)]


# configs/layer_selection.yaml
layer_injection:
  # Automatic selection based on model size
  auto_select: true
  
  # Manual override
  manual_layers: null  # Set to list like [3, 6, 9] to override
  
  # Injection strength per layer
  layer_weights:
    early: 0.3  # Lower weight for early layers
    middle: 1.0  # Full weight for middle layers
    late: 0.5  # Medium weight for late layers
  
  # Injection method
  method: "residual_add"  # Options: residual_add, gated, replace
  
  # Scale factor
  base_scale: 0.5  # Multiplied by layer weight
```

## 6.2 Layer Selection Ablation Tests

```python
# tests/test_layer_selection.py

class TestLayerSelection:
    """Tests to validate optimal layer selection."""
    
    def test_early_layers_only(self):
        """Test injection only at early layers."""
        config = SCIConfig(injection_layers=[2, 3])
        # Expected: OK syntax learning, poor compositional generalization
        
    def test_middle_layers_only(self):
        """Test injection only at middle layers."""
        config = SCIConfig(injection_layers=[5, 6, 7])
        # Expected: Best compositional generalization
        
    def test_late_layers_only(self):
        """Test injection only at late layers."""
        config = SCIConfig(injection_layers=[10, 11])
        # Expected: Poor - too late to influence generation
        
    def test_all_layers(self):
        """Test injection at all layers."""
        config = SCIConfig(injection_layers=list(range(1, 12)))
        # Expected: Potentially unstable, slower training
        
    def test_optimal_selection(self):
        """Test optimal layer selection [3, 6, 9]."""
        config = SCIConfig(injection_layers=[3, 6, 9])
        # Expected: Best balance of all factors
```

---

# PART 7: ABLATION STUDY REQUIREMENTS

## 7.1 Mandatory Ablation Configurations

```yaml
# configs/ablations.yaml

ablations:
  # Full SCI (baseline for comparison)
  - name: "sci_full"
    description: "Full SCI with all components"
    se: true
    ce: true
    al: true
    cbm: true
    scl: true
    orthogonality: true
    
  # Ablate AbstractionLayer
  - name: "no_abstraction_layer"
    description: "SCI without AbstractionLayer (direct input to SE)"
    se: true
    ce: true
    al: false  # ABLATED
    cbm: true
    scl: true
    orthogonality: true
    
  # Ablate SCL
  - name: "no_scl"
    description: "SCI without Structural Contrastive Learning"
    se: true
    ce: true
    al: true
    cbm: true
    scl: false  # ABLATED
    orthogonality: true
    
  # Ablate Content Encoder
  - name: "no_content_encoder"
    description: "SCI without Content Encoder (structure only)"
    se: true
    ce: false  # ABLATED
    al: true
    cbm: true  # Modified to work with structure only
    scl: true
    orthogonality: false  # N/A without CE
    
  # Ablate CBM
  - name: "no_cbm"
    description: "SCI without Causal Binding (simple concatenation)"
    se: true
    ce: true
    al: true
    cbm: false  # ABLATED
    scl: true
    orthogonality: true
    
  # Ablate GNN
  - name: "no_gnn"
    description: "SCI without GNN (no causal graph reasoning)"
    se: true
    ce: true
    al: true
    cbm: true
    scl: true
    orthogonality: true
    se_gnn_layers: 0  # No GNN
    
  # Ablate orthogonality
  - name: "no_orthogonality"
    description: "SCI without orthogonality constraint"
    se: true
    ce: true
    al: true
    cbm: true
    scl: true
    orthogonality: false  # ABLATED
```

## 7.2 Ablation Execution Script

```python
# scripts/run_ablations.py

import os
import yaml
from train import train_model
from evaluate import evaluate_model

def run_ablations(config_path, output_dir):
    """Run all ablation experiments."""
    
    with open(config_path, 'r') as f:
        ablation_configs = yaml.safe_load(f)['ablations']
    
    results = {}
    
    for ablation in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running ablation: {ablation['name']}")
        print(f"Description: {ablation['description']}")
        print(f"{'='*60}\n")
        
        # Create output directory
        ablation_dir = os.path.join(output_dir, ablation['name'])
        os.makedirs(ablation_dir, exist_ok=True)
        
        # Train model with ablation config
        model, metrics_history = train_model(
            config=ablation,
            output_dir=ablation_dir
        )
        
        # Evaluate on all splits
        eval_results = {}
        
        for split in ['in_dist', 'length_ood', 'template_ood']:
            eval_results[split] = evaluate_model(
                model=model,
                split=split,
                output_dir=ablation_dir
            )
        
        results[ablation['name']] = {
            'config': ablation,
            'eval_results': eval_results,
            'training_history': metrics_history
        }
        
        # Save results
        with open(os.path.join(ablation_dir, 'results.json'), 'w') as f:
            json.dump(results[ablation['name']], f, indent=2)
    
    # Generate comparison table
    generate_ablation_table(results, output_dir)
    
    return results


def generate_ablation_table(results, output_dir):
    """Generate ablation comparison table for paper."""
    
    table = []
    table.append("| Configuration | In-Dist | Length OOD | Template OOD | Δ from Full |")
    table.append("|---------------|---------|------------|--------------|-------------|")
    
    full_score = results['sci_full']['eval_results']['length_ood']['exact_match']
    
    for name, data in results.items():
        in_dist = data['eval_results']['in_dist']['exact_match']
        length_ood = data['eval_results']['length_ood']['exact_match']
        template_ood = data['eval_results']['template_ood']['exact_match']
        delta = length_ood - full_score
        
        table.append(f"| {name} | {in_dist:.1%} | {length_ood:.1%} | {template_ood:.1%} | {delta:+.1%} |")
    
    with open(os.path.join(output_dir, 'ablation_table.md'), 'w') as f:
        f.write('\n'.join(table))
```

---

# PART 8: MASTER CHECKLIST

## Pre-Implementation Checklist
- [ ] Read and understand all 4 SCI Master Implementation Guides
- [ ] Understand SCAN dataset format and evaluation criteria
- [ ] Set up development environment with PyTorch, transformers
- [ ] Verify GPU availability and memory (need 24GB+ for full training)
- [ ] Create project directory structure as specified

## Module Implementation Checklist
- [ ] AbstractionLayer implemented and tested
- [ ] StructuralEncoder implemented and tested
- [ ] ContentEncoder implemented and tested
- [ ] CausalBindingMechanism implemented and tested
- [ ] SCLLoss implemented and tested
- [ ] OrthogonalityLoss implemented and tested
- [ ] All hooks registered and verified
- [ ] All tensor shapes verified
- [ ] All gradient flows verified

## Data Preparation Checklist
- [ ] SCAN dataset downloaded
- [ ] Length split created correctly (train ≤22, test >22)
- [ ] Template split created correctly
- [ ] Data leakage checker implemented and passed
- [ ] SCL pair generator implemented and tested
- [ ] DataLoader with proper collation implemented
- [ ] EOS token handling verified

## Training Setup Checklist
- [ ] Hyperparameters configured as specified
- [ ] Optimizer groups set up (base LR vs SCI LR)
- [ ] LR scheduler configured (cosine with warmup)
- [ ] Loss weights configured with warmup schedule
- [ ] Gradient clipping enabled
- [ ] Mixed precision configured
- [ ] Checkpoint manager implemented
- [ ] Training logger implemented
- [ ] Early stopping implemented
- [ ] Overfitting detector implemented

## Fairness Checklist
- [ ] Baseline model prepared with identical config
- [ ] Same training steps for both
- [ ] Same effective batch size for both
- [ ] Same random seeds (42, 123, 456)
- [ ] Same evaluation script for both
- [ ] Same generation config for both

## Evaluation Checklist
- [ ] In-distribution evaluation implemented
- [ ] Length OOD evaluation implemented
- [ ] Template OOD evaluation implemented
- [ ] Exact match metric implemented
- [ ] Token accuracy metric implemented
- [ ] Length accuracy metric implemented
- [ ] Error analysis logging implemented

## Ablation Checklist
- [ ] Full SCI ablation run
- [ ] No AbstractionLayer ablation run
- [ ] No SCL ablation run
- [ ] No ContentEncoder ablation run
- [ ] No CBM ablation run
- [ ] No GNN ablation run
- [ ] No Orthogonality ablation run
- [ ] Ablation table generated

## Final Verification Checklist
- [ ] All tests pass
- [ ] Training completes without errors
- [ ] Val exact match > 95% (sanity check)
- [ ] Length OOD exact match > 80% (target)
- [ ] Template OOD exact match > 85% (target)
- [ ] Training logs complete and reproducible
- [ ] Checkpoints saved and loadable
- [ ] Results comparable to baseline with fair comparison

---

# APPENDIX: Quick Reference Commands

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test module
pytest tests/test_structural_encoder.py -v

# Train SCI (full)
python train.py --config configs/sci_optimal.yaml --output_dir outputs/sci_full

# Train baseline
python train.py --config configs/baseline.yaml --output_dir outputs/baseline

# Resume training
python train.py --config configs/sci_optimal.yaml --resume_from_checkpoint outputs/sci_full/checkpoints/latest

# Evaluate
python evaluate.py --model_path outputs/sci_full/best_model --split length_ood

# Run ablations
python scripts/run_ablations.py --config configs/ablations.yaml --output_dir outputs/ablations

# Generate ablation table
python scripts/generate_tables.py --results_dir outputs/ablations

# Check data leakage
python scripts/check_data_leakage.py --data_dir data/scan/length_split

# Verify hooks
python scripts/verify_hooks.py --model_path outputs/sci_full/best_model
```

---

**END OF DOCUMENT**

This document version: 1.0
Last updated: {current_date}
Author: AI Engineering Standards Team

The AI agent MUST confirm completion of ALL checklist items before declaring the SCI implementation complete.
