# SCI Hyperparameter Deep Dive: Stable Training for Long Sequences

## Document Purpose
This document provides in-depth reasoning for SCI hyperparameter selection, with special focus on:
1. Stable training for short AND long sequences (up to 288 tokens)
2. Pattern learning consistency across sequence lengths
3. EOS token prediction for exact match
4. Preventing training collapse and local minima

---

# SECTION 1: THE SCAN LENGTH CHALLENGE

## 1.1 Understanding the Problem

```
SCAN Length Split Statistics:
├── Training: Sequences with ≤22 action outputs
│   ├── Min output length: 1 token (e.g., "walk" → "WALK")
│   ├── Max output length: ~88 tokens
│   └── Mean output length: ~12 tokens
│
└── Test (OOD): Sequences with >22 action outputs  
    ├── Min output length: ~92 tokens
    ├── Max output length: 288 tokens
    └── Mean output length: ~150 tokens

LENGTH GENERALIZATION CHALLENGE:
- Model trains on SHORT sequences (mean=12)
- Model tests on LONG sequences (mean=150)
- Must extrapolate 12.5x beyond training distribution
```

## 1.2 Why This Is Hard

```python
"""
CORE CHALLENGES FOR LENGTH GENERALIZATION:

1. POSITIONAL ENCODING EXTRAPOLATION
   - Standard sinusoidal/learned positions don't extrapolate well
   - Model may never see positions >88 during training
   - Test requires generating at positions up to 288

2. ATTENTION PATTERN COLLAPSE
   - Self-attention can collapse for very long sequences
   - Early tokens may lose influence on late tokens
   - Repetitive patterns ("WALK WALK WALK...") hard to maintain

3. EOS PREDICTION ACCURACY
   - Must predict EXACTLY when to stop
   - One token early = wrong answer
   - One token late = wrong answer
   - No partial credit for almost-right length

4. STRUCTURAL PATTERN LEARNING
   - "twice" = repeat action 2 times → OK for short sequences
   - "around right thrice and opposite left twice" → Complex nesting
   - Must learn pattern INDEPENDENTLY of length

5. CONTENT CONSISTENCY
   - "WALK" at position 1 must mean same as "WALK" at position 287
   - Content encoder must be length-invariant
   - Binding must work at any position
"""
```

---

# SECTION 2: HYPERPARAMETER CONFIGURATION FOR LENGTH INVARIANCE

## 2.1 Positional Encoding Strategy

```yaml
# CRITICAL: Positional encoding for length generalization

positional_encoding:
  # Option 1: Rotary Position Embeddings (RECOMMENDED)
  type: "rotary"  # RoPE from RoFormer
  rotary_config:
    dim: 64  # Per-head dimension
    base: 10000  # Base for rotation frequencies
    # RoPE naturally extrapolates to unseen positions
    
  # Option 2: ALiBi (Alternative)
  # type: "alibi"
  # alibi_config:
  #   slopes: "geometric"  # Geometric sequence of slopes
  #   Advantage: No learned positions, perfect extrapolation
  #   Disadvantage: May hurt short-range patterns
  
  # Option 3: Relative Position Bias
  # type: "relative_bias"
  # relative_config:
  #   max_distance: 512  # Clip distances beyond this
  #   num_buckets: 32  # Bucket similar distances
  
  # DO NOT USE: Standard learned embeddings
  # type: "learned"  # WILL FAIL on length generalization
  
  # WHY ROTARY/ALIBI:
  # - No learned position parameters that overfit to training lengths
  # - Relative position information generalizes to any length
  # - Structural patterns don't depend on absolute position
```

## 2.2 Structural Encoder Hyperparameters

```yaml
structural_encoder:
  # NUMBER OF SLOTS (Critical for pattern capture)
  num_slots: 8
  # REASONING:
  # - SCAN structures rarely need >6 slots
  # - Example: "jump around left twice and walk opposite right thrice"
  #   Needs: [ACTION1, MODIFIER1, DIR1, REPEAT1, AND, ACTION2, MODIFIER2, DIR2, REPEAT2]
  #   = 9 elements, but some share slots (REPEAT1, REPEAT2 share "repeat" slot)
  # - 8 slots provides buffer without unnecessary complexity
  # - More slots = more parameters = harder to train
  
  # SLOT QUERY INITIALIZATION
  slot_init: "xavier_uniform"
  slot_init_scale: 0.02  # Small init for stable training
  # REASONING:
  # - Large init causes attention to saturate early
  # - Small init allows gradual learning of slot specialization
  
  # CROSS-ATTENTION CONFIGURATION
  cross_attention:
    num_heads: 8
    head_dim: 96  # hidden_dim / num_heads
    dropout: 0.1
    # CRITICAL: Use causal attention bias
    attention_bias: true
    
  # GNN CONFIGURATION
  gnn:
    num_layers: 2
    # REASONING:
    # - 2 layers allows 2-hop reasoning in causal graph
    # - More layers can cause oversmoothing
    # - Causal graphs in SCAN are shallow
    
    hidden_dim: 768
    edge_dim: 64
    residual: true  # Residual connections for stability
    layer_norm: true  # LayerNorm for training stability
```

## 2.3 AbstractionLayer Hyperparameters

```yaml
abstraction_layer:
  # FEATURE DIMENSIONS
  num_features: 8
  # REASONING:
  # - 8 features capture different aspects of structuralness
  # - Dimension 1: Is it a function word? (twice, and, thrice)
  # - Dimension 2: Is it a direction? (left, right)
  # - Dimension 3: Is it an action? (walk, run, jump)
  # - Etc.
  # - Too few features = can't distinguish structure/content
  # - Too many features = overfitting
  
  # STRUCTURALNESS TEMPERATURE
  temperature: 1.0
  learned_temperature: true
  temperature_min: 0.5
  temperature_max: 2.0
  # REASONING:
  # - Temperature controls sharpness of structure/content separation
  # - Low temperature (0.5) = hard decisions (0 or 1)
  # - High temperature (2.0) = soft decisions (smooth between 0 and 1)
  # - Learning temperature allows model to find optimal sharpness
  # - Clamping prevents collapse (too sharp) or uselessness (too soft)
  
  # FEATURE AGGREGATION
  aggregation: "weighted_mean"
  # Options: max, mean, weighted_mean, attention
  # REASONING:
  # - Weighted mean allows different features to contribute differently
  # - Max is too aggressive (loses information)
  # - Plain mean doesn't distinguish important features
  
  # STRUCTURAL SCORE INITIALIZATION
  score_bias_init: 0.0  # Start neutral
  # REASONING:
  # - Starting at 0.5 for all tokens
  # - Model learns which are structure vs content
  # - Bias towards structure (positive init) or content (negative) 
  #   could hurt learning
```

## 2.4 Content Encoder Hyperparameters

```yaml
content_encoder:
  # NUMBER OF LAYERS
  num_layers: 2
  # REASONING:
  # - Lightweight: content extraction doesn't need deep processing
  # - 2 layers enough to refine shared embeddings
  # - More layers = more parameters = slower training
  
  # LAYER CONFIGURATION
  layer_config:
    hidden_dim: 768
    num_heads: 8
    mlp_ratio: 4  # MLP hidden = 4 * hidden_dim = 3072
    dropout: 0.1
    
  # SHARED EMBEDDINGS
  use_shared_embeddings: true
  freeze_embeddings: false  # Allow fine-tuning
  # REASONING:
  # - Shared embeddings = content meaning from pretrained model
  # - NOT frozen because content meaning may need task adaptation
  # - Sharing reduces parameters and enforces consistency
  
  # ORTHOGONALITY CONFIGURATION
  orthogonality:
    method: "projection"  # Project content to be ⊥ structure
    # Alternative: "penalty" (soft constraint via loss)
    
    projection_dim: 768
    num_projection_layers: 1  # Linear projection
    
    # For penalty method:
    # penalty_weight: 0.1
    # penalty_schedule: "constant"  # or "warmup"
```

## 2.5 Causal Binding Mechanism Hyperparameters

```yaml
cbm:
  # BINDING ATTENTION
  binding_attention:
    num_heads: 8
    head_dim: 96
    dropout: 0.1
    # Query: structural slots
    # Key/Value: content tokens
    
  # CAUSAL INTERVENTION
  causal_intervention:
    enabled: true
    method: "do_calculus"  # or "soft_intervention"
    intervention_strength: 1.0
    # REASONING:
    # - do_calculus: Hard intervention following Pearl's framework
    # - soft_intervention: Differentiable approximation
    # - strength=1.0 means full intervention
    
  # BINDING ITERATIONS
  num_iterations: 3
  # REASONING:
  # - Multiple iterations allow refinement of binding
  # - Iteration 1: Initial binding (noisy)
  # - Iteration 2: Refine based on causal graph
  # - Iteration 3: Final binding
  # - More iterations help for complex structures
  # - Diminishing returns after 3
  
  # BROADCAST CONFIGURATION
  broadcast:
    method: "attention_weighted"
    # Options: uniform, attention_weighted, learned
    # REASONING:
    # - attention_weighted: Tokens get slot info proportional to attention
    # - Allows smooth transition between slot influences
    
  # INJECTION SCALE
  injection_scale: 0.5
  injection_method: "residual_add"
  # REASONING:
  # - Scale <1.0 prevents CBM from dominating base model
  # - 0.5 provides balanced contribution
  # - residual_add preserves base model knowledge
```

---

# SECTION 3: TRAINING DYNAMICS CONFIGURATION

## 3.1 Loss Function Weights

```yaml
loss_weights:
  # TASK LOSS (Cross-entropy for generation)
  task: 1.0  # Always weight 1.0 (baseline)
  
  # SCL LOSS
  scl:
    initial: 0.0
    final: 0.3
    warmup_epochs: 2
    warmup_schedule: "linear"
    # REASONING:
    # - Start at 0 to let task loss stabilize
    # - Increase to 0.3 for structural learning
    # - Linear warmup prevents sudden gradient shock
    # - 0.3 balances structure learning with task performance
    
  # ORTHOGONALITY LOSS
  orthogonality:
    weight: 0.1
    warmup_epochs: 0  # Apply from start
    # REASONING:
    # - Low weight (0.1) as soft constraint
    # - Applied from start to prevent structure-content entanglement
    
  # EOS PREDICTION LOSS
  eos:
    weight: 2.0
    # REASONING:
    # - CRITICAL for exact match
    # - Higher weight emphasizes stopping at correct position
    # - Model must learn exact length, not approximate
    
  # LENGTH PREDICTION (Auxiliary)
  length_prediction:
    enabled: true
    weight: 0.05
    # REASONING:
    # - Auxiliary task helps model understand expected length
    # - Low weight (0.05) doesn't dominate main task
    # - Predicts output length from structural slots
```

## 3.2 Learning Rate Configuration

```yaml
learning_rate:
  # BASE MODEL LR
  base_lr: 2e-5
  # REASONING:
  # - Standard for transformer fine-tuning
  # - Lower preserves pretrained knowledge
  # - Too high (>5e-5) causes catastrophic forgetting
  
  # SCI MODULE LR
  sci_lr: 5e-5
  # REASONING:
  # - 2.5x base LR (randomly initialized modules need faster learning)
  # - Ensures SCI catches up to pretrained base
  # - AbstractionLayer needs to learn quickly to be useful
  
  # SEPARATE LR FOR DIFFERENT SCI COMPONENTS
  component_lr_multipliers:
    abstraction_layer: 2.0  # 10e-5 effective
    structural_encoder: 1.0  # 5e-5 effective
    content_encoder: 1.0  # 5e-5 effective
    cbm: 1.0  # 5e-5 effective
    slot_queries: 0.5  # 2.5e-5 effective (slower for stability)
  # REASONING:
  # - AbstractionLayer needs fast learning (critical component)
  # - Slot queries need slow learning (prevent collapse)
  
  # SCHEDULER
  scheduler: "cosine_with_warmup"
  warmup_ratio: 0.1  # 10% of total steps
  min_lr_ratio: 0.01  # Final LR = 1% of peak
  # REASONING:
  # - Warmup prevents early instability
  # - Cosine provides smooth decay
  # - Low minimum LR allows final convergence
  
  # WARMUP SPECIFICS
  warmup:
    type: "linear"
    start_factor: 0.1  # Start at 10% of target LR
    # REASONING:
    # - Very low starting LR for stability
    # - Linear increase to target
```

## 3.3 Batch and Gradient Configuration

```yaml
batch_config:
  # BATCH SIZE
  batch_size: 8
  # REASONING:
  # - Fits in 24GB GPU memory with long sequences
  # - Larger batches more stable but memory-limited
  
  # GRADIENT ACCUMULATION
  gradient_accumulation_steps: 4
  # REASONING:
  # - Effective batch size = 8 * 4 = 32
  # - 32 is good for SCL (need enough negatives in batch)
  # - Larger effective batch = more stable gradients
  
  # GRADIENT CLIPPING
  max_grad_norm: 1.0
  # REASONING:
  # - Prevents gradient explosion
  # - Critical for SCL loss (can have large gradients)
  # - 1.0 is standard for transformers

gradient_checkpointing:
  enabled: true
  # REASONING:
  # - Saves memory for long sequences
  # - Allows larger batches
  # - Small compute overhead (~20%)
```

## 3.4 Preventing Training Collapse

```yaml
anti_collapse_measures:
  # TEMPERATURE CLAMPING
  min_temperature: 0.5
  max_temperature: 2.0
  # Prevents: AbstractionLayer collapsing to all-0 or all-1 scores
  
  # SLOT DIVERSITY LOSS
  slot_diversity:
    enabled: true
    weight: 0.01
    min_entropy: 1.0
    # Prevents: All slots learning same representation
    # Encourages: Slot specialization
  
  # EDGE SPARSITY
  edge_sparsity:
    enabled: true
    weight: 0.01
    target_sparsity: 0.5
    # Prevents: Fully connected causal graph (no structure)
    # Encourages: Sparse, meaningful causal edges
  
  # BINDING ENTROPY
  binding_entropy:
    enabled: true
    weight: 0.01
    max_entropy: 3.0
    # Prevents: All content binding to single slot
    # Encourages: Distributed binding across relevant slots
  
  # GRADIENT MONITORING
  gradient_monitoring:
    enabled: true
    log_norms: true
    alert_threshold: 100.0  # Alert if gradient norm > 100
    # Prevents: Undetected gradient explosion
  
  # LOSS SPIKE DETECTION
  loss_spike_detection:
    enabled: true
    spike_threshold: 3.0  # Loss > 3x recent average
    action: "rollback"  # or "skip", "reduce_lr"
    # Prevents: Training derailment from bad batches
```

---

# SECTION 4: LENGTH-SPECIFIC TRAINING STRATEGIES

## 4.1 Curriculum Learning for Length

```yaml
curriculum_learning:
  enabled: true
  
  # LENGTH CURRICULUM
  length_curriculum:
    # Phase 1: Short sequences only
    phase1:
      epochs: [0, 3]  # Epochs 0-3
      max_output_length: 20
      description: "Learn basic patterns on short outputs"
      
    # Phase 2: Medium sequences
    phase2:
      epochs: [3, 8]  # Epochs 3-8
      max_output_length: 50
      description: "Extend to medium-length outputs"
      
    # Phase 3: All training sequences
    phase3:
      epochs: [8, 20]  # Epochs 8-20
      max_output_length: 88  # Full training distribution
      description: "Full training distribution"
  
  # STRUCTURE COMPLEXITY CURRICULUM
  structure_curriculum:
    # Phase 1: Simple structures
    phase1:
      epochs: [0, 5]
      max_clauses: 2  # "X twice", "X and Y"
      
    # Phase 2: Medium complexity
    phase2:
      epochs: [5, 12]
      max_clauses: 4  # "X twice and Y opposite left"
      
    # Phase 3: Full complexity
    phase3:
      epochs: [12, 20]
      max_clauses: null  # All structures
```

## 4.2 Length-Aware Data Augmentation

```yaml
data_augmentation:
  # LENGTH AUGMENTATION
  length_augmentation:
    enabled: true
    
    # Duplicate short sequences with different formatting
    short_sequence_oversampling: 2.0  # 2x oversample sequences <10 tokens
    
    # Add positional noise to encourage length invariance
    positional_noise:
      enabled: true
      noise_std: 0.01
      # Adds small noise to positional encodings
      # Forces model to not rely too heavily on exact positions
  
  # STRUCTURE-PRESERVING AUGMENTATION
  structure_augmentation:
    enabled: true
    
    # Swap content words while preserving structure
    content_swapping:
      enabled: true
      swap_rate: 0.3  # 30% of training examples
      # "walk twice" → "run twice" (same structure)
    
    # Add equivalent structures
    structure_synonyms:
      enabled: true
      # "X twice" → "X two times" (if vocabulary supports)
```

## 4.3 EOS Token Training Strategy

```yaml
eos_training:
  # EOS TOKEN CONFIGURATION
  eos_token_id: 2  # Usually <eos> or </s>
  
  # TRAINING TARGETS
  target_format: "output<eos>"  # Always end with EOS
  
  # EOS LOSS BOOSTING
  eos_loss_weight: 2.0
  # REASONING:
  # - Critical for exact match
  # - Model must learn precise stopping point
  
  # EOS POSITION SUPERVISION
  eos_position_auxiliary:
    enabled: true
    predict_length: true  # Predict expected output length
    predict_eos_position: true  # Predict position of EOS
    auxiliary_weight: 0.05
  
  # EOS SAMPLING DURING TRAINING
  teacher_forcing_eos:
    enabled: true
    # Use teacher forcing for EOS token
    # Model sees correct EOS position during training
  
  # EOS GENERATION STRATEGY
  generation:
    enforce_eos: true  # Stop immediately at EOS
    eos_threshold: 0.5  # Generate EOS if P(EOS) > 0.5
    max_length_buffer: 10  # Stop at max_expected_length + 10
```

---

# SECTION 5: MONITORING & DEBUGGING

## 5.1 Training Metrics to Monitor

```yaml
monitoring:
  # LOG EVERY N STEPS
  log_interval: 10
  
  # METRICS TO LOG
  metrics:
    # Loss components
    - total_loss
    - task_loss
    - scl_loss
    - orthogonality_loss
    - eos_loss
    
    # AbstractionLayer metrics
    - content_word_avg_score
    - structure_word_avg_score
    - score_separation_ratio
    
    # Structural encoder metrics
    - slot_utilization  # How many slots are actively used
    - slot_entropy  # Diversity of slot representations
    - edge_sparsity  # Sparsity of causal graph
    
    # Content encoder metrics
    - content_structure_cosine  # Should be ~0
    
    # CBM metrics
    - binding_entropy
    - binding_confidence
    
    # Generation metrics (during validation)
    - exact_match
    - token_accuracy
    - length_accuracy
    - eos_accuracy
    
    # Training dynamics
    - learning_rate
    - gradient_norm
    - gpu_memory_used
    
  # VISUALIZATION
  visualize:
    - attention_heatmaps  # Every 500 steps
    - structuralness_scores  # Every 500 steps
    - slot_embeddings_tsne  # Every epoch
    - loss_curves  # Real-time
```

## 5.2 Debugging Checkpoints

```python
"""
DEBUGGING CHECKPOINTS: Run these checks at specific training stages.

AFTER 100 STEPS:
- Verify AbstractionLayer scores are changing
- Verify gradients are non-zero for all SCI modules
- Verify loss is decreasing

AFTER 1 EPOCH:
- Verify content/structure scores are separating
- Verify slot utilization > 50%
- Verify SCL loss is meaningful (not NaN/Inf)

AFTER 5 EPOCHS:
- Verify val exact match > 80% (in-distribution)
- Verify structure words have score > 0.7
- Verify content words have score < 0.3
- Verify cosine(structure, content) < 0.2

AFTER 10 EPOCHS:
- Verify val exact match > 95% (in-distribution)
- Verify structural invariance: same-structure sim > 0.9
- Verify EOS prediction accuracy > 90%
"""

class DebugCheckpoints:
    def check_step_100(self, trainer):
        """Checks after 100 training steps."""
        metrics = trainer.get_metrics()
        
        # Check scores are changing
        assert metrics['score_std'] > 0.01, "AbstractionLayer scores not changing"
        
        # Check gradients
        for name, param in trainer.model.named_parameters():
            if 'sci' in name.lower() and param.grad is not None:
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
        
        # Check loss decreasing
        assert metrics['loss'] < metrics['initial_loss'], "Loss not decreasing"
        
    def check_epoch_1(self, trainer):
        """Checks after 1 epoch."""
        metrics = trainer.get_metrics()
        
        # Check score separation
        assert metrics['score_separation_ratio'] > 1.2, \
            "Content/structure scores not separating"
        
        # Check slot utilization
        assert metrics['slot_utilization'] > 0.5, \
            "Slots not being utilized"
        
    def check_epoch_5(self, trainer):
        """Checks after 5 epochs."""
        metrics = trainer.get_metrics()
        
        # Check in-distribution performance
        assert metrics['val_exact_match'] > 0.80, \
            f"Poor in-distribution: {metrics['val_exact_match']}"
        
        # Check score separation
        assert metrics['content_avg_score'] < 0.3, \
            f"Content scores too high: {metrics['content_avg_score']}"
        assert metrics['structure_avg_score'] > 0.7, \
            f"Structure scores too low: {metrics['structure_avg_score']}"
```

---

# SECTION 6: FINAL CONFIGURATION FILE

```yaml
# configs/sci_production.yaml
# PRODUCTION-READY CONFIGURATION FOR SCAN LENGTH SPLIT

experiment:
  name: "sci_scan_length"
  description: "SCI for SCAN length generalization"
  seeds: [42, 123, 456]
  
model:
  base_model: "google/t5-small"
  max_length: 512
  
  # Positional encoding (CRITICAL)
  positional_encoding:
    type: "rotary"
    base: 10000
    
sci:
  # Structural Encoder
  structural_encoder:
    num_slots: 8
    num_gnn_layers: 2
    hidden_dim: 768
    dropout: 0.1
    
  # AbstractionLayer
  abstraction_layer:
    num_features: 8
    temperature: 1.0
    learned_temperature: true
    
  # Content Encoder  
  content_encoder:
    num_layers: 2
    use_shared_embeddings: true
    
  # CBM
  cbm:
    num_heads: 8
    num_iterations: 3
    injection_scale: 0.5
    
  # Injection
  injection_layers: [3, 6, 9]
  
training:
  # Optimizer
  optimizer: "adamw"
  base_lr: 2.0e-5
  sci_lr: 5.0e-5
  weight_decay: 0.01
  
  # Schedule
  scheduler: "cosine_with_warmup"
  warmup_ratio: 0.1
  min_lr_ratio: 0.01
  
  # Batch
  batch_size: 8
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  
  # Duration
  num_epochs: 20
  
  # Loss weights
  loss_weights:
    task: 1.0
    scl: 0.3
    scl_warmup_epochs: 2
    orthogonality: 0.1
    eos: 2.0
    
  # Early stopping
  early_stopping:
    patience: 5
    min_delta: 0.001
    metric: "val_exact_match"
    
  # Checkpointing
  checkpoint_steps: 1000
  save_total_limit: 3
  
evaluation:
  # Generation
  max_new_tokens: 300
  do_sample: false
  temperature: 1.0
  num_beams: 1
  repetition_penalty: 1.0
  length_penalty: 1.0
  
  # Metrics
  metrics: ["exact_match", "token_accuracy", "length_accuracy"]
  
logging:
  log_steps: 10
  tensorboard: true
  wandb:
    enabled: true
    project: "sci-scan-length"
```

---

# APPENDIX: Hyperparameter Tuning Guide

## If Model Doesn't Converge
```yaml
adjustments:
  - increase warmup_ratio: 0.15
  - decrease sci_lr: 3.0e-5
  - increase scl_warmup_epochs: 3
  - check gradient norms (should be <10)
```

## If Model Overfits
```yaml
adjustments:
  - increase dropout: 0.2
  - decrease num_epochs: 15
  - increase weight_decay: 0.02
  - enable gradient checkpointing
```

## If Length Generalization Fails
```yaml
adjustments:
  - verify rotary positional encoding is active
  - increase num_slots: 10
  - enable length curriculum learning
  - increase eos_loss_weight: 3.0
  - check EOS token handling in data collator
```

## If AbstractionLayer Doesn't Separate
```yaml
adjustments:
  - increase abstraction_lr: 1.5e-4
  - decrease temperature: 0.5
  - add more training epochs
  - verify SCL pairs are correct
```

## If Training Is Unstable
```yaml
adjustments:
  - decrease sci_lr: 2.0e-5
  - increase warmup_ratio: 0.15
  - enable loss spike detection
  - decrease injection_scale: 0.3
```

---

**END OF HYPERPARAMETER DOCUMENT**
