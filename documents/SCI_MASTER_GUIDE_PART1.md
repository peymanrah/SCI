# SCI IMPLEMENTATION MASTER GUIDE FOR AI AGENTS

## ⚠️ CRITICAL READING INSTRUCTIONS FOR AI AGENT

**READ THIS ENTIRE DOCUMENT BEFORE WRITING ANY CODE.**

This guide provides EXACT, LINE-BY-LINE instructions for implementing SCI (Structural Causal Invariance). You MUST follow these instructions precisely. Do NOT:
- Skip any component
- Simplify the architecture
- Merge separate modules
- Ignore verification tests
- Proceed to training without passing all tests

**IMPLEMENTATION ORDER IS CRITICAL.** Follow the numbered steps exactly.

---

## TABLE OF CONTENTS

1. [Project Overview and Theory](#1-project-overview)
2. [Benchmark Selection Rationale](#2-benchmark-selection)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Configuration System](#5-configuration-system)
6. [Core Architecture Implementation](#6-core-architecture)
7. [Automated Pair Generation](#7-pair-generation)
8. [Data Pipeline](#8-data-pipeline)
9. [Training Pipeline](#9-training-pipeline)
10. [Evaluation Pipeline](#10-evaluation-pipeline)
11. [Data Leakage Prevention](#11-data-leakage)
12. [Architecture Verification Tests](#12-verification-tests)
13. [Training Regime](#13-training-regime)
14. [Expected Results](#14-expected-results)

---

## 1. PROJECT OVERVIEW

### 1.1 What is SCI?

SCI (Structural Causal Invariance) is a novel architecture that learns to separate STRUCTURE from CONTENT in compositional tasks.

**Core Principle:** For inputs with the SAME structure but DIFFERENT content, the structural representation should be IDENTICAL.

Example:
- "walk twice" and "run twice" have SAME structure: "ACTION twice"
- "walk twice" and "walk and run" have DIFFERENT structure

### 1.2 Why This Matters

Current LLMs fail at compositional generalization because they entangle structure and content. SCI explicitly separates them, enabling:
- Generalization to longer sequences (SCAN length split)
- Generalization to new structural combinations (SCAN template split)
- Transfer across similar compositional tasks

### 1.3 Four Core Components

1. **Structural Encoder (SE):** Extracts structure-only representation
2. **Content Encoder (CE):** Extracts role-agnostic content representation  
3. **Causal Binding Mechanism (CBM):** Combines structure + content for generation
4. **Structural Contrastive Loss (SCL):** Training objective enforcing invariance

**ALL FOUR COMPONENTS ARE REQUIRED.** Do not skip any.

---

## 2. BENCHMARK SELECTION RATIONALE

### 2.1 Why SCAN is the PRIMARY Training Benchmark

SCAN is ideal for SCI because:

1. **Known Grammar:** We can programmatically determine structural equivalence
2. **Clear Compositional Structure:** Commands decompose into structure + content
3. **Standard Benchmark:** Direct comparison with prior work
4. **Automatic Pair Generation:** No human annotation needed

SCAN Grammar:
```
command → action | action direction | command "twice" | command "thrice" | 
          command "and" command | command "after" command
action → "walk" | "run" | "jump" | "look"
direction → "left" | "right" | "around left" | "around right" | "opposite left" | "opposite right"
```

### 2.2 Evaluation Strategy

| Phase | Dataset | Split | Purpose |
|-------|---------|-------|---------|
| Training | SCAN | length (train) | Learn structural invariance |
| In-Dist Eval | SCAN | length (test, short) | Verify basic learning |
| OOD Eval | SCAN | length (test, long) | Test length generalization |
| Comp Gen Eval | SCAN | template_around_right | Test structural generalization |
| Transfer Eval | COGS | gen | Test cross-dataset transfer |

### 2.3 What NOT to Evaluate (Initially)

- **BBH:** Too diverse, no clear structural patterns for pair generation
- **GSM8K:** Math reasoning ≠ compositional structure
- **General NLU:** Not the target of SCI

**These can be evaluated AFTER proving SCI works on compositional tasks.**

---

## 3. PROJECT STRUCTURE

**AI AGENT: Create this EXACT directory structure first.**

```
SCI/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── configs/
│   ├── base_config.yaml              # Shared settings
│   ├── sci_full.yaml                 # Full SCI configuration
│   ├── baseline_no_sci.yaml          # Baseline (fine-tuning only)
│   ├── ablation_no_se.yaml           # Without Structural Encoder
│   ├── ablation_no_ce.yaml           # Without Content Encoder
│   ├── ablation_no_cbm.yaml          # Without Causal Binding
│   ├── ablation_no_scl.yaml          # Without SCL Loss
│   └── ablation_scl_only.yaml        # Only SCL (no CBM injection)
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── abstraction_layer.py      # THE key novel component
│   │   ├── structural_encoder.py     # Structural Encoder
│   │   ├── content_encoder.py        # Content Encoder
│   │   ├── causal_binding.py         # Causal Binding Mechanism
│   │   ├── sci_wrapper.py            # Wraps base model + SCI
│   │   └── hooks.py                  # Forward hooks for CBM injection
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── scl_loss.py              # Structural Contrastive Loss
│   │   ├── orthogonality_loss.py    # Structure-Content orthogonality
│   │   └── combined_loss.py         # All losses combined
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── grammar/
│   │   │   ├── __init__.py
│   │   │   ├── scan_grammar.py      # SCAN grammar parser
│   │   │   └── cogs_grammar.py      # COGS structure extraction
│   │   ├── pair_generator.py        # Automated pair generation
│   │   ├── scan_dataset.py          # SCAN data loading
│   │   ├── cogs_dataset.py          # COGS data loading
│   │   ├── collator.py              # Batch collation with pairs
│   │   └── data_utils.py            # Utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   ├── optimizer.py             # Optimizer configuration
│   │   └── scheduler.py             # LR scheduling
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py             # Main evaluation
│   │   ├── metrics.py               # Accuracy metrics
│   │   ├── invariance_checker.py    # Check structural invariance
│   │   └── generation.py            # Generation utilities
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Config loading/merging
│       ├── logging.py               # Logging setup
│       ├── checkpointing.py         # Save/load models
│       └── seed.py                  # Reproducibility
│
├── scripts/
│   ├── setup_environment.sh         # Environment setup
│   ├── download_data.py             # Download datasets
│   ├── generate_pairs.py            # Pre-generate training pairs
│   ├── verify_implementation.py     # MUST RUN before training
│   ├── train.py                     # Main training script
│   ├── evaluate.py                  # Evaluation script
│   ├── run_ablations.py             # All ablation studies
│   └── analyze_results.py           # Results analysis
│
├── tests/
│   ├── __init__.py
│   ├── test_abstraction_layer.py    # Test AbstractionLayer
│   ├── test_structural_encoder.py   # Test SE outputs
│   ├── test_content_encoder.py      # Test CE outputs
│   ├── test_causal_binding.py       # Test CBM
│   ├── test_scl_loss.py             # Test loss computation
│   ├── test_pair_generation.py      # Test pair generator
│   ├── test_data_leakage.py         # CRITICAL: verify no leakage
│   ├── test_invariance.py           # Test structural invariance
│   ├── test_integration.py          # End-to-end tests
│   └── conftest.py                  # Pytest fixtures
│
├── notebooks/
│   ├── 01_explore_scan.ipynb        # Explore SCAN dataset
│   ├── 02_visualize_structure.ipynb # Visualize SE outputs
│   └── 03_analyze_results.ipynb     # Analyze experiments
│
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── results/
    └── figures/
```

---

## 4. ENVIRONMENT SETUP

### Step 4.1: Create requirements.txt

**AI AGENT: Create file `requirements.txt` with this EXACT content:**

```
# Core ML
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0

# Optimization
bitsandbytes>=0.41.0

# Data processing
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0

# Configuration
pyyaml>=6.0
omegaconf>=2.3.0

# Logging and tracking
wandb>=0.16.0
tensorboard>=2.15.0
tqdm>=4.66.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Code quality
black>=23.0.0
isort>=5.12.0
flake8>=6.1.0

# Utilities
python-dotenv>=1.0.0
rich>=13.0.0
```

### Step 4.2: Create setup.py

**AI AGENT: Create file `setup.py`:**

```python
from setuptools import setup, find_packages

setup(
    name="sci",
    version="0.1.0",
    description="Structural Causal Invariance for Compositional Generalization",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
    },
)
```

### Step 4.3: Create pyproject.toml

**AI AGENT: Create file `pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sci"
version = "0.1.0"
description = "Structural Causal Invariance"
requires-python = ">=3.10"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"
```

### Step 4.4: Setup Script

**AI AGENT: Create file `scripts/setup_environment.sh`:**

```bash
#!/bin/bash
set -e

echo "=== SCI Environment Setup ==="

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo "Activate environment with: source venv/bin/activate"
```

---

## 5. CONFIGURATION SYSTEM

### 5.1 Base Configuration

**AI AGENT: Create file `configs/base_config.yaml`:**

```yaml
# configs/base_config.yaml
# Base configuration - inherited by all other configs

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
model:
  # Base model to use (will be frozen or fine-tuned based on settings)
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  
  # Precision settings
  torch_dtype: "float16"  # float16, bfloat16, float32
  
  # Model loading
  load_in_8bit: false
  load_in_4bit: false
  
  # Gradient checkpointing (saves memory)
  gradient_checkpointing: true

# =============================================================================
# SCI COMPONENT CONFIGURATION
# Each component can be enabled/disabled for ablations
# =============================================================================
sci:
  # Global SCI settings
  enabled: true
  d_model: null  # null = auto-detect from base model
  
  # ---------------------------------------------
  # Structural Encoder (SE)
  # Extracts structure-only representation
  # ---------------------------------------------
  structural_encoder:
    enabled: true
    
    # Number of structural slots (abstract positions in structure)
    n_slots: 8
    
    # Number of GNN layers for structural reasoning
    n_gnn_layers: 2
    
    # Number of attention heads for cross-attention
    n_heads: 4
    
    # Abstraction layer settings (THE KEY INNOVATION)
    abstraction:
      enabled: true  # DO NOT DISABLE
      hidden_multiplier: 2
      residual_init: 0.1  # Initial residual gate value
      dropout: 0.1
    
    # Dropout for cross-attention
    dropout: 0.1
  
  # ---------------------------------------------
  # Content Encoder (CE)
  # Extracts role-agnostic content embeddings
  # ---------------------------------------------
  content_encoder:
    enabled: true
    
    # Number of refiner layers
    n_refiner_layers: 2
    
    # Share embeddings with base model (recommended)
    share_base_embeddings: true
    
    # Orthogonality settings
    orthogonality:
      enabled: true
      projection_type: "gram_schmidt"  # gram_schmidt, learned
  
  # ---------------------------------------------
  # Causal Binding Mechanism (CBM)
  # Combines structure + content for generation
  # ---------------------------------------------
  causal_binding:
    enabled: true
    
    # Which transformer layers to inject CBM
    # For TinyLlama (22 layers): inject at layers 6, 11, 16
    injection_layers: [6, 11, 16]
    
    # Number of message passing iterations
    n_iterations: 3
    
    # Combination method: "gated", "residual", "replace"
    combination: "gated"
  
  # ---------------------------------------------
  # Structural Contrastive Learning (SCL)
  # Training objective for structural invariance
  # ---------------------------------------------
  contrastive_learning:
    enabled: true
    
    # Temperature for InfoNCE loss
    temperature: 0.07
    
    # Weight of SCL loss relative to LM loss
    weight: 0.1
    
    # SCL warmup (gradually increase weight)
    warmup_epochs: 2
    
    # Hard negative mining
    use_hard_negatives: true
    hard_negative_ratio: 0.3
    
    # Minimum positive pairs per batch (for valid SCL)
    min_positives_per_batch: 4

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
training:
  # Dataset settings
  dataset: "scan"
  split: "length"  # length, simple, addprim_jump, template_around_right
  
  # Batch settings
  batch_size: 16
  gradient_accumulation_steps: 2
  effective_batch_size: 32  # batch_size * gradient_accumulation
  
  # Sequence length
  max_seq_length: 128
  
  # Training duration
  num_epochs: 20
  max_steps: null  # null = use num_epochs
  
  # Optimization
  learning_rate: 5.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Learning rate schedule
  lr_scheduler: "linear"  # linear, cosine, constant
  warmup_ratio: 0.1
  
  # Checkpointing
  save_strategy: "epoch"
  save_total_limit: 3
  
  # Evaluation during training
  eval_strategy: "epoch"
  eval_steps: null
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 5
    metric: "eval_accuracy"
    mode: "max"
  
  # Mixed precision
  fp16: true
  bf16: false
  
  # Reproducibility
  seed: 42

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
evaluation:
  # Batch size for evaluation
  batch_size: 32
  
  # Generation settings
  generation:
    max_new_tokens: 64
    do_sample: false  # Greedy decoding for reproducibility
    num_beams: 1
  
  # Datasets to evaluate on
  datasets:
    - name: "scan"
      split: "length"
      subset: "test"
    - name: "scan"
      split: "template_around_right"
      subset: "test"
  
  # Metrics to compute
  metrics:
    - "exact_match"
    - "token_accuracy"

# =============================================================================
# PAIR GENERATION CONFIGURATION
# =============================================================================
pair_generation:
  # Method: "grammar" (for SCAN), "llm" (for other datasets)
  method: "grammar"
  
  # For grammar-based generation
  grammar:
    cache_pairs: true
    cache_path: "outputs/pair_cache"
  
  # For LLM-based generation (if needed)
  llm:
    model: "gpt-4"  # or local model
    prompt_template: "structure_classification"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging:
  # Weights & Biases
  wandb:
    enabled: true
    project: "sci-compositional"
    entity: null  # Your W&B username
    tags: ["sci", "compositional"]
  
  # Console logging
  log_level: "INFO"
  log_every_n_steps: 10
  
  # TensorBoard
  tensorboard:
    enabled: true
    log_dir: "outputs/logs/tensorboard"

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
output:
  # Base output directory
  base_dir: "outputs"
  
  # Subdirectories
  checkpoint_dir: "${output.base_dir}/checkpoints"
  log_dir: "${output.base_dir}/logs"
  results_dir: "${output.base_dir}/results"
  figure_dir: "${output.base_dir}/figures"
  
  # Experiment naming
  experiment_name: "sci_${training.dataset}_${training.split}"
```

### 5.2 Full SCI Configuration

**AI AGENT: Create file `configs/sci_full.yaml`:**

```yaml
# configs/sci_full.yaml
# Full SCI configuration - all components enabled

defaults:
  - base_config

# Override experiment name
output:
  experiment_name: "sci_full_${training.dataset}_${training.split}"

# All SCI components enabled (inherits from base)
sci:
  enabled: true
  structural_encoder:
    enabled: true
  content_encoder:
    enabled: true
  causal_binding:
    enabled: true
  contrastive_learning:
    enabled: true
```

### 5.3 Baseline Configuration (No SCI)

**AI AGENT: Create file `configs/baseline_no_sci.yaml`:**

```yaml
# configs/baseline_no_sci.yaml
# Baseline: Standard fine-tuning WITHOUT any SCI components

defaults:
  - base_config

output:
  experiment_name: "baseline_no_sci_${training.dataset}_${training.split}"

# DISABLE all SCI components
sci:
  enabled: false
  structural_encoder:
    enabled: false
  content_encoder:
    enabled: false
  causal_binding:
    enabled: false
  contrastive_learning:
    enabled: false

# Standard fine-tuning settings
training:
  learning_rate: 2.0e-5  # Lower LR for fine-tuning without SCI
```

### 5.4 Ablation Configurations

**AI AGENT: Create these ablation config files:**

```yaml
# configs/ablation_no_se.yaml
defaults:
  - base_config

output:
  experiment_name: "ablation_no_se_${training.dataset}_${training.split}"

sci:
  enabled: true
  structural_encoder:
    enabled: false  # DISABLED
  content_encoder:
    enabled: true
  causal_binding:
    enabled: true
  contrastive_learning:
    enabled: false  # Requires SE, so also disabled
```

```yaml
# configs/ablation_no_ce.yaml
defaults:
  - base_config

output:
  experiment_name: "ablation_no_ce_${training.dataset}_${training.split}"

sci:
  enabled: true
  structural_encoder:
    enabled: true
  content_encoder:
    enabled: false  # DISABLED
  causal_binding:
    enabled: true   # Uses SE output directly
  contrastive_learning:
    enabled: true
```

```yaml
# configs/ablation_no_cbm.yaml
defaults:
  - base_config

output:
  experiment_name: "ablation_no_cbm_${training.dataset}_${training.split}"

sci:
  enabled: true
  structural_encoder:
    enabled: true
  content_encoder:
    enabled: true
  causal_binding:
    enabled: false  # DISABLED - no injection into transformer
  contrastive_learning:
    enabled: true   # SCL still trains SE
```

```yaml
# configs/ablation_no_scl.yaml
defaults:
  - base_config

output:
  experiment_name: "ablation_no_scl_${training.dataset}_${training.split}"

sci:
  enabled: true
  structural_encoder:
    enabled: true
  content_encoder:
    enabled: true
  causal_binding:
    enabled: true
  contrastive_learning:
    enabled: false  # DISABLED - no structural invariance training
    weight: 0.0
```

---

## 6. CORE ARCHITECTURE IMPLEMENTATION

### 6.1 Abstraction Layer (THE KEY INNOVATION)

**AI AGENT: Create file `src/models/abstraction_layer.py`:**

**THIS IS THE MOST IMPORTANT FILE. DO NOT SIMPLIFY.**

```python
"""
src/models/abstraction_layer.py

THE KEY NOVEL COMPONENT OF SCI.

The AbstractionLayer learns to identify and preserve ONLY structural information
while suppressing content-specific features. This is what enables structural
invariance - the same structure with different content produces identical
structural representations.

HOW IT WORKS:
1. A "structural detector" network produces per-feature "structuralness" scores
2. Features with high scores (structural) are preserved
3. Features with low scores (content) are suppressed
4. Training with SCL causes structural features to have consistent scores
   across inputs with the same structure

DO NOT:
- Remove the structural_detector
- Replace with simple linear layer
- Remove the residual_gate
- Skip the layer normalization

VERIFICATION:
- After training, inputs with same structure should produce similar outputs
- The structuralness scores should be consistent for structural features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AbstractionLayer(nn.Module):
    """
    Learns to separate structural features from content features.
    
    This is the core innovation of SCI. It learns which aspects of the
    representation are structural (invariant to content substitution) and
    which are content-specific.
    
    Args:
        d_model: Hidden dimension
        hidden_multiplier: Multiplier for hidden layer size
        residual_init: Initial value for residual gate (allows content through initially)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_multiplier: int = 2,
        residual_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        hidden_dim = d_model * hidden_multiplier
        
        # ===========================================
        # STRUCTURAL DETECTOR
        # This network learns which features are structural
        # Output is sigmoid, giving "structuralness" score per feature
        # ===========================================
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid()  # CRITICAL: Output in [0, 1]
        )
        
        # ===========================================
        # RESIDUAL GATE
        # Allows some content information through during early training
        # Gradually learns to suppress as SCL trains the detector
        # ===========================================
        self.residual_gate = nn.Parameter(torch.tensor(residual_init))
        
        # ===========================================
        # NORMALIZATION
        # Ensures stable gradients
        # ===========================================
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.structural_detector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_scores: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply structural abstraction.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_scores: If True, also return structuralness scores
            
        Returns:
            abstracted: Tensor with content suppressed [batch_size, seq_len, d_model]
            scores (optional): Structuralness scores [batch_size, seq_len, d_model]
        """
        # Detect which features are structural
        structural_scores = self.structural_detector(x)  # [B, N, D]
        
        # Apply soft masking:
        # - High score features (structural) are preserved
        # - Low score features (content) are suppressed but residual allows some through
        abstracted = (
            x * structural_scores +  # Structural part
            self.residual_gate * x * (1 - structural_scores)  # Residual content
        )
        
        # Normalize for stability
        abstracted = self.layer_norm(abstracted)
        
        if return_scores:
            return abstracted, structural_scores
        return abstracted
    
    def get_structuralness_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get structuralness scores for analysis.
        
        Use this to verify that:
        1. Same-structure inputs have similar score patterns
        2. Different-structure inputs have different score patterns
        """
        with torch.no_grad():
            return self.structural_detector(x)
    
    def get_residual_gate_value(self) -> float:
        """Get current residual gate value (should decrease during training)."""
        return self.residual_gate.item()


class AbstractionLayerWithAttention(nn.Module):
    """
    Enhanced abstraction layer that uses self-attention to identify structural patterns.
    
    This variant uses attention to find relationships between positions,
    which can help identify structural patterns that span multiple tokens.
    
    Use this if the basic AbstractionLayer doesn't achieve good invariance.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        hidden_multiplier: int = 2,
        residual_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Self-attention for structural pattern detection
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Structural detector (applied after attention)
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_multiplier, d_model),
            nn.Sigmoid()
        )
        
        self.residual_gate = nn.Parameter(torch.tensor(residual_init))
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention-based abstraction."""
        # Self-attention to identify structural relationships
        if attention_mask is not None:
            # Convert to attention mask format (True = ignore)
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
            
        attended, _ = self.self_attention(
            x, x, x,
            key_padding_mask=key_padding_mask
        )
        
        # Detect structural features
        structural_scores = self.structural_detector(attended)
        
        # Apply masking with residual
        abstracted = (
            x * structural_scores +
            self.residual_gate * x * (1 - structural_scores)
        )
        
        abstracted = self.layer_norm(abstracted)
        
        if return_scores:
            return abstracted, structural_scores
        return abstracted
```

### 6.2 Structural Encoder

**AI AGENT: Create file `src/models/structural_encoder.py`:**

```python
"""
src/models/structural_encoder.py

Structural Encoder (SE) - Extracts structural representation from input.

Architecture:
1. AbstractionLayer: Filter content, keep structure
2. Structure Queries: Learnable queries for structural slots
3. Cross-Attention: Queries attend to abstracted input
4. Edge Predictor: Predict causal graph between slots
5. Structural GNN: Reason over the graph

Output:
- structural_graph: [B, n_slots, d_model] - Structural slot representations
- edge_weights: [B, n_slots, n_slots] - Causal graph adjacency

CRITICAL REQUIREMENTS:
1. AbstractionLayer MUST be used
2. Cross-attention must respect instruction mask (no data leakage)
3. Edge predictor must produce normalized weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .abstraction_layer import AbstractionLayer


class EdgePredictor(nn.Module):
    """
    Predicts edge weights (causal relationships) between structural slots.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, slot_values: torch.Tensor) -> torch.Tensor:
        """
        Predict edge weights.
        
        Args:
            slot_values: [B, K, D] slot representations
            
        Returns:
            edge_weights: [B, K, K] normalized edge weights
        """
        B, K, D = slot_values.shape
        
        # Create all pairs
        slots_i = slot_values.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, D]
        slots_j = slot_values.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, D]
        
        # Concatenate pairs
        pairs = torch.cat([slots_i, slots_j], dim=-1)  # [B, K, K, 2D]
        
        # Score edges
        edge_scores = self.edge_mlp(pairs).squeeze(-1)  # [B, K, K]
        
        # Normalize (softmax over source dimension)
        edge_weights = F.softmax(edge_scores, dim=2)  # [B, K, K]
        
        return edge_weights


class StructuralGNNLayer(nn.Module):
    """
    Single layer of structural GNN.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Update function (GRU cell)
        self.gru = nn.GRUCell(d_model, d_model)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        slot_values: torch.Tensor,  # [B, K, D]
        edge_weights: torch.Tensor  # [B, K, K]
    ) -> torch.Tensor:
        """Single GNN layer forward pass."""
        B, K, D = slot_values.shape
        
        # Compute messages
        h_i = slot_values.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, D]
        h_j = slot_values.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, D]
        
        messages = self.message_mlp(torch.cat([h_i, h_j], dim=-1))  # [B, K, K, D]
        
        # Weight by edges and aggregate
        weighted_messages = messages * edge_weights.unsqueeze(-1)  # [B, K, K, D]
        aggregated = weighted_messages.sum(dim=2)  # [B, K, D]
        
        # Update with GRU
        h_flat = slot_values.reshape(B * K, D)
        agg_flat = aggregated.reshape(B * K, D)
        updated = self.gru(agg_flat, h_flat).reshape(B, K, D)
        
        # Residual + norm
        return self.layer_norm(slot_values + updated)


class StructuralGNN(nn.Module):
    """
    Multi-layer structural GNN.
    """
    
    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            StructuralGNNLayer(d_model, dropout) for _ in range(n_layers)
        ])
        
    def forward(
        self,
        slot_values: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """Process through all GNN layers."""
        h = slot_values
        for layer in self.layers:
            h = layer(h, edge_weights)
        return h


class StructuralEncoder(nn.Module):
    """
    Complete Structural Encoder.
    
    Extracts structural representation that is invariant to content.
    """
    
    def __init__(
        self,
        d_model: int,
        n_slots: int = 8,
        n_gnn_layers: int = 2,
        n_heads: int = 4,
        abstraction_hidden_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_slots = n_slots
        
        # =========================================
        # ABSTRACTION LAYER - THE KEY COMPONENT
        # DO NOT REMOVE OR SIMPLIFY
        # =========================================
        self.abstraction = AbstractionLayer(
            d_model=d_model,
            hidden_multiplier=abstraction_hidden_mult,
            dropout=dropout
        )
        
        # =========================================
        # STRUCTURE QUERIES
        # Learnable queries that extract structural information
        # =========================================
        self.structure_queries = nn.Parameter(
            torch.randn(n_slots, d_model) * 0.02
        )
        
        # =========================================
        # CROSS-ATTENTION
        # Structure queries attend to abstracted input
        # =========================================
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # =========================================
        # EDGE PREDICTOR
        # Predicts causal graph over structural slots
        # =========================================
        self.edge_predictor = EdgePredictor(d_model, dropout)
        
        # =========================================
        # STRUCTURAL GNN
        # Reasons over the structural graph
        # =========================================
        self.structural_gnn = StructuralGNN(d_model, n_gnn_layers, dropout)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        hidden_states: torch.Tensor,     # [B, N, D]
        attention_mask: torch.Tensor,    # [B, N] - 1 for valid tokens
        return_abstraction_scores: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract structural representation.
        
        Args:
            hidden_states: Input embeddings from tokenizer/embedding layer
            attention_mask: Mask indicating valid tokens (instruction only!)
            return_abstraction_scores: If True, also return abstraction scores
            
        Returns:
            structural_graph: [B, n_slots, d_model] structural representation
            edge_weights: [B, n_slots, n_slots] causal graph
            abstraction_scores (optional): [B, N, d_model] structuralness scores
        """
        B = hidden_states.shape[0]
        
        # Step 1: Apply abstraction to filter out content
        if return_abstraction_scores:
            abstracted, abs_scores = self.abstraction(hidden_states, return_scores=True)
        else:
            abstracted = self.abstraction(hidden_states)
            abs_scores = None
        
        # Step 2: Expand structure queries for batch
        queries = self.structure_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        
        # Step 3: Cross-attention (queries attend to abstracted input)
        # CRITICAL: Use attention_mask to prevent attending to response tokens
        key_padding_mask = (attention_mask == 0)  # True = ignore
        
        slot_values, attn_weights = self.cross_attention(
            query=queries,
            key=abstracted,
            value=abstracted,
            key_padding_mask=key_padding_mask
        )  # slot_values: [B, K, D]
        
        # Step 4: Predict causal edges
        edge_weights = self.edge_predictor(slot_values)  # [B, K, K]
        
        # Step 5: GNN reasoning
        structural_graph = self.structural_gnn(slot_values, edge_weights)  # [B, K, D]
        
        # Step 6: Output projection
        structural_graph = self.output_norm(self.output_proj(structural_graph))
        
        if return_abstraction_scores:
            return structural_graph, edge_weights, abs_scores
        return structural_graph, edge_weights
    
    def get_attention_pattern(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get cross-attention pattern for visualization.
        Shows which input tokens each structural slot attends to.
        """
        B = hidden_states.shape[0]
        abstracted = self.abstraction(hidden_states)
        queries = self.structure_queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = (attention_mask == 0)
        
        _, attn_weights = self.cross_attention(
            query=queries,
            key=abstracted,
            value=abstracted,
            key_padding_mask=key_padding_mask
        )
        
        return attn_weights  # [B, K, N]
```

### 6.3 Content Encoder

**AI AGENT: Create file `src/models/content_encoder.py`:**

```python
"""
src/models/content_encoder.py

Content Encoder (CE) - Extracts role-agnostic content embeddings.

The key property: The same word should have the same content embedding
regardless of its structural role.

Example:
- "walk" in "walk twice" should have same content embedding as
- "walk" in "walk and run"

This is achieved through:
1. Content refiner: Removes any structural information leaking from position
2. Orthogonal projector: Projects content orthogonal to structural subspace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContentRefiner(nn.Module):
    """
    Refines content embeddings to remove structural contamination.
    """
    
    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine content embeddings."""
        for layer, norm in zip(self.layers, self.layer_norms):
            x = norm(x + layer(x))
        return x


class OrthogonalProjector(nn.Module):
    """
    Projects content to be orthogonal to structural representation.
    
    Uses Gram-Schmidt-like projection to remove structural components
    from content embeddings.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Learnable projection for soft orthogonalization
        self.pre_proj = nn.Linear(d_model, d_model)
        self.post_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        content: torch.Tensor,      # [B, N, D]
        structure: torch.Tensor     # [B, K, D]
    ) -> torch.Tensor:
        """
        Project content orthogonal to structural subspace.
        
        Args:
            content: Content embeddings
            structure: Structural slot representations
            
        Returns:
            Orthogonalized content embeddings
        """
        # Apply pre-projection
        content = self.pre_proj(content)
        
        # Project out structural components (Gram-Schmidt style)
        K = structure.shape[1]
        
        for k in range(K):
            s_k = structure[:, k:k+1, :]  # [B, 1, D]
            s_k_norm = F.normalize(s_k, dim=-1)
            
            # Project content onto structural direction
            proj_coeff = (content * s_k_norm).sum(dim=-1, keepdim=True)
            proj = proj_coeff * s_k_norm
            
            # Remove projection
            content = content - proj
            
        # Apply post-projection
        content = self.post_proj(content)
        
        return content


class ContentEncoder(nn.Module):
    """
    Complete Content Encoder.
    
    Produces role-agnostic content embeddings that are orthogonal to structure.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: Optional[int] = None,
        n_refiner_layers: int = 2,
        share_base_embeddings: bool = True,
        base_embeddings: Optional[nn.Embedding] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.share_base_embeddings = share_base_embeddings
        
        # Embeddings
        if share_base_embeddings and base_embeddings is not None:
            self.embeddings = base_embeddings
        else:
            if vocab_size is None:
                raise ValueError("vocab_size required when not sharing embeddings")
            self.embeddings = nn.Embedding(vocab_size, d_model)
            
        # Content refiner
        self.refiner = ContentRefiner(d_model, n_refiner_layers, dropout)
        
        # Orthogonal projector
        self.orthogonal_projector = OrthogonalProjector(d_model)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        input_ids: torch.Tensor,                         # [B, N]
        structural_features: Optional[torch.Tensor] = None,  # [B, K, D]
        attention_mask: Optional[torch.Tensor] = None    # [B, N]
    ) -> torch.Tensor:
        """
        Encode content.
        
        Args:
            input_ids: Token IDs
            structural_features: Structural representation for orthogonalization
            attention_mask: Mask for valid tokens
            
        Returns:
            content: [B, N, D] role-agnostic content embeddings
        """
        # Get embeddings
        content = self.embeddings(input_ids)  # [B, N, D]
        
        # Refine to remove structural contamination
        content = self.refiner(content)
        
        # Project orthogonal to structure
        if structural_features is not None:
            content = self.orthogonal_projector(content, structural_features)
            
        # Normalize
        content = self.output_norm(content)
        
        return content
    
    def compute_orthogonality_loss(
        self,
        content: torch.Tensor,    # [B, N, D]
        structure: torch.Tensor   # [B, K, D]
    ) -> torch.Tensor:
        """
        Compute loss penalizing non-orthogonality.
        
        Returns mean absolute cosine similarity (should be minimized).
        """
        # Average pool
        content_avg = content.mean(dim=1)      # [B, D]
        structure_avg = structure.mean(dim=1)  # [B, D]
        
        # Normalize
        content_norm = F.normalize(content_avg, dim=-1)
        structure_norm = F.normalize(structure_avg, dim=-1)
        
        # Cosine similarity
        cos_sim = (content_norm * structure_norm).sum(dim=-1)  # [B]
        
        # Loss is absolute cosine similarity (want it to be 0)
        loss = cos_sim.abs().mean()
        
        return loss
```

### 6.4 Causal Binding Mechanism

**AI AGENT: Create file `src/models/causal_binding.py`:**

```python
"""
src/models/causal_binding.py

Causal Binding Mechanism (CBM) - Combines structure and content.

This component implements HOW content fills structural slots using
causal intervention (do-operator from causal inference).

Key features:
1. Binding attention: Determines which content fills which slot
2. Causal intervention: Propagates through structural graph
3. Broadcast: Maps slot values back to token positions
4. Gated combination: Merges with transformer hidden states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalInterventionLayer(nn.Module):
    """
    Implements causal intervention (do-operator) for slot values.
    
    Propagates information through the structural causal graph,
    respecting the causal relationships between slots.
    """
    
    def __init__(self, d_model: int, n_iterations: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.n_iterations = n_iterations
        
        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Update function
        self.gru = nn.GRUCell(d_model, d_model)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        slot_values: torch.Tensor,   # [B, K, D]
        edge_weights: torch.Tensor   # [B, K, K]
    ) -> torch.Tensor:
        """
        Apply causal intervention.
        
        Propagates information through the causal graph for n_iterations.
        """
        B, K, D = slot_values.shape
        h = slot_values
        
        for _ in range(self.n_iterations):
            # Compute messages along edges
            h_i = h.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, D]
            h_j = h.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, D]
            
            messages = self.message_fn(torch.cat([h_i, h_j], dim=-1))
            
            # Weight by edges and aggregate
            weighted = messages * edge_weights.unsqueeze(-1)
            aggregated = weighted.sum(dim=2)  # Sum over sources
            
            # Update
            h_flat = h.reshape(B * K, D)
            agg_flat = aggregated.reshape(B * K, D)
            h = self.gru(agg_flat, h_flat).reshape(B, K, D)
            
        return self.layer_norm(h)


class CausalBindingMechanism(nn.Module):
    """
    Complete Causal Binding Mechanism.
    
    Combines structural representation with content to condition
    the transformer's generation.
    """
    
    def __init__(
        self,
        d_model: int,
        n_slots: int,
        n_iterations: int = 3,
        combination: str = "gated",  # gated, residual, replace
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_slots = n_slots
        self.combination = combination
        
        # Binding attention: content -> slots
        self.binding_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Causal intervention
        self.intervention = CausalInterventionLayer(
            d_model, n_iterations, dropout
        )
        
        # Broadcast attention: slots -> tokens
        self.broadcast_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Combination gate
        if combination == "gated":
            self.gate_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        hidden_states: torch.Tensor,     # [B, N, D] from transformer layer
        structural_graph: torch.Tensor,  # [B, K, D] from SE
        edge_weights: torch.Tensor,      # [B, K, K] from SE
        content: torch.Tensor,           # [B, N, D] from CE
        attention_mask: Optional[torch.Tensor] = None  # [B, N]
    ) -> torch.Tensor:
        """
        Apply causal binding.
        
        Args:
            hidden_states: Current transformer hidden states
            structural_graph: Structural representation from SE
            edge_weights: Causal graph from SE
            content: Content representation from CE
            attention_mask: Token-level attention mask
            
        Returns:
            Modified hidden states incorporating structure + content
        """
        B, N, D = hidden_states.shape
        
        # Prepare masks
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        # Step 1: Bind content to structural slots
        # Structural slots query content to get their values
        bound_slots, _ = self.binding_attention(
            query=structural_graph,  # [B, K, D]
            key=content,             # [B, N, D]
            value=content,           # [B, N, D]
            key_padding_mask=key_padding_mask
        )  # [B, K, D]
        
        # Step 2: Causal intervention
        # Propagate through structural graph
        intervened_slots = self.intervention(bound_slots, edge_weights)  # [B, K, D]
        
        # Step 3: Broadcast back to tokens
        # Tokens query slots to get their contributions
        token_contributions, _ = self.broadcast_attention(
            query=hidden_states,     # [B, N, D]
            key=intervened_slots,    # [B, K, D]
            value=intervened_slots   # [B, K, D]
        )  # [B, N, D]
        
        # Step 4: Combine with original hidden states
        if self.combination == "gated":
            combined = torch.cat([hidden_states, token_contributions], dim=-1)
            gate = self.gate_net(combined)  # [B, N, D]
            output = hidden_states * (1 - gate) + token_contributions * gate
        elif self.combination == "residual":
            output = hidden_states + 0.1 * token_contributions
        elif self.combination == "replace":
            output = token_contributions
        else:
            raise ValueError(f"Unknown combination: {self.combination}")
            
        # Project and normalize
        output = self.output_norm(self.output_proj(output))
        
        return output
```

**CONTINUE TO PART 2 for: SCI Wrapper, Losses, Data Pipeline, Pair Generation**
