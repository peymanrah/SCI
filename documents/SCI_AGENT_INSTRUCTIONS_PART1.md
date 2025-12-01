# SCI (Structural Causal Invariance) - Complete AI Agent Implementation Guide

## CRITICAL INSTRUCTIONS FOR AI AGENT (READ FIRST)

This document provides **line-by-line, file-by-file** instructions for implementing SCI. 

**YOU MUST:**
1. Follow the EXACT project structure
2. Implement ALL verification tests BEFORE training
3. Run verification tests after each component implementation
4. NOT simplify or skip any component - each is critical
5. Implement the full architecture, not a stripped-down version

**COMMON MISTAKES TO AVOID:**
- Do NOT skip the AbstractionLayer - it's the key innovation
- Do NOT merge SE and CE into one encoder
- Do NOT implement CBM as simple concatenation
- Do NOT ignore pair_labels in the loss computation
- Do NOT train without verifying structural invariance

---

## Project Structure

```
SCI/
├── README.md                           # Project documentation
├── AGENT_INSTRUCTIONS.md               # This file
├── requirements.txt                    # Dependencies
├── setup.py                           # Package installation
│
├── configs/
│   ├── sci_tinyllama.yaml             # Main SCI configuration
│   ├── baseline.yaml                  # Baseline (no SCI) config
│   ├── ablation_no_se.yaml            # Ablation: without Structural Encoder
│   ├── ablation_no_ce.yaml            # Ablation: without Content Encoder
│   ├── ablation_no_cbm.yaml           # Ablation: without Causal Binding
│   └── ablation_no_scl.yaml           # Ablation: without SCL loss
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── structural_encoder.py      # Structural Encoder (SE)
│   │   ├── content_encoder.py         # Content Encoder (CE)
│   │   ├── causal_binding.py          # Causal Binding Mechanism (CBM)
│   │   └── sci_model.py               # Main SCI model integration
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── structural_contrastive.py  # SCL loss implementation
│   │   └── combined_loss.py           # Combined loss function
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── structure_extractor.py     # Extract structural patterns
│   │   ├── pair_generator.py          # Generate training pairs
│   │   ├── scan_dataset.py            # SCAN dataset with pairs
│   │   └── cogs_dataset.py            # COGS dataset with pairs
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main training loop
│   │   └── scheduler.py               # Learning rate scheduling
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py               # Main evaluation
│   │   ├── metrics.py                 # Accuracy, exact match, etc.
│   │   └── invariance_tests.py        # Structural invariance verification
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Config loading
│       ├── logging_utils.py           # Logging
│       └── checkpoint.py              # Checkpointing
│
├── scripts/
│   ├── download_data.py               # Download datasets
│   ├── train.py                       # Main training script
│   ├── evaluate.py                    # Evaluation script
│   ├── run_ablations.py               # Run all ablation studies
│   └── verify_implementation.py       # Verify SCI implementation
│
├── tests/
│   ├── __init__.py
│   ├── test_structural_encoder.py     # SE unit tests
│   ├── test_content_encoder.py        # CE unit tests
│   ├── test_causal_binding.py         # CBM unit tests
│   ├── test_pair_generator.py         # Pair generation tests
│   ├── test_invariance.py             # Structural invariance tests
│   └── test_data_leakage.py           # Verify no data leakage
│
└── outputs/
    ├── checkpoints/                   # Model checkpoints
    ├── logs/                          # Training logs
    ├── results/                       # Evaluation results
    └── figures/                       # Generated figures
```

---

## Phase 1: Environment Setup

### Step 1.1: Create requirements.txt

```
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pyyaml>=6.0
wandb>=0.15.0
tqdm>=4.65.0
pytest>=7.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Step 1.2: Create setup.py

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="sci",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
    ],
)
```

### Step 1.3: Install and Verify

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Phase 2: Configuration Files

### Step 2.1: Main SCI Configuration (configs/sci_tinyllama.yaml)

**AI AGENT: This config controls ALL aspects of training. Study it carefully.**

```yaml
# configs/sci_tinyllama.yaml

# =========================================
# SCI Configuration for TinyLlama
# =========================================

# Model settings
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype: "float16"
  
# SCI Component Configuration
# CRITICAL: Each component can be enabled/disabled for ablations
sci:
  structural_encoder:
    enabled: true
    n_slots: 8                    # Number of abstract structural slots
    n_layers: 2                   # GNN layers for structural reasoning
    n_heads: 4                    # Attention heads in cross-attention
    abstraction_hidden_mult: 2    # Hidden multiplier in abstraction layer
    
  content_encoder:
    enabled: true
    refiner_layers: 2             # Layers in content refiner
    shared_embeddings: true       # Share embeddings with base model
    orthogonality_weight: 0.01    # Weight for orthogonality loss
    
  causal_binding:
    enabled: true
    injection_layers: [6, 11, 16] # Which transformer layers to inject CBM
    n_iterations: 3               # Message passing iterations
    
  contrastive_learning:
    enabled: true
    temperature: 0.07             # Contrastive loss temperature
    scl_weight: 0.1               # Weight for SCL loss
    use_hard_negatives: true      # Use hard negative mining
    hard_negative_ratio: 0.3      # Ratio of hardest negatives to use
    warmup_epochs: 2              # Epochs before full SCL weight

# Training settings
training:
  # Data
  dataset: "scan"
  split: "length"                 # length, template, addprim_jump
  train_subset: "train"
  eval_subset: "test"
  max_length: 128
  
  # Optimization
  batch_size: 16                  # Effective batch size
  gradient_accumulation: 2        # Accumulation steps
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 500
  max_epochs: 20
  
  # Precision
  fp16: true
  gradient_checkpointing: true
  
  # Pair generation
  pairs_per_example: 5            # Target positive pairs per example
  min_positives_per_batch: 4      # Minimum positive pairs in batch
  
  # Checkpointing
  save_every: 1                   # Save every N epochs
  eval_every: 1                   # Evaluate every N epochs
  
# Evaluation settings
evaluation:
  datasets:
    - name: "scan"
      split: "length"
    - name: "scan"
      split: "template"
    - name: "cogs"
      split: "gen"
  batch_size: 32
  max_samples: null               # null = use all samples

# Logging
logging:
  use_wandb: true
  project_name: "sci-compositional"
  log_every: 10                   # Log every N steps

# Output paths
output:
  checkpoint_dir: "outputs/checkpoints"
  log_dir: "outputs/logs"
  results_dir: "outputs/results"
```

### Step 2.2: Baseline Configuration (configs/baseline.yaml)

```yaml
# configs/baseline.yaml
# Baseline: Standard fine-tuning WITHOUT SCI components

model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype: "float16"

# ALL SCI components disabled
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
  gradient_accumulation: 2
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 500
  max_epochs: 20
  fp16: true
  gradient_checkpointing: true
  save_every: 1
  eval_every: 1

evaluation:
  datasets:
    - name: "scan"
      split: "length"
    - name: "scan"
      split: "template"
  batch_size: 32

logging:
  use_wandb: true
  project_name: "sci-compositional"
  log_every: 10

output:
  checkpoint_dir: "outputs/checkpoints/baseline"
  log_dir: "outputs/logs/baseline"
  results_dir: "outputs/results/baseline"
```

### Step 2.3: Ablation Configs

**AI AGENT: Create these configs for ablation studies. Each disables ONE component.**

```yaml
# configs/ablation_no_se.yaml
# Ablation: Without Structural Encoder
# Tests: Does SE contribute to compositional generalization?

model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype: "float16"

sci:
  structural_encoder:
    enabled: false                # DISABLED
  content_encoder:
    enabled: true
    refiner_layers: 2
    shared_embeddings: true
    orthogonality_weight: 0.01
  causal_binding:
    enabled: true
    injection_layers: [6, 11, 16]
    n_iterations: 3
  contrastive_learning:
    enabled: false                # Requires SE, so disabled

training:
  dataset: "scan"
  split: "length"
  batch_size: 16
  learning_rate: 5e-5
  max_epochs: 20
  fp16: true

output:
  checkpoint_dir: "outputs/checkpoints/ablation_no_se"
```

```yaml
# configs/ablation_no_scl.yaml
# Ablation: Without Structural Contrastive Loss
# Tests: Is SCL necessary or can structure emerge without it?

model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype: "float16"

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
  contrastive_learning:
    enabled: false                # DISABLED - key ablation
    scl_weight: 0.0

training:
  dataset: "scan"
  split: "length"
  batch_size: 16
  learning_rate: 5e-5
  max_epochs: 20
  fp16: true

output:
  checkpoint_dir: "outputs/checkpoints/ablation_no_scl"
```

---

## Phase 3: Model Implementation

### Step 3.1: Structural Encoder (src/models/structural_encoder.py)

**AI AGENT: This is the MOST CRITICAL component. The AbstractionLayer is the key innovation.**

```python
# src/models/structural_encoder.py
"""
Structural Encoder (SE) - Core SCI Component

PURPOSE: Extract structural patterns from input, invariant to content.

KEY INNOVATION: AbstractionLayer learns to identify structural vs. content features
WITHOUT explicit supervision.

IMPLEMENTATION REQUIREMENTS:
1. AbstractionLayer MUST be present - this is the novel component
2. Structure queries attend ONLY to instruction (not response)
3. GNN builds causal graph over slots
4. Output must be consistent across structurally-similar inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AbstractionLayer(nn.Module):
    """
    THE KEY NOVEL COMPONENT.
    
    Learns to identify and preserve ONLY structural information.
    Uses learned masking that suppresses content-specific features.
    
    How it works:
    1. Structural detector produces "structuralness" scores for each feature
    2. High scores = structural (keep), low scores = content (suppress)
    3. Trained end-to-end with SCL - features consistent across positive pairs get high scores
    
    DO NOT SIMPLIFY THIS - it's what makes SCI novel.
    """
    
    def __init__(self, d_model: int, hidden_mult: int = 2):
        super().__init__()
        
        # Structural feature detector
        # This learns which aspects of the representation are structural
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.Sigmoid()  # Outputs [0, 1] structuralness scores
        )
        
        # Residual gate (learnable)
        # Allows some content information to pass through initially
        self.residual_gate = nn.Parameter(torch.tensor(0.1))
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply structural abstraction.
        
        Args:
            x: [B, N, D] input features
            
        Returns:
            abstracted: [B, N, D] features with content suppressed
        """
        # Detect structural features
        structural_scores = self.structural_detector(x)  # [B, N, D]
        
        # Mask: keep structural, suppress content
        # But allow residual to help training initially
        abstracted = x * structural_scores + self.residual_gate * x * (1 - structural_scores)
        
        # Normalize
        abstracted = self.layer_norm(abstracted)
        
        return abstracted
    
    def get_structuralness_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw structuralness scores for analysis."""
        return self.structural_detector(x)


class StructuralGNN(nn.Module):
    """
    Graph Neural Network for structural reasoning.
    
    Builds causal relationships between structural slots.
    """
    
    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        
        self.layers = nn.ModuleList([
            StructuralGNNLayer(d_model) for _ in range(n_layers)
        ])
        
    def forward(
        self, 
        slot_values: torch.Tensor,  # [B, K, D]
        edge_weights: torch.Tensor  # [B, K, K]
    ) -> torch.Tensor:
        """
        Process slot values through GNN.
        """
        h = slot_values
        for layer in self.layers:
            h = layer(h, edge_weights)
        return h


class StructuralGNNLayer(nn.Module):
    """Single GNN layer."""
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.update_fn = nn.GRUCell(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        h: torch.Tensor,           # [B, K, D]
        edge_weights: torch.Tensor # [B, K, K]
    ) -> torch.Tensor:
        B, K, D = h.shape
        
        # Compute messages from all pairs
        # h_i concatenated with h_j for all i, j
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, D]
        h_j = h.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, D]
        
        messages = self.message_fn(torch.cat([h_i, h_j], dim=-1))  # [B, K, K, D]
        
        # Weight messages by edge weights
        weighted_messages = messages * edge_weights.unsqueeze(-1)  # [B, K, K, D]
        
        # Aggregate messages (sum over source nodes)
        aggregated = weighted_messages.sum(dim=2)  # [B, K, D]
        
        # Update node states
        h_flat = h.view(B * K, D)
        agg_flat = aggregated.view(B * K, D)
        updated = self.update_fn(agg_flat, h_flat).view(B, K, D)
        
        # Residual + LayerNorm
        return self.layer_norm(h + updated)


class EdgePredictor(nn.Module):
    """
    Predicts causal edges between structural slots.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.edge_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, slot_values: torch.Tensor) -> torch.Tensor:
        """
        Predict edge weights between slots.
        
        Args:
            slot_values: [B, K, D]
            
        Returns:
            edge_weights: [B, K, K] (normalized)
        """
        B, K, D = slot_values.shape
        
        # Compute pairwise features
        h_i = slot_values.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, D]
        h_j = slot_values.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, D]
        
        pairwise = torch.cat([h_i, h_j], dim=-1)  # [B, K, K, 2D]
        
        # Score edges
        scores = self.edge_scorer(pairwise).squeeze(-1)  # [B, K, K]
        
        # Normalize (softmax over source nodes)
        edge_weights = F.softmax(scores, dim=-1)
        
        return edge_weights


class StructuralEncoder(nn.Module):
    """
    Complete Structural Encoder.
    
    Architecture:
    1. AbstractionLayer: Filter out content, keep structure
    2. Cross-attention: Structure queries attend to abstracted input
    3. EdgePredictor: Predict causal graph over slots
    4. StructuralGNN: Reason over structural graph
    """
    
    def __init__(
        self,
        d_model: int,
        n_slots: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
        abstraction_hidden_mult: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_slots = n_slots
        
        # AbstractionLayer - THE KEY INNOVATION
        self.abstraction = AbstractionLayer(d_model, abstraction_hidden_mult)
        
        # Learnable structure queries (like prompts but for structure)
        self.structure_queries = nn.Parameter(
            torch.randn(n_slots, d_model) * 0.02
        )
        
        # Cross-attention from queries to abstracted input
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Edge predictor and GNN
        self.edge_predictor = EdgePredictor(d_model)
        self.structural_gnn = StructuralGNN(d_model, n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,  # [B, N, D]
        attention_mask: torch.Tensor  # [B, N] - 1 for valid tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract structural representation.
        
        Args:
            hidden_states: Input embeddings
            attention_mask: Mask for valid tokens (instruction only!)
            
        Returns:
            structural_graph: [B, K, D] - structural slot representations
            edge_weights: [B, K, K] - causal graph adjacency
        """
        B = hidden_states.shape[0]
        device = hidden_states.device
        
        # Step 1: Apply abstraction to filter content
        abstracted = self.abstraction(hidden_states)  # [B, N, D]
        
        # Step 2: Expand queries for batch
        queries = self.structure_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        
        # Create attention mask for cross-attention
        # Key padding mask: True = ignore, False = attend
        key_padding_mask = (attention_mask == 0)  # [B, N]
        
        # Step 3: Cross-attention - queries attend to abstracted input
        slot_values, _ = self.cross_attention(
            query=queries,          # [B, K, D]
            key=abstracted,         # [B, N, D]
            value=abstracted,       # [B, N, D]
            key_padding_mask=key_padding_mask
        )  # [B, K, D]
        
        # Step 4: Predict causal edges between slots
        edge_weights = self.edge_predictor(slot_values)  # [B, K, K]
        
        # Step 5: Reason over structural graph with GNN
        structural_graph = self.structural_gnn(slot_values, edge_weights)  # [B, K, D]
        
        # Step 6: Project output
        structural_graph = self.output_proj(structural_graph)
        
        return structural_graph, edge_weights
```

### Step 3.2: Content Encoder (src/models/content_encoder.py)

```python
# src/models/content_encoder.py
"""
Content Encoder (CE) - Role-Agnostic Content Representation

PURPOSE: Learn content embeddings that are independent of structural role.

KEY PROPERTY: Same entity has same embedding regardless of where it appears
in the structure. "walk" has same content embedding whether in "walk twice"
or "walk after run".

IMPLEMENTATION REQUIREMENTS:
1. Content embeddings must be orthogonal to structural embeddings
2. Refiner removes any structural contamination
3. Can share base embeddings with transformer for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContentRefiner(nn.Module):
    """
    Refines content embeddings to remove structural contamination.
    
    Ensures content representations capture ONLY entity identity,
    not their role in the structure.
    """
    
    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine content embeddings."""
        for layer in self.layers:
            x = x + layer(x)  # Residual connections
        return x


class OrthogonalProjector(nn.Module):
    """
    Projects content embeddings to be orthogonal to structural embeddings.
    
    This ENFORCES explicit separation of structure and content.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable projection for soft orthogonality
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        content: torch.Tensor,      # [B, N, D]
        structure: torch.Tensor     # [B, K, D]
    ) -> torch.Tensor:
        """
        Project content to be orthogonal to structural subspace.
        
        Uses Gram-Schmidt-like projection.
        """
        B, N, D = content.shape
        K = structure.shape[1]
        
        # Apply learnable transformation first
        content = self.proj(content)
        
        # Project out structural components
        # For each structural slot, remove its component from content
        for k in range(K):
            s = structure[:, k:k+1, :]  # [B, 1, D]
            s_norm = F.normalize(s, dim=-1)  # Normalize
            
            # Project content onto structural direction
            proj_coeff = (content * s_norm).sum(dim=-1, keepdim=True)  # [B, N, 1]
            proj = proj_coeff * s_norm  # [B, N, D]
            
            # Subtract projection
            content = content - proj
            
        return content
    
    def compute_orthogonality_loss(
        self,
        content: torch.Tensor,   # [B, N, D]
        structure: torch.Tensor  # [B, K, D]
    ) -> torch.Tensor:
        """
        Compute loss that penalizes non-orthogonality.
        
        Returns cosine similarity (should be minimized).
        """
        # Average pool content and structure
        content_avg = content.mean(dim=1)     # [B, D]
        structure_avg = structure.mean(dim=1) # [B, D]
        
        # Normalize
        content_norm = F.normalize(content_avg, dim=-1)
        structure_norm = F.normalize(structure_avg, dim=-1)
        
        # Cosine similarity (want this to be 0)
        similarity = (content_norm * structure_norm).sum(dim=-1)  # [B]
        
        # Loss is absolute cosine similarity
        loss = similarity.abs().mean()
        
        return loss


class ContentEncoder(nn.Module):
    """
    Complete Content Encoder.
    
    Architecture:
    1. Base embeddings (optionally shared with transformer)
    2. ContentRefiner: Remove structural contamination
    3. OrthogonalProjector: Ensure orthogonality to structure
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_refiner_layers: int = 2,
        share_embeddings: bool = True,
        base_embeddings: Optional[nn.Embedding] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.share_embeddings = share_embeddings
        
        # Embeddings
        if share_embeddings and base_embeddings is not None:
            self.embedding = base_embeddings
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)
            
        # Content refiner
        self.refiner = ContentRefiner(d_model, n_refiner_layers)
        
        # Orthogonal projector
        self.orthogonal_proj = OrthogonalProjector(d_model)
        
    def forward(
        self, 
        input_ids: torch.Tensor,                    # [B, N]
        structural_features: Optional[torch.Tensor] = None  # [B, K, D]
    ) -> torch.Tensor:
        """
        Encode content.
        
        Args:
            input_ids: Token IDs
            structural_features: Structural representation for orthogonalization
            
        Returns:
            content: [B, N, D] role-agnostic content embeddings
        """
        # Get base embeddings
        content = self.embedding(input_ids)  # [B, N, D]
        
        # Refine to remove structural contamination
        content = self.refiner(content)
        
        # Project to be orthogonal to structure
        if structural_features is not None:
            content = self.orthogonal_proj(content, structural_features)
            
        return content
    
    def compute_orthogonality_loss(
        self,
        content: torch.Tensor,
        structure: torch.Tensor
    ) -> torch.Tensor:
        """Compute orthogonality loss for training."""
        return self.orthogonal_proj.compute_orthogonality_loss(content, structure)
```

### Step 3.3: Causal Binding Mechanism (src/models/causal_binding.py)

```python
# src/models/causal_binding.py
"""
Causal Binding Mechanism (CBM) - How Content Fills Structure

PURPOSE: Implement HOW content fills structural slots using causal intervention.

KEY INNOVATION: Uses learned causal intervention (do-operator) to bind
content to structure, enabling counterfactual composition.

IMPLEMENTATION REQUIREMENTS:
1. Binding attention determines which content fills which slot
2. CausalInterventionLayer propagates through structural graph
3. Broadcast back to token positions for transformer injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalInterventionLayer(nn.Module):
    """
    Implements do(X) operator for slot values.
    
    This enables COUNTERFACTUAL COMPOSITION:
    Given a structure and NEW content, generate outputs
    for combinations never seen during training.
    """
    
    def __init__(self, d_model: int, n_iterations: int = 3):
        super().__init__()
        
        self.n_iterations = n_iterations
        
        # Message function for causal propagation
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Update function (GRU cell)
        self.update_fn = nn.GRUCell(d_model, d_model)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        slot_values: torch.Tensor,   # [B, K, D]
        edge_weights: torch.Tensor   # [B, K, K]
    ) -> torch.Tensor:
        """
        Propagate through causal structure.
        """
        B, K, D = slot_values.shape
        
        h = slot_values
        
        for _ in range(self.n_iterations):
            # Compute messages along edges
            messages = torch.zeros_like(h)
            
            # h_i = source, h_j = target
            h_expanded_i = h.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, D]
            h_expanded_j = h.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, D]
            
            # Pairwise messages
            pairwise = torch.cat([h_expanded_i, h_expanded_j], dim=-1)  # [B, K, K, 2D]
            all_messages = self.message_fn(pairwise)  # [B, K, K, D]
            
            # Weight by edges and aggregate
            weighted = all_messages * edge_weights.unsqueeze(-1)  # [B, K, K, D]
            messages = weighted.sum(dim=1)  # [B, K, D] - sum over sources
            
            # Update slot values
            h_flat = h.view(B * K, D)
            msg_flat = messages.view(B * K, D)
            h = self.update_fn(msg_flat, h_flat).view(B, K, D)
            
        # Final normalization
        h = self.layer_norm(h)
        
        return h


class CausalBindingMechanism(nn.Module):
    """
    Complete Causal Binding Mechanism.
    
    Combines structure and content to condition transformer generation.
    
    Architecture:
    1. Binding attention: Determine which content goes to which slot
    2. Causal intervention: Propagate through structural graph
    3. Broadcast: Map slot values back to token positions
    4. Combine: Merge with transformer hidden states
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_slots: int,
        n_iterations: int = 3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_slots = n_slots
        
        # Binding attention - which content fills which slot
        self.binding_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Causal intervention layer
        self.intervention = CausalInterventionLayer(d_model, n_iterations)
        
        # Reverse attention - broadcast slots back to tokens
        self.broadcast_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Combination gate - how much to incorporate binding
        self.combination_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(
        self,
        hidden_states: torch.Tensor,     # [B, N, D] - transformer hidden states
        structural_graph: torch.Tensor,  # [B, K, D] - from SE
        edge_weights: torch.Tensor,      # [B, K, K] - causal graph
        content: torch.Tensor,           # [B, N, D] - from CE
        attention_mask: Optional[torch.Tensor] = None  # [B, N]
    ) -> torch.Tensor:
        """
        Apply causal binding to transformer hidden states.
        
        Returns modified hidden states conditioned on structure+content.
        """
        B, N, D = hidden_states.shape
        K = self.n_slots
        
        # Step 1: Bind content to structural slots
        # Structural slots query content to get filled values
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
            
        bound_slots, _ = self.binding_attention(
            query=structural_graph,  # [B, K, D]
            key=content,             # [B, N, D]
            value=content,           # [B, N, D]
            key_padding_mask=key_padding_mask
        )  # [B, K, D]
        
        # Step 2: Apply causal intervention
        intervened_slots = self.intervention(bound_slots, edge_weights)  # [B, K, D]
        
        # Step 3: Broadcast slot values back to token positions
        # Tokens query slots to get their contributions
        token_contributions, _ = self.broadcast_attention(
            query=hidden_states,     # [B, N, D]
            key=intervened_slots,    # [B, K, D]
            value=intervened_slots   # [B, K, D]
        )  # [B, N, D]
        
        # Step 4: Combine with original hidden states
        combined_features = torch.cat([hidden_states, token_contributions], dim=-1)
        
        # Gated combination
        gate = self.combination_gate(combined_features)
        output = hidden_states * (1 - gate) + token_contributions * gate
        
        # Alternative: direct combination with projection
        # output = self.output_proj(combined_features)
        
        return output
```

---

**CONTINUE TO PART 2 FOR: Main Model, Losses, Data Pipeline, Training, Evaluation**
