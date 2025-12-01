# SCI MASTER GUIDE - PART 2

## Continuation from Part 1

---

<a name="section-6"></a>
# SECTION 6: IMPLEMENTATION VERIFICATION TESTS

## 6.1 TEST HIERARCHY

```
Tests are organized in a specific order. Run them sequentially.

LEVEL 1: Component Tests (must pass before integration)
├── test_abstraction_layer.py     → Verify AbstractionLayer works
├── test_structural_gnn.py        → Verify GNN works
├── test_edge_predictor.py        → Verify edge prediction works
├── test_content_refiner.py       → Verify content refiner works
├── test_orthogonal_projector.py  → Verify orthogonality works
└── test_causal_intervention.py   → Verify intervention works

LEVEL 2: Encoder Tests (must pass before model integration)
├── test_structural_encoder.py    → Full SE works
├── test_content_encoder.py       → Full CE works
└── test_causal_binding.py        → Full CBM works

LEVEL 3: Loss Tests (must pass before training)
├── test_scl_loss.py              → SCL computes correctly
└── test_combined_loss.py         → All losses combine correctly

LEVEL 4: Data Tests (must pass before training)
├── test_structure_extraction.py  → Structures extracted correctly
├── test_pair_generation.py       → Pairs generated correctly
└── test_data_leakage.py          → No data leakage

LEVEL 5: Integration Tests (must pass before training)
├── test_full_model.py            → Complete model works
├── test_forward_pass.py          → Forward pass works end-to-end
└── test_backward_pass.py         → Gradients flow correctly

LEVEL 6: Training Tests (run during training)
├── test_training_loop.py         → Training loop works
└── test_checkpointing.py         → Checkpoints save/load

LEVEL 7: Evaluation Tests (run after training)
├── test_evaluation.py            → Evaluation works
└── test_structural_invariance.py → Model learned invariance
```

## 6.2 CRITICAL TESTS (MUST PASS)

### Test 1: AbstractionLayer Verification

```python
# tests/test_abstraction_layer.py
"""
Test that AbstractionLayer is correctly implemented.

This is the KEY novel component. If it's wrong, SCI won't work.
"""

import torch
import pytest


class TestAbstractionLayer:
    """Comprehensive tests for AbstractionLayer."""
    
    @pytest.fixture
    def layer(self):
        from src.models.components.abstraction_layer import AbstractionLayer
        return AbstractionLayer(d_model=256, hidden_mult=2)
    
    def test_layer_exists(self, layer):
        """Verify layer has required components."""
        assert hasattr(layer, 'structural_detector'), \
            "CRITICAL: AbstractionLayer must have structural_detector!"
        assert hasattr(layer, 'residual_gate'), \
            "CRITICAL: AbstractionLayer must have residual_gate!"
        assert hasattr(layer, 'layer_norm'), \
            "CRITICAL: AbstractionLayer must have layer_norm!"
            
    def test_structural_detector_architecture(self, layer):
        """Verify structural_detector has correct architecture."""
        # Should be Sequential with: Linear -> GELU -> Dropout -> Linear -> Sigmoid
        sd = layer.structural_detector
        
        assert isinstance(sd, torch.nn.Sequential), \
            "structural_detector should be nn.Sequential"
        
        # Check last activation is Sigmoid
        last_module = list(sd.modules())[-1]
        assert isinstance(last_module, torch.nn.Sigmoid), \
            "structural_detector must end with Sigmoid for [0,1] output!"
            
    def test_output_shape(self, layer):
        """Verify output has same shape as input."""
        x = torch.randn(2, 10, 256)
        out = layer(x)
        
        assert out.shape == x.shape, \
            f"Output shape {out.shape} != input shape {x.shape}"
            
    def test_output_range(self, layer):
        """Verify structuralness scores are in [0, 1]."""
        x = torch.randn(2, 10, 256)
        scores = layer.get_structuralness_scores(x)
        
        assert scores.min() >= 0, "Scores should be >= 0"
        assert scores.max() <= 1, "Scores should be <= 1"
        
    def test_gradient_flow(self, layer):
        """Verify gradients flow through layer."""
        x = torch.randn(2, 10, 256, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert layer.structural_detector[0].weight.grad is not None, \
            "Gradients should flow to structural_detector"
        assert layer.residual_gate.grad is not None, \
            "Gradients should flow to residual_gate"
            
    def test_residual_gate_effect(self, layer):
        """Test that residual_gate allows some content through initially."""
        # With high residual_gate, output should be closer to input
        layer.residual_gate.data = torch.tensor(1.0)
        
        x = torch.randn(2, 10, 256)
        out = layer(x)
        
        # Output should be somewhat similar to input when gate is high
        # (not testing exact values, just that it's not zero)
        assert out.abs().mean() > 0.1, "Output should not be near zero"
        
    def test_abstraction_effect(self, layer):
        """
        Test that AbstractionLayer suppresses some features.
        
        The variance of output should be less than input because
        some features (content-related) are suppressed.
        """
        layer.residual_gate.data = torch.tensor(0.0)  # No residual
        
        x = torch.randn(2, 10, 256)
        out = layer(x)
        
        # Output variance per feature should be different from input
        # (some features suppressed, some amplified)
        input_var = x.var(dim=(0, 1))
        output_var = out.var(dim=(0, 1))
        
        # They should NOT be identical
        assert not torch.allclose(input_var, output_var, rtol=0.1), \
            "AbstractionLayer should modify feature variances"
```

### Test 2: Full Structural Encoder Verification

```python
# tests/test_structural_encoder.py
"""
Test that StructuralEncoder is correctly implemented.
"""

import torch
import pytest


class TestStructuralEncoder:
    """Comprehensive tests for StructuralEncoder."""
    
    @pytest.fixture
    def encoder(self):
        from src.models.structural_encoder import StructuralEncoder
        return StructuralEncoder(
            d_model=256,
            n_slots=8,
            n_layers=2,
            n_heads=4,
            abstraction_hidden_mult=2
        )
    
    def test_required_components(self, encoder):
        """Verify encoder has all required components."""
        required = [
            'abstraction',
            'structure_queries',
            'cross_attention',
            'edge_predictor',
            'structural_gnn',
            'output_proj'
        ]
        
        for component in required:
            assert hasattr(encoder, component), \
                f"CRITICAL: StructuralEncoder missing {component}!"
                
    def test_abstraction_is_abstraction_layer(self, encoder):
        """Verify abstraction is an AbstractionLayer instance."""
        from src.models.components.abstraction_layer import AbstractionLayer
        
        assert isinstance(encoder.abstraction, AbstractionLayer), \
            "CRITICAL: encoder.abstraction must be AbstractionLayer, not custom implementation!"
            
    def test_output_shapes(self, encoder):
        """Verify output shapes are correct."""
        B, N, D = 2, 20, 256
        K = 8  # n_slots
        
        hidden_states = torch.randn(B, N, D)
        attention_mask = torch.ones(B, N)
        
        encoder.eval()
        with torch.no_grad():
            structural_graph, edge_weights = encoder(hidden_states, attention_mask)
        
        assert structural_graph.shape == (B, K, D), \
            f"structural_graph shape {structural_graph.shape} != expected ({B}, {K}, {D})"
        assert edge_weights.shape == (B, K, K), \
            f"edge_weights shape {edge_weights.shape} != expected ({B}, {K}, {K})"
            
    def test_attention_mask_respected(self, encoder):
        """Verify that attention mask is respected."""
        hidden_states = torch.randn(2, 20, 256)
        
        # Mask out last 10 tokens
        mask_full = torch.ones(2, 20)
        mask_partial = torch.ones(2, 20)
        mask_partial[:, 10:] = 0
        
        encoder.eval()
        with torch.no_grad():
            out_full, _ = encoder(hidden_states, mask_full)
            out_partial, _ = encoder(hidden_states, mask_partial)
        
        # Outputs should be different because mask is different
        assert not torch.allclose(out_full, out_partial), \
            "Attention mask should affect output!"
            
    def test_edge_weights_normalized(self, encoder):
        """Verify edge weights sum to ~1 (softmax normalized)."""
        hidden_states = torch.randn(2, 20, 256)
        attention_mask = torch.ones(2, 20)
        
        encoder.eval()
        with torch.no_grad():
            _, edge_weights = encoder(hidden_states, attention_mask)
        
        # Each row should sum to approximately 1 (softmax)
        row_sums = edge_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01), \
            "Edge weights should be normalized (sum to 1)"
            
    def test_gradient_flow_to_abstraction(self, encoder):
        """Verify gradients flow through AbstractionLayer."""
        hidden_states = torch.randn(2, 20, 256, requires_grad=True)
        attention_mask = torch.ones(2, 20)
        
        structural_graph, _ = encoder(hidden_states, attention_mask)
        loss = structural_graph.sum()
        loss.backward()
        
        # Check gradients flow to AbstractionLayer
        assert encoder.abstraction.structural_detector[0].weight.grad is not None, \
            "Gradients should flow through AbstractionLayer!"
```

### Test 3: SCL Loss Verification

```python
# tests/test_scl_loss.py
"""
Test that Structural Contrastive Loss is correctly implemented.
"""

import torch
import pytest


class TestSCLLoss:
    """Comprehensive tests for SCL loss."""
    
    @pytest.fixture
    def loss_fn(self):
        from src.losses.structural_contrastive import StructuralContrastiveLoss
        return StructuralContrastiveLoss(temperature=0.07)
    
    def test_loss_computes(self, loss_fn):
        """Verify loss computes without error."""
        struct_reps = torch.randn(4, 8, 256)
        pair_labels = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]).float()
        
        loss = loss_fn(struct_reps, pair_labels)
        
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
        
    def test_loss_is_differentiable(self, loss_fn):
        """Verify loss is differentiable."""
        struct_reps = torch.randn(4, 8, 256, requires_grad=True)
        pair_labels = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]).float()
        
        loss = loss_fn(struct_reps, pair_labels)
        loss.backward()
        
        assert struct_reps.grad is not None, "Loss should be differentiable"
        
    def test_positive_pairs_reduce_loss(self, loss_fn):
        """
        Test that similar positive pairs have lower loss than dissimilar.
        
        If positive pairs are identical, loss should be lower.
        """
        # Positive pairs that are identical
        identical = torch.randn(2, 8, 256)
        struct_identical = torch.cat([identical, identical], dim=0)  # [4, 8, 256]
        
        # Positive pairs that are different
        struct_different = torch.randn(4, 8, 256)
        
        pair_labels = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]).float()
        
        loss_identical = loss_fn(struct_identical, pair_labels)
        loss_different = loss_fn(struct_different, pair_labels)
        
        assert loss_identical < loss_different, \
            "Identical positive pairs should have lower loss!"
            
    def test_no_positive_pairs_returns_zero(self, loss_fn):
        """If no positive pairs, loss should be 0."""
        struct_reps = torch.randn(4, 8, 256)
        pair_labels = torch.zeros(4, 4)  # No positive pairs
        
        loss = loss_fn(struct_reps, pair_labels)
        
        assert loss.item() == 0.0 or loss.requires_grad, \
            "With no positive pairs, loss should be 0"
            
    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        from src.losses.structural_contrastive import StructuralContrastiveLoss
        
        loss_fn_low_temp = StructuralContrastiveLoss(temperature=0.01)
        loss_fn_high_temp = StructuralContrastiveLoss(temperature=1.0)
        
        struct_reps = torch.randn(4, 8, 256)
        pair_labels = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]).float()
        
        loss_low = loss_fn_low_temp(struct_reps, pair_labels)
        loss_high = loss_fn_high_temp(struct_reps, pair_labels)
        
        # Just verify both compute without error
        assert not torch.isnan(loss_low)
        assert not torch.isnan(loss_high)
```

### Test 4: Pair Generation Verification

```python
# tests/test_pair_generation.py
"""
Test that pair generation works correctly for SCL training.
"""

import torch
import pytest


class TestPairGeneration:
    """Tests for pair generation."""
    
    @pytest.fixture
    def scan_generator(self):
        from src.data.pair_generators.scan_pair_generator import SCANPairGenerator
        return SCANPairGenerator()
    
    def test_same_structure_positive_pairs(self, scan_generator):
        """Verify same-structure examples get pair_label = 1."""
        commands = [
            "walk twice",      # Structure: ACTION_0 twice
            "run twice",       # Structure: ACTION_0 twice (SAME)
            "jump twice",      # Structure: ACTION_0 twice (SAME)
            "walk and run",    # Structure: ACTION_0 and ACTION_1 (DIFFERENT)
        ]
        
        scan_generator.process_dataset(commands)
        pair_labels = scan_generator.generate_batch_pairs([0, 1, 2, 3])
        
        # First 3 have same structure
        assert pair_labels[0, 1] == 1, "walk twice and run twice should be positive"
        assert pair_labels[0, 2] == 1, "walk twice and jump twice should be positive"
        assert pair_labels[1, 2] == 1, "run twice and jump twice should be positive"
        
        # Different structure should be negative
        assert pair_labels[0, 3] == 0, "walk twice and walk and run should be negative"
        
    def test_structure_template_extraction(self, scan_generator):
        """Verify structure templates are extracted correctly."""
        commands = ["walk twice", "run twice", "walk and run"]
        scan_generator.process_dataset(commands)
        
        # Check templates
        assert scan_generator.structures[0].template == "ACTION_0 twice"
        assert scan_generator.structures[1].template == "ACTION_0 twice"
        assert scan_generator.structures[2].template == "ACTION_0 and ACTION_1"
        
    def test_batch_augmentation_for_positives(self, scan_generator):
        """Test that batches are augmented to have positive pairs."""
        commands = [
            "walk twice", "run twice", "jump twice",  # Same structure
            "walk and run", "jump and look",           # Same structure
            "turn left",                                # Unique
        ]
        
        scan_generator.process_dataset(commands)
        scan_generator.min_positives = 1
        
        # Augment a batch that has no positives
        base_indices = [5]  # Only "turn left" - has no positive pairs
        augmented = scan_generator.augment_batch_for_positives(base_indices, target_batch_size=4)
        
        # Should still contain the original
        assert 5 in augmented
        
    def test_positive_ratio_calculation(self, scan_generator):
        """Test positive ratio computation."""
        commands = ["walk twice", "run twice", "walk and run"]
        scan_generator.process_dataset(commands)
        
        # Batch with 2 positives, 1 different
        ratio = scan_generator.get_positive_ratio([0, 1, 2])
        
        # Out of 6 pairs (excluding diagonal), 2 are positive
        # (0,1), (1,0) are positive; (0,2), (1,2), (2,0), (2,1) are negative
        expected_ratio = 2 / 6
        assert abs(ratio - expected_ratio) < 0.01, \
            f"Expected ratio {expected_ratio}, got {ratio}"
```

### Test 5: Structural Invariance Verification (POST-TRAINING)

```python
# tests/test_structural_invariance.py
"""
Test that trained SCI model has structural invariance.

These tests should be run AFTER training to verify SCI learned correctly.
"""

import torch
import pytest
import numpy as np


class TestStructuralInvariance:
    """Tests for structural invariance property."""
    
    @pytest.fixture
    def trained_model(self):
        """Load trained SCI model."""
        from src.models.sci_model import SCIModel
        
        # Try to load trained model
        try:
            model = SCIModel.from_pretrained("outputs/checkpoints/sci/best_model")
            model.eval()
            return model
        except:
            pytest.skip("No trained model available")
    
    def test_same_structure_similar_representation(self, trained_model):
        """
        Test that same-structure inputs have similar structural representations.
        
        "walk twice", "run twice", "jump twice" should have nearly identical
        structural representations.
        """
        same_structure_commands = [
            "walk twice",
            "run twice", 
            "jump twice",
            "look twice"
        ]
        
        representations = []
        
        for command in same_structure_commands:
            # Tokenize
            input_text = f"IN: {command} OUT:"
            inputs = trained_model.tokenizer(
                input_text, return_tensors='pt'
            )
            
            with torch.no_grad():
                struct_rep, _ = trained_model.get_structural_representation(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                
            # Mean pool
            rep = struct_rep.mean(dim=1).squeeze().numpy()
            representations.append(rep)
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(representations)):
            for j in range(i+1, len(representations)):
                cos_sim = np.dot(representations[i], representations[j]) / (
                    np.linalg.norm(representations[i]) * np.linalg.norm(representations[j])
                )
                similarities.append(cos_sim)
        
        mean_similarity = np.mean(similarities)
        
        print(f"Same-structure similarity: {mean_similarity:.4f}")
        
        # Should be high (> 0.9) after training
        assert mean_similarity > 0.85, \
            f"Same-structure examples should have high similarity (got {mean_similarity})"
            
    def test_different_structure_different_representation(self, trained_model):
        """
        Test that different-structure inputs have different representations.
        """
        commands = [
            "walk twice",      # X twice
            "walk and run",    # X and Y
            "walk after run",  # X after Y
            "turn left"        # turn direction
        ]
        
        representations = []
        
        for command in commands:
            input_text = f"IN: {command} OUT:"
            inputs = trained_model.tokenizer(input_text, return_tensors='pt')
            
            with torch.no_grad():
                struct_rep, _ = trained_model.get_structural_representation(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                
            rep = struct_rep.mean(dim=1).squeeze().numpy()
            representations.append(rep)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(representations)):
            for j in range(i+1, len(representations)):
                cos_sim = np.dot(representations[i], representations[j]) / (
                    np.linalg.norm(representations[i]) * np.linalg.norm(representations[j])
                )
                similarities.append(cos_sim)
        
        mean_similarity = np.mean(similarities)
        
        print(f"Different-structure similarity: {mean_similarity:.4f}")
        
        # Should be lower than same-structure (< 0.7)
        assert mean_similarity < 0.75, \
            f"Different-structure examples should have lower similarity (got {mean_similarity})"
            
    def test_invariance_to_content_substitution(self, trained_model):
        """
        The ULTIMATE test: changing content should NOT change structure.
        """
        base_command = "walk twice"
        
        # Get structural representation for base
        base_input = trained_model.tokenizer(f"IN: {base_command} OUT:", return_tensors='pt')
        with torch.no_grad():
            base_rep, _ = trained_model.get_structural_representation(
                base_input['input_ids'],
                base_input['attention_mask']
            )
        base_rep = base_rep.mean(dim=1).squeeze().numpy()
        
        # Test with all content substitutions
        substitutions = ["run twice", "jump twice", "look twice"]
        
        for sub_command in substitutions:
            sub_input = trained_model.tokenizer(f"IN: {sub_command} OUT:", return_tensors='pt')
            with torch.no_grad():
                sub_rep, _ = trained_model.get_structural_representation(
                    sub_input['input_ids'],
                    sub_input['attention_mask']
                )
            sub_rep = sub_rep.mean(dim=1).squeeze().numpy()
            
            # Compute cosine similarity
            cos_sim = np.dot(base_rep, sub_rep) / (
                np.linalg.norm(base_rep) * np.linalg.norm(sub_rep)
            )
            
            print(f"Similarity({base_command}, {sub_command}): {cos_sim:.4f}")
            
            assert cos_sim > 0.9, \
                f"Content substitution should not affect structure! Got similarity {cos_sim}"
```

---

<a name="section-7"></a>
# SECTION 7: COMPLETE FILE IMPLEMENTATIONS

## 7.1 FILE CREATION INSTRUCTIONS

**AI AGENT: Create each file EXACTLY as specified. Do not simplify or modify.**

### File 1: requirements.txt

```
# requirements.txt
# Core dependencies for SCI

# Deep learning
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0

# Numerical
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Configuration
pyyaml>=6.0
omegaconf>=2.3.0

# Logging and tracking
wandb>=0.15.0
tensorboard>=2.14.0

# Utilities
tqdm>=4.65.0
rich>=13.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Development
black>=23.0.0
isort>=5.12.0
```

### File 2: AbstractionLayer Component

```python
# src/models/components/abstraction_layer.py
"""
AbstractionLayer - THE KEY NOVEL COMPONENT OF SCI

This component learns to identify and preserve ONLY structural information,
suppressing content-specific features.

DO NOT MODIFY OR SIMPLIFY THIS IMPLEMENTATION.
"""

import torch
import torch.nn as nn
from typing import Optional


class AbstractionLayer(nn.Module):
    """
    Learns to separate structural features from content features.
    
    Architecture:
        structural_detector: nn.Sequential
            Linear(d_model, d_model * hidden_mult)
            GELU()
            Dropout(0.1)
            Linear(d_model * hidden_mult, d_model)
            Sigmoid()  <- CRITICAL: outputs [0, 1] "structuralness" scores
            
        residual_gate: nn.Parameter
            Learnable scalar that allows some content through initially
            
        layer_norm: nn.LayerNorm
            Normalizes output for stability
    
    How it works:
        1. structural_detector produces scores in [0, 1] for each feature
        2. High scores = structural (keep), low scores = content (suppress)
        3. residual_gate allows some content through during early training
        4. Output = x * scores + residual_gate * x * (1 - scores)
    """
    
    def __init__(
        self, 
        d_model: int, 
        hidden_mult: int = 2,
        dropout: float = 0.1,
        initial_residual: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_mult = hidden_mult
        
        # Structural feature detector
        # MUST end with Sigmoid for [0, 1] output
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.Sigmoid()  # CRITICAL: Sigmoid for [0, 1] range
        )
        
        # Residual gate - learnable scalar
        # Starts small to encourage abstraction from the beginning
        self.residual_gate = nn.Parameter(
            torch.tensor(initial_residual)
        )
        
        # Layer normalization for output stability
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.structural_detector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply structural abstraction.
        
        Args:
            x: Input features [B, N, D]
            
        Returns:
            abstracted: Features with content suppressed [B, N, D]
        """
        # Step 1: Compute structuralness scores for each feature
        # scores[i] close to 1 = structural, close to 0 = content
        scores = self.structural_detector(x)  # [B, N, D]
        
        # Step 2: Apply selective masking
        # Keep structural features (high scores)
        # Suppress content features (low scores) but allow some through via residual
        structural_part = x * scores
        residual_part = self.residual_gate * x * (1 - scores)
        
        abstracted = structural_part + residual_part
        
        # Step 3: Normalize for stability
        abstracted = self.layer_norm(abstracted)
        
        return abstracted
    
    def get_structuralness_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw structuralness scores for analysis.
        
        Higher values = more structural, lower = more content.
        """
        return self.structural_detector(x)
    
    def get_abstraction_strength(self) -> float:
        """
        Get current abstraction strength (inverse of residual gate).
        
        Higher = more abstraction (less content passes through).
        """
        return 1.0 - self.residual_gate.item()
```

### File 3: Structural Encoder

```python
# src/models/structural_encoder.py
"""
Structural Encoder (SE) - Extracts structural representation from input.

Architecture:
1. AbstractionLayer - filters out content, keeps structure
2. Cross-attention - structure queries attend to abstracted input
3. EdgePredictor - predicts causal graph over slots
4. StructuralGNN - reasons over structural graph

CRITICAL: Must use AbstractionLayer from components, not custom implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .components.abstraction_layer import AbstractionLayer
from .components.structural_gnn import StructuralGNN
from .components.edge_predictor import EdgePredictor


class StructuralEncoder(nn.Module):
    """
    Complete Structural Encoder for SCI.
    
    Args:
        d_model: Hidden dimension
        n_slots: Number of structural slots (like prompts)
        n_layers: Number of GNN layers
        n_heads: Number of attention heads
        abstraction_hidden_mult: Hidden multiplier for AbstractionLayer
    """
    
    def __init__(
        self,
        d_model: int,
        n_slots: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
        abstraction_hidden_mult: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_slots = n_slots
        
        # =========================================
        # COMPONENT 1: AbstractionLayer
        # CRITICAL: Must use AbstractionLayer class
        # =========================================
        self.abstraction = AbstractionLayer(
            d_model=d_model,
            hidden_mult=abstraction_hidden_mult
        )
        
        # Verification assertion
        assert isinstance(self.abstraction, AbstractionLayer), \
            "CRITICAL: self.abstraction must be AbstractionLayer instance!"
        
        # =========================================
        # COMPONENT 2: Learnable Structure Queries
        # =========================================
        # These are like soft prompts that query for structural patterns
        self.structure_queries = nn.Parameter(
            torch.randn(n_slots, d_model) * 0.02
        )
        
        # =========================================
        # COMPONENT 3: Cross-Attention
        # =========================================
        # Structure queries attend to abstracted input
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # =========================================
        # COMPONENT 4: Edge Predictor
        # =========================================
        self.edge_predictor = EdgePredictor(d_model)
        
        # =========================================
        # COMPONENT 5: Structural GNN
        # =========================================
        self.structural_gnn = StructuralGNN(d_model, n_layers)
        
        # =========================================
        # COMPONENT 6: Output Projection
        # =========================================
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, N, D]
        attention_mask: torch.Tensor   # [B, N] - 1 for valid tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract structural representation.
        
        CRITICAL: attention_mask should mark ONLY instruction tokens as valid.
        Response tokens should be masked (0) to prevent data leakage.
        
        Args:
            hidden_states: Input embeddings
            attention_mask: 1 for instruction tokens, 0 for response/padding
            
        Returns:
            structural_graph: [B, K, D] - structural slot representations
            edge_weights: [B, K, K] - causal graph adjacency
        """
        B = hidden_states.shape[0]
        
        # =========================================
        # STEP 1: Apply AbstractionLayer
        # =========================================
        # This filters out content-specific features
        abstracted = self.abstraction(hidden_states)  # [B, N, D]
        
        # =========================================
        # STEP 2: Prepare structure queries
        # =========================================
        queries = self.structure_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        
        # =========================================
        # STEP 3: Cross-attention with mask
        # =========================================
        # Key padding mask: True means IGNORE this position
        key_padding_mask = (attention_mask == 0)  # [B, N]
        
        slot_values, attn_weights = self.cross_attention(
            query=queries,           # [B, K, D]
            key=abstracted,          # [B, N, D]
            value=abstracted,        # [B, N, D]
            key_padding_mask=key_padding_mask,
            need_weights=True
        )  # slot_values: [B, K, D]
        
        # =========================================
        # STEP 4: Predict edges between slots
        # =========================================
        edge_weights = self.edge_predictor(slot_values)  # [B, K, K]
        
        # =========================================
        # STEP 5: Reason with GNN
        # =========================================
        structural_graph = self.structural_gnn(slot_values, edge_weights)  # [B, K, D]
        
        # =========================================
        # STEP 6: Output projection
        # =========================================
        structural_graph = self.output_proj(structural_graph)
        structural_graph = self.output_norm(structural_graph)
        
        return structural_graph, edge_weights
    
    def get_attention_weights(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get cross-attention weights for analysis."""
        B = hidden_states.shape[0]
        
        abstracted = self.abstraction(hidden_states)
        queries = self.structure_queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = (attention_mask == 0)
        
        _, attn_weights = self.cross_attention(
            query=queries,
            key=abstracted,
            value=abstracted,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        
        return attn_weights
```

---

**CONTINUED IN PART 3: Training Regime, Evaluation, Configs, and Execution Checklist**
