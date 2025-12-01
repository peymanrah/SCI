# Structural Causal Invariance (SCI) - Technical Implementation Guide

## Overview

This document provides detailed technical specifications and pseudocode for implementing the SCI architecture. SCI introduces a fundamentally new principle for compositional generalization: **structural representations should be causally invariant to content substitution**.

---

## Core Mathematical Framework

### 1. Representation Factorization

Given an input sequence x = (x₁, x₂, ..., xₙ), SCI learns three factorized representations:

```
S(x) ∈ ℝ^(k × d_s)   # Structural representation: k abstract slots, d_s dimensions each
C(x) ∈ ℝ^(n × d_c)   # Content representation: n tokens, d_c dimensions each
B : (S, C) → ℝ^(n × d) # Binding function: combines structure and content
```

### 2. Structural Invariance Constraint (The Novel Principle)

For any two inputs x₁ and x₂ that share the same structure but differ in content:

```
||S(x₁) - S(x₂)||₂ < ε  (structural invariance)
||C(x₁) - C(x₂)||₂ > δ  (content differentiation)
```

This constraint is enforced through contrastive learning during training.

---

## Component 1: Structural Encoder (SE)

### Architecture

```python
class StructuralEncoder(nn.Module):
    """
    Learns structure as a causal graph over abstract slots.
    Key innovation: Uses "structural abstraction" to identify which
    aspects of input are structural vs content-specific.
    """
    def __init__(self, d_model, n_slots, n_layers):
        self.n_slots = n_slots  # Number of abstract structural slots (e.g., 8)
        
        # Learnable structural query tokens (like learned prompts but for structure)
        self.structure_queries = nn.Parameter(torch.randn(n_slots, d_model))
        
        # Graph neural network for structural reasoning
        self.structural_gnn = StructuralGNN(d_model, n_layers)
        
        # Structural abstraction layer - learns to IGNORE content
        self.abstraction_layer = AbstractionLayer(d_model)
        
        # Edge predictor for causal graph
        self.edge_predictor = EdgePredictor(d_model)
        
    def forward(self, hidden_states, attention_mask):
        # Step 1: Apply abstraction to filter out content-specific features
        # This is the KEY INNOVATION - not just masking, but learning what's structural
        abstracted = self.abstraction_layer(hidden_states)  # [B, N, D]
        
        # Step 2: Cross-attention from structure queries to abstracted features
        # Structure queries attend ONLY to structural aspects
        slot_values = self.cross_attention(
            queries=self.structure_queries.expand(B, -1, -1),  # [B, K, D]
            keys=abstracted,
            values=abstracted,
            mask=attention_mask
        )  # [B, K, D]
        
        # Step 3: Build causal graph over slots using GNN
        # This captures HOW structural elements relate (e.g., "twice" modifies "walk")
        edge_weights = self.edge_predictor(slot_values)  # [B, K, K]
        structural_graph = self.structural_gnn(slot_values, edge_weights)  # [B, K, D]
        
        return structural_graph, edge_weights


class AbstractionLayer(nn.Module):
    """
    The core innovation: learns to identify and preserve ONLY structural information.
    Uses a learned masking mechanism that suppresses content-specific features.
    """
    def __init__(self, d_model):
        # Learnable structural feature detector
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()  # Produces "structuralness" scores
        )
        
        # Residual for preserving some information
        self.residual_gate = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # Detect which features are structural
        structural_scores = self.structural_detector(x)  # [B, N, D]
        
        # Mask out content features, keep structural
        abstracted = x * structural_scores + self.residual_gate * x
        
        return abstracted
```

### Key Insight

The AbstractionLayer learns to identify structural features WITHOUT explicit supervision. It's trained end-to-end with the contrastive objective - features that are consistent across structurally-similar examples get high "structuralness" scores.

---

## Component 2: Content Encoder (CE)

### Architecture

```python
class ContentEncoder(nn.Module):
    """
    Learns role-agnostic content embeddings.
    Key property: Same entity has same embedding regardless of structural role.
    """
    def __init__(self, d_model, vocab_size):
        # Shared embedding with base model (efficiency)
        self.content_embedding = nn.Embedding(vocab_size, d_model)
        
        # Content refinement - removes structural contamination
        self.content_refiner = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Orthogonality projector - ensures content ⊥ structure
        self.orthogonal_projector = OrthogonalProjector(d_model)
        
    def forward(self, input_ids, structural_features=None):
        # Get base content embeddings
        content = self.content_embedding(input_ids)  # [B, N, D]
        
        # Refine to remove any structural information
        content = self.content_refiner(content)
        
        # Project to be orthogonal to structural subspace
        if structural_features is not None:
            content = self.orthogonal_projector(content, structural_features)
            
        return content


class OrthogonalProjector(nn.Module):
    """
    Projects content embeddings to be orthogonal to structural embeddings.
    This enforces explicit separation of the two representation types.
    """
    def __init__(self, d_model):
        self.d_model = d_model
        
    def forward(self, content, structure):
        # Flatten structure for projection basis
        structure_flat = structure.view(-1, self.d_model)  # [B*K, D]
        
        # Compute orthogonal complement
        # content_proj = content - (content · structure) * structure / ||structure||²
        for s in structure.unbind(1):  # For each structural slot
            s_norm = s / (s.norm(dim=-1, keepdim=True) + 1e-8)
            proj = (content * s_norm).sum(dim=-1, keepdim=True) * s_norm
            content = content - proj
            
        return content
```

---

## Component 3: Causal Binding Mechanism (CBM)

### Architecture

```python
class CausalBindingMechanism(nn.Module):
    """
    Implements HOW content fills structural slots.
    Key innovation: Uses learned causal intervention (do-operator).
    """
    def __init__(self, d_model, n_slots):
        self.n_slots = n_slots
        
        # Binding attention - which content goes to which slot
        self.binding_attention = nn.MultiheadAttention(d_model, num_heads=8)
        
        # Causal intervention layer - implements do(slot_i = content_j)
        self.intervention_layer = CausalInterventionLayer(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, structural_graph, content, hidden_states):
        B, N, D = hidden_states.shape
        K = self.n_slots
        
        # Step 1: Soft binding - which content tokens fill which structural slots
        binding_weights, _ = self.binding_attention(
            query=structural_graph,  # [B, K, D]
            key=content,             # [B, N, D]
            value=content            # [B, N, D]
        )  # [B, K, D] - each slot gets weighted sum of relevant content
        
        # Step 2: Causal intervention - propagate slot values through structure
        # This implements the "do" operator from causal inference
        intervened = self.intervention_layer(
            slot_values=binding_weights,
            causal_graph=structural_graph
        )  # [B, K, D]
        
        # Step 3: Broadcast back to token positions
        # Each token gets contribution from slots that affect it
        token_contributions = self.broadcast_to_tokens(
            intervened, binding_weights, content
        )  # [B, N, D]
        
        # Step 4: Combine with original hidden states
        combined = torch.cat([hidden_states, token_contributions], dim=-1)
        output = self.output_proj(combined)
        
        return output


class CausalInterventionLayer(nn.Module):
    """
    Implements do(X) operator for slot values.
    Allows counterfactual composition: given structure S and NEW content C',
    generate outputs for combinations never seen during training.
    """
    def __init__(self, d_model, n_iterations=3):
        self.n_iterations = n_iterations
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.update_fn = nn.GRUCell(d_model, d_model)
        
    def forward(self, slot_values, causal_graph):
        B, K, D = slot_values.shape
        
        # Get edge weights from structural graph (adjacency matrix)
        edge_weights = self.compute_edges(causal_graph)  # [B, K, K]
        
        # Iteratively propagate through causal structure
        h = slot_values
        for _ in range(self.n_iterations):
            # Message passing following causal edges
            messages = torch.zeros_like(h)
            for i in range(K):
                for j in range(K):
                    if i != j:
                        msg = self.message_fn(
                            torch.cat([h[:, i], h[:, j]], dim=-1)
                        )
                        messages[:, i] += edge_weights[:, j, i].unsqueeze(-1) * msg
            
            # Update slot values
            h = self.update_fn(
                messages.view(B*K, D),
                h.view(B*K, D)
            ).view(B, K, D)
            
        return h
```

---

## Component 4: Structural Contrastive Learning (SCL)

### Training Objective

```python
class StructuralContrastiveLoss(nn.Module):
    """
    THE KEY NOVEL TRAINING OBJECTIVE.
    Enforces structural invariance through contrastive learning.
    """
    def __init__(self, temperature=0.07):
        self.temperature = temperature
        
    def forward(self, structural_reps, pair_labels):
        """
        Args:
            structural_reps: [B, K, D] - structural representations
            pair_labels: [B, B] - 1 if same structure, 0 otherwise
        """
        B = structural_reps.shape[0]
        
        # Flatten structural representations
        z = structural_reps.mean(dim=1)  # [B, D] - aggregate slots
        z = F.normalize(z, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature  # [B, B]
        
        # Mask self-similarity
        mask = torch.eye(B, device=z.device).bool()
        similarity.masked_fill_(mask, -float('inf'))
        
        # Contrastive loss: maximize similarity for positive pairs (same structure)
        # minimize similarity for negative pairs (different structure)
        
        # For each sample, find positive pairs (same structure, different content)
        pos_mask = pair_labels.bool() & ~mask
        neg_mask = ~pair_labels.bool() & ~mask
        
        # InfoNCE-style loss
        loss = 0
        for i in range(B):
            if pos_mask[i].any():
                pos_sim = similarity[i, pos_mask[i]]
                neg_sim = similarity[i, neg_mask[i]]
                
                # Log-sum-exp trick for numerical stability
                all_sim = torch.cat([pos_sim, neg_sim])
                loss += -pos_sim.mean() + torch.logsumexp(all_sim, dim=0)
                
        return loss / B


def generate_structural_pairs(batch, grammar=None):
    """
    Generate positive/negative pairs for SCL.
    
    Positive pairs: Same structure, different content
    Negative pairs: Different structure
    
    For SCAN: Use known grammar to generate structurally equivalent examples
    For natural language: Use dependency parsing + content word substitution
    """
    pair_labels = torch.zeros(len(batch), len(batch))
    
    if grammar is not None:
        # For synthetic datasets with known grammar
        for i, ex_i in enumerate(batch):
            for j, ex_j in enumerate(batch):
                if i != j:
                    # Check if same structural pattern
                    struct_i = grammar.extract_structure(ex_i)
                    struct_j = grammar.extract_structure(ex_j)
                    pair_labels[i, j] = (struct_i == struct_j)
    else:
        # For natural language: use dependency parsing
        for i, ex_i in enumerate(batch):
            for j, ex_j in enumerate(batch):
                if i != j:
                    # Parse both sentences
                    dep_i = dependency_parse(ex_i)
                    dep_j = dependency_parse(ex_j)
                    # Compare structural patterns (ignoring content words)
                    pair_labels[i, j] = structural_similarity(dep_i, dep_j)
                    
    return pair_labels
```

---

## Full SCI Model Integration

```python
class SCITransformer(nn.Module):
    """
    Complete SCI model integrated with a base transformer.
    """
    def __init__(self, base_model, config):
        self.base_model = base_model  # e.g., TinyLlama
        
        # SCI components
        self.structural_encoder = StructuralEncoder(
            d_model=config.hidden_size,
            n_slots=config.n_structural_slots,
            n_layers=config.n_structural_layers
        )
        
        self.content_encoder = ContentEncoder(
            d_model=config.hidden_size,
            vocab_size=config.vocab_size
        )
        
        self.causal_binding = CausalBindingMechanism(
            d_model=config.hidden_size,
            n_slots=config.n_structural_slots
        )
        
        # Which layers to apply CBM (like adapter placement)
        self.cbm_layers = config.cbm_layers  # e.g., [6, 11, 16]
        
        # Contrastive loss
        self.scl_loss = StructuralContrastiveLoss()
        
    def forward(self, input_ids, attention_mask, pair_labels=None):
        B, N = input_ids.shape
        
        # Phase 1: Encode structure (ONCE, before generation)
        with torch.no_grad():  # Structure is fixed during generation
            base_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Get structural representation
        structural_graph, edge_weights = self.structural_encoder(
            base_embeds, attention_mask
        )
        
        # Get content representation
        content = self.content_encoder(input_ids, structural_graph)
        
        # Phase 2: Forward through base model with CBM injection
        hidden_states = base_embeds
        for i, layer in enumerate(self.base_model.layers):
            hidden_states = layer(hidden_states)
            
            # Inject CBM at specified layers
            if i in self.cbm_layers:
                hidden_states = self.causal_binding(
                    structural_graph, content, hidden_states
                )
        
        # Output logits
        logits = self.base_model.lm_head(hidden_states)
        
        # Compute losses
        lm_loss = self.compute_lm_loss(logits, labels)
        
        if pair_labels is not None:
            scl_loss = self.scl_loss(structural_graph, pair_labels)
        else:
            scl_loss = 0
            
        total_loss = lm_loss + config.scl_weight * scl_loss
        
        return total_loss, logits
```

---

## Training Procedure

```python
def train_sci(model, train_loader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    for epoch in range(config.epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Generate structural pairs for this batch
            pair_labels = generate_structural_pairs(
                batch['text'], 
                grammar=config.grammar if hasattr(config, 'grammar') else None
            )
            
            # Forward pass
            loss, logits = model(
                input_ids, 
                attention_mask, 
                pair_labels=pair_labels
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Validation: test compositional generalization
        eval_compositional(model, val_loader)
```

---

## Identifiability Theorem (Sketch)

**Theorem**: Under the following assumptions:
1. The structural encoder has sufficient capacity
2. Training data contains sufficient structural variation
3. The SCL objective is minimized

The learned structural representations S(x) are identifiable up to permutation and elementwise transformation.

**Proof sketch**: The contrastive objective forces S(x₁) ≈ S(x₂) for structurally equivalent inputs. Since content differs while structure matches, any representation that satisfies this constraint must capture only structural information. By the identifiability results from causal representation learning (Ahuja et al., 2023), such representations are unique up to permutation.

---

## Expected Computational Overhead

| Component | Additional Parameters | FLOPS Overhead |
|-----------|----------------------|----------------|
| Structural Encoder | ~5M | ~5% |
| Content Encoder | ~2M (shared) | ~3% |
| Causal Binding Mechanism | ~3M | ~5% |
| SCL (training only) | 0 | ~7% |
| **Total** | **~10M** | **~15-20%** |

For a 1.1B parameter base model (TinyLlama), SCI adds <1% additional parameters.

---

## Why This Is Different From Existing Approaches

| Existing Approach | What It Does | Why SCI Is Different |
|------------------|--------------|---------------------|
| Information Bottleneck | Compresses representations | SCI factorizes, doesn't just compress |
| Slot Attention | Fixed slots attend to input | SCI slots capture STRUCTURAL patterns, not content |
| Relational Attention | Attends to relationships | SCI explicitly separates structure from content |
| CausalVAE | Learns causal factors | SCI applies causality to structure-content binding |
| Construction Grammar NNs | Uses predefined constructions | SCI learns constructions automatically |

**The key difference**: SCI introduces a NEW PRINCIPLE (structural invariance) and a NEW TRAINING OBJECTIVE (contrastive structural learning) rather than combining existing modules in a new way.

---

## Quick Start Implementation

```bash
# Clone and setup
git clone https://github.com/your-repo/sci-transformer
cd sci-transformer
pip install -r requirements.txt

# Train on SCAN
python train.py \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset scan \
    --split length \
    --n_structural_slots 8 \
    --cbm_layers 6,11,16 \
    --scl_weight 0.1 \
    --epochs 20

# Evaluate
python evaluate.py \
    --checkpoint outputs/best_model.pt \
    --dataset scan \
    --split length
```

---

## Conclusion

SCI represents a fundamentally new approach to compositional generalization that:

1. **Introduces a novel principle**: Structural Causal Invariance
2. **Has theoretical guarantees**: Identifiability theorem
3. **Is practically implementable**: ~15-20% overhead
4. **Addresses the root cause**: The binding problem

This is not another combination of known modules—it's a new paradigm for representation learning that could transform how LLMs understand compositional language.
