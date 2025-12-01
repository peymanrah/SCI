# SCI Agent Instructions - Part 2: Main Model, Losses, and Data

## Step 3.4: Main SCI Model (src/models/sci_model.py)

**AI AGENT: This integrates all SCI components with the base transformer.**

```python
# src/models/sci_model.py
"""
Main SCI Model - Integrates all SCI components with base transformer.

CRITICAL IMPLEMENTATION NOTES:
1. SE and CE process ONLY the instruction, NOT the response
2. CBM is injected at specific layers (configured)
3. All four components must be present for full SCI
4. Ablations disable specific components via config
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .structural_encoder import StructuralEncoder
from .content_encoder import ContentEncoder
from .causal_binding import CausalBindingMechanism


class SCIModel(nn.Module):
    """
    Complete SCI Model integrating all components.
    
    Architecture:
    - Base Model: Pre-trained transformer (e.g., TinyLlama)
    - Structural Encoder (SE): Extracts structural representation
    - Content Encoder (CE): Extracts role-agnostic content
    - Causal Binding Mechanism (CBM): Combines structure and content
    
    The CBM is injected at specified transformer layers via forward hooks.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['base_model'],
            torch_dtype=torch.float16 if config['training']['fp16'] else torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
        
        # Get model dimensions
        self.hidden_size = self.base_model.config.hidden_size
        self.num_layers = self.base_model.config.num_hidden_layers
        
        # Initialize SCI components based on config
        sci_config = config['sci']
        
        # Structural Encoder
        if sci_config['structural_encoder'].get('enabled', True):
            self.structural_encoder = StructuralEncoder(
                d_model=self.hidden_size,
                n_slots=sci_config['structural_encoder']['n_slots'],
                n_layers=sci_config['structural_encoder']['n_layers'],
                n_heads=sci_config['structural_encoder']['n_heads'],
                abstraction_hidden_mult=sci_config['structural_encoder']['abstraction_hidden_mult']
            )
        else:
            self.structural_encoder = None
            
        # Content Encoder
        if sci_config['content_encoder'].get('enabled', True):
            base_embeddings = self.base_model.get_input_embeddings()
            self.content_encoder = ContentEncoder(
                d_model=self.hidden_size,
                vocab_size=self.base_model.config.vocab_size,
                n_refiner_layers=sci_config['content_encoder']['refiner_layers'],
                share_embeddings=sci_config['content_encoder']['shared_embeddings'],
                base_embeddings=base_embeddings if sci_config['content_encoder']['shared_embeddings'] else None
            )
        else:
            self.content_encoder = None
            
        # Causal Binding Mechanism
        if sci_config['causal_binding'].get('enabled', True):
            self.causal_binding = CausalBindingMechanism(
                d_model=self.hidden_size,
                n_slots=sci_config['structural_encoder']['n_slots'],
                n_iterations=sci_config['causal_binding']['n_iterations'],
            )
            self.cbm_layers = sci_config['causal_binding']['injection_layers']
        else:
            self.causal_binding = None
            self.cbm_layers = []
            
        # Store SCI representations for each forward pass
        self.current_structural_graph = None
        self.current_edge_weights = None
        self.current_content = None
        
        # Register forward hooks for CBM injection
        self._register_cbm_hooks()
        
    def _register_cbm_hooks(self):
        """Register forward hooks to inject CBM at specified layers."""
        if self.causal_binding is None:
            return
            
        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                # outputs is a tuple (hidden_states, ...)
                hidden_states = outputs[0]
                
                # Apply CBM if we have SCI representations
                if (self.current_structural_graph is not None and 
                    self.current_content is not None):
                    
                    # Get attention mask from the batch
                    # This is stored during the forward pass
                    attention_mask = getattr(self, '_current_attention_mask', None)
                    
                    modified = self.causal_binding(
                        hidden_states=hidden_states,
                        structural_graph=self.current_structural_graph,
                        edge_weights=self.current_edge_weights,
                        content=self.current_content,
                        attention_mask=attention_mask
                    )
                    
                    # Return modified hidden states
                    return (modified,) + outputs[1:]
                    
                return outputs
            return hook
        
        # Get transformer layers
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Cannot find transformer layers in base model")
        
        # Register hooks at specified layers
        for layer_idx in self.cbm_layers:
            if layer_idx < len(layers):
                layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                
    def get_instruction_mask(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Create mask that is 1 for instruction tokens, 0 for response tokens.
        
        CRITICAL: This prevents data leakage - SE and CE must NOT see the response.
        
        Labels have -100 for instruction tokens (not included in loss).
        """
        # labels == -100 indicates tokens that are NOT part of the target
        # These are the instruction tokens
        instruction_mask = (labels == -100).long()
        
        return instruction_mask
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pair_labels: Optional[torch.Tensor] = None,  # For SCL
        return_loss: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SCI components.
        
        Args:
            input_ids: [B, N] token IDs
            attention_mask: [B, N] attention mask
            labels: [B, N] target labels (-100 for instruction tokens)
            pair_labels: [B, B] matrix indicating structural similarity
            return_loss: Whether to compute and return losses
            
        Returns:
            Dictionary with logits, losses, and SCI representations
        """
        B, N = input_ids.shape
        
        # Store attention mask for hooks
        self._current_attention_mask = attention_mask
        
        # Get instruction mask (CRITICAL for no data leakage)
        if labels is not None:
            instruction_mask = self.get_instruction_mask(input_ids, labels)
        else:
            # During inference, treat everything as instruction
            instruction_mask = attention_mask
        
        # Get base embeddings for instruction tokens only
        base_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Mask out response tokens for SCI encoders
        instruction_embeddings = base_embeddings * instruction_mask.unsqueeze(-1)
        
        # ===================
        # SCI Encoding Phase
        # ===================
        
        # Structural Encoder
        if self.structural_encoder is not None:
            structural_graph, edge_weights = self.structural_encoder(
                instruction_embeddings,
                instruction_mask  # Only attend to instruction tokens
            )
            self.current_structural_graph = structural_graph
            self.current_edge_weights = edge_weights
        else:
            structural_graph = None
            edge_weights = None
            self.current_structural_graph = None
            self.current_edge_weights = None
            
        # Content Encoder
        if self.content_encoder is not None:
            # Create instruction-only input_ids
            instruction_ids = input_ids * instruction_mask
            content = self.content_encoder(
                instruction_ids,
                structural_features=structural_graph
            )
            self.current_content = content
        else:
            content = None
            self.current_content = None
            
        # ===================
        # Base Model Forward
        # ===================
        
        # Run base model (CBM is injected via hooks)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        # ===================
        # Loss Computation
        # ===================
        
        result = {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'structural_graph': structural_graph,
            'edge_weights': edge_weights,
            'content': content,
        }
        
        if return_loss and labels is not None:
            # Language modeling loss (from base model)
            lm_loss = outputs.loss
            result['lm_loss'] = lm_loss
            
            # Orthogonality loss (content ⊥ structure)
            if self.content_encoder is not None and structural_graph is not None:
                ortho_loss = self.content_encoder.compute_orthogonality_loss(
                    content, structural_graph
                )
                result['orthogonality_loss'] = ortho_loss
            else:
                ortho_loss = 0.0
                result['orthogonality_loss'] = torch.tensor(0.0)
                
            # Total loss (SCL is computed externally with pair_labels)
            result['total_loss'] = lm_loss + 0.01 * ortho_loss
            
        # Clear stored representations
        self.current_structural_graph = None
        self.current_edge_weights = None
        self.current_content = None
        
        return result
    
    def get_structural_representation(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get structural representation for analysis/comparison.
        
        Used for testing structural invariance.
        """
        if self.structural_encoder is None:
            raise ValueError("Structural encoder not enabled")
            
        base_embeddings = self.base_model.get_input_embeddings()(input_ids)
        return self.structural_encoder(base_embeddings, attention_mask)
    
    def save_pretrained(self, save_path: str):
        """Save SCI model and base model."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(os.path.join(save_path, 'base_model'))
        self.tokenizer.save_pretrained(os.path.join(save_path, 'base_model'))
        
        # Save SCI components
        sci_state = {}
        if self.structural_encoder is not None:
            sci_state['structural_encoder'] = self.structural_encoder.state_dict()
        if self.content_encoder is not None:
            sci_state['content_encoder'] = self.content_encoder.state_dict()
        if self.causal_binding is not None:
            sci_state['causal_binding'] = self.causal_binding.state_dict()
            
        torch.save(sci_state, os.path.join(save_path, 'sci_components.pt'))
        
        # Save config
        import yaml
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
            
    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load SCI model from checkpoint."""
        import os
        import yaml
        
        # Load config
        with open(os.path.join(load_path, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
            
        # Create model
        model = cls(config)
        
        # Load SCI components
        sci_state = torch.load(os.path.join(load_path, 'sci_components.pt'))
        
        if 'structural_encoder' in sci_state and model.structural_encoder is not None:
            model.structural_encoder.load_state_dict(sci_state['structural_encoder'])
        if 'content_encoder' in sci_state and model.content_encoder is not None:
            model.content_encoder.load_state_dict(sci_state['content_encoder'])
        if 'causal_binding' in sci_state and model.causal_binding is not None:
            model.causal_binding.load_state_dict(sci_state['causal_binding'])
            
        return model
```

---

## Step 3.5: Structural Contrastive Loss (src/losses/structural_contrastive.py)

```python
# src/losses/structural_contrastive.py
"""
Structural Contrastive Learning (SCL) Loss

THE KEY NOVEL TRAINING OBJECTIVE.

Enforces structural invariance: inputs with same structure but different content
should have SIMILAR structural representations.

CRITICAL: This is what makes SCI work. Without proper SCL, the model
won't learn to separate structure from content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StructuralContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for structural invariance.
    
    Positive pairs: Same structure, different content
    Negative pairs: Different structure
    
    Objective: Pull together structural representations of positive pairs,
    push apart representations of negative pairs.
    """
    
    def __init__(
        self, 
        temperature: float = 0.07,
        normalize: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(
        self,
        structural_reps: torch.Tensor,  # [B, K, D] - structural representations
        pair_labels: torch.Tensor       # [B, B] - 1 if same structure, 0 otherwise
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            structural_reps: Structural graph representations from SE
            pair_labels: Binary matrix indicating structural equivalence
            
        Returns:
            Contrastive loss value
        """
        B = structural_reps.shape[0]
        device = structural_reps.device
        
        # Aggregate slot representations (mean pooling)
        z = structural_reps.mean(dim=1)  # [B, D]
        
        # Normalize
        if self.normalize:
            z = F.normalize(z, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature  # [B, B]
        
        # Create masks
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = pair_labels.bool() & ~self_mask  # Positive pairs (excluding self)
        neg_mask = ~pair_labels.bool() & ~self_mask  # Negative pairs
        
        # Check if we have valid pairs
        if not pos_mask.any():
            # No positive pairs in batch - return 0 loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # InfoNCE loss
        # For each anchor, compute log(exp(pos) / (exp(pos) + sum(exp(neg))))
        loss = 0.0
        n_valid = 0
        
        for i in range(B):
            if not pos_mask[i].any():
                continue
                
            # Positive similarities for anchor i
            pos_sim = similarity[i][pos_mask[i]]  # [num_pos]
            
            # Negative similarities for anchor i
            neg_sim = similarity[i][neg_mask[i]]  # [num_neg]
            
            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # For multiple positives, average over them
            for pos in pos_sim:
                numerator = torch.exp(pos)
                denominator = numerator + torch.exp(neg_sim).sum()
                loss += -torch.log(numerator / (denominator + 1e-8))
                n_valid += 1
        
        if n_valid > 0:
            loss = loss / n_valid
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss
    
    
class HardNegativeContrastiveLoss(nn.Module):
    """
    Enhanced contrastive loss with hard negative mining.
    
    Focuses on the hardest negatives (most similar but structurally different)
    to provide stronger learning signal.
    """
    
    def __init__(
        self, 
        temperature: float = 0.07,
        hard_negative_ratio: float = 0.3  # Use top 30% hardest negatives
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        
    def forward(
        self,
        structural_reps: torch.Tensor,
        pair_labels: torch.Tensor
    ) -> torch.Tensor:
        B = structural_reps.shape[0]
        device = structural_reps.device
        
        z = F.normalize(structural_reps.mean(dim=1), dim=-1)
        similarity = torch.matmul(z, z.T) / self.temperature
        
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = pair_labels.bool() & ~self_mask
        neg_mask = ~pair_labels.bool() & ~self_mask
        
        if not pos_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = 0.0
        n_valid = 0
        
        for i in range(B):
            if not pos_mask[i].any():
                continue
                
            pos_sim = similarity[i][pos_mask[i]]
            neg_sim = similarity[i][neg_mask[i]]
            
            if len(neg_sim) == 0:
                continue
            
            # Hard negative mining: select top-k most similar negatives
            k = max(1, int(len(neg_sim) * self.hard_negative_ratio))
            hard_neg_sim, _ = torch.topk(neg_sim, k)
            
            for pos in pos_sim:
                numerator = torch.exp(pos)
                denominator = numerator + torch.exp(hard_neg_sim).sum()
                loss += -torch.log(numerator / (denominator + 1e-8))
                n_valid += 1
        
        return loss / max(n_valid, 1)
```

---

## Step 3.6: Combined Loss (src/losses/combined_loss.py)

```python
# src/losses/combined_loss.py
"""
Combined Loss for SCI Training

Combines:
1. Language Modeling Loss (cross-entropy)
2. Structural Contrastive Loss (SCL)
3. Orthogonality Loss (optional)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .structural_contrastive import StructuralContrastiveLoss, HardNegativeContrastiveLoss


class SCICombinedLoss(nn.Module):
    """
    Combined loss function for SCI training.
    
    total_loss = lm_loss + scl_weight * scl_loss + ortho_weight * ortho_loss
    """
    
    def __init__(
        self,
        scl_weight: float = 0.1,
        ortho_weight: float = 0.01,
        temperature: float = 0.07,
        use_hard_negatives: bool = True
    ):
        super().__init__()
        self.scl_weight = scl_weight
        self.ortho_weight = ortho_weight
        
        # SCL loss
        if use_hard_negatives:
            self.scl_loss_fn = HardNegativeContrastiveLoss(temperature=temperature)
        else:
            self.scl_loss_fn = StructuralContrastiveLoss(temperature=temperature)
            
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        pair_labels: Optional[torch.Tensor] = None,
        scl_weight_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            model_outputs: Dictionary from SCIModel forward pass
            pair_labels: [B, B] structural similarity matrix
            scl_weight_override: Override SCL weight (for warmup scheduling)
            
        Returns:
            Dictionary with individual losses and total loss
        """
        device = model_outputs['logits'].device
        
        # LM loss (already computed by base model)
        lm_loss = model_outputs.get('lm_loss', torch.tensor(0.0, device=device))
        
        # SCL loss
        if pair_labels is not None and model_outputs.get('structural_graph') is not None:
            scl_loss = self.scl_loss_fn(
                model_outputs['structural_graph'],
                pair_labels
            )
        else:
            scl_loss = torch.tensor(0.0, device=device)
            
        # Orthogonality loss
        ortho_loss = model_outputs.get('orthogonality_loss', torch.tensor(0.0, device=device))
        
        # Compute total loss
        scl_weight = scl_weight_override if scl_weight_override is not None else self.scl_weight
        total_loss = lm_loss + scl_weight * scl_loss + self.ortho_weight * ortho_loss
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'scl_loss': scl_loss,
            'orthogonality_loss': ortho_loss
        }
```

---

## Phase 4: Data and Pair Generation

### Step 4.1: Structure Extractor (src/data/structure_extractor.py)

**AI AGENT: This is how we automatically identify structural patterns for pair generation.**

```python
# src/data/structure_extractor.py
"""
Structure Extractor - Automatically extracts structural patterns from examples.

This is CRITICAL for generating training pairs without human annotation.

For SCAN: Uses the known grammar rules
For COGS: Uses semantic structure patterns
For general text: Uses dependency parsing
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StructuralPattern:
    """Represents an extracted structural pattern."""
    pattern_type: str           # e.g., "twice", "and", "after"
    pattern_template: str       # e.g., "X twice", "X and Y"
    content_slots: List[str]    # e.g., ["walk"], ["run", "jump"]
    raw_text: str               # Original text


class SCANStructureExtractor:
    """
    Structure extractor for SCAN dataset.
    
    SCAN has a known grammar, so we can extract exact structural patterns.
    
    Grammar rules:
    - "X twice" -> repeat X
    - "X thrice" -> repeat X 3 times
    - "X and Y" -> sequence X then Y
    - "X after Y" -> sequence Y then X
    - "turn left/right" -> direction modifier
    - "turn opposite left/right" -> double turn
    - "turn around left/right" -> full rotation
    
    Structure = the compositional pattern
    Content = the action words (walk, run, jump, look)
    """
    
    # Action words (content)
    ACTIONS = {'walk', 'run', 'jump', 'look'}
    
    # Structural modifiers
    MODIFIERS = {'twice', 'thrice', 'and', 'after', 'around', 'opposite'}
    
    # Directions
    DIRECTIONS = {'left', 'right'}
    
    def extract(self, command: str) -> StructuralPattern:
        """
        Extract structural pattern from SCAN command.
        
        Returns a StructuralPattern where content words are replaced with slots.
        """
        command = command.lower().strip()
        
        # Tokenize
        tokens = command.split()
        
        # Identify structure and content
        structure_tokens = []
        content_slots = []
        slot_counter = 0
        
        for token in tokens:
            if token in self.ACTIONS:
                # Replace action with slot
                slot_name = f"ACTION_{slot_counter}"
                structure_tokens.append(slot_name)
                content_slots.append(token)
                slot_counter += 1
            else:
                # Keep structural words
                structure_tokens.append(token)
                
        pattern_template = ' '.join(structure_tokens)
        
        # Determine pattern type
        pattern_type = self._identify_pattern_type(tokens)
        
        return StructuralPattern(
            pattern_type=pattern_type,
            pattern_template=pattern_template,
            content_slots=content_slots,
            raw_text=command
        )
    
    def _identify_pattern_type(self, tokens: List[str]) -> str:
        """Identify the main structural pattern."""
        if 'twice' in tokens:
            return 'repetition_twice'
        elif 'thrice' in tokens:
            return 'repetition_thrice'
        elif 'and' in tokens:
            return 'sequence_and'
        elif 'after' in tokens:
            return 'sequence_after'
        elif 'around' in tokens:
            return 'rotation_around'
        elif 'opposite' in tokens:
            return 'rotation_opposite'
        else:
            return 'simple'
            
    def are_structurally_equivalent(
        self, 
        pattern1: StructuralPattern, 
        pattern2: StructuralPattern
    ) -> bool:
        """
        Check if two patterns have the same structure.
        
        Two patterns are structurally equivalent if they have the same template
        (ignoring the specific content words).
        """
        return pattern1.pattern_template == pattern2.pattern_template
    
    def generate_structural_variant(
        self, 
        pattern: StructuralPattern,
        new_content: List[str]
    ) -> str:
        """
        Generate a new command with same structure but different content.
        """
        if len(new_content) != len(pattern.content_slots):
            raise ValueError("Content length mismatch")
            
        result = pattern.pattern_template
        for i, content in enumerate(new_content):
            result = result.replace(f"ACTION_{i}", content)
            
        return result


class COGSStructureExtractor:
    """
    Structure extractor for COGS dataset.
    
    COGS uses semantic structures with predicates and arguments.
    Structure = predicate patterns and argument relationships
    Content = specific entities and nouns
    """
    
    # Common predicates (structural)
    PREDICATES = {
        'agent', 'theme', 'recipient', 'goal', 'source',
        'nmod', 'ccomp', 'xcomp'
    }
    
    def extract(self, sentence: str, logical_form: Optional[str] = None) -> StructuralPattern:
        """
        Extract structural pattern from COGS example.
        
        If logical_form is provided, use it directly.
        Otherwise, use heuristic extraction.
        """
        if logical_form:
            return self._extract_from_logical_form(sentence, logical_form)
        else:
            return self._extract_heuristic(sentence)
            
    def _extract_from_logical_form(
        self, 
        sentence: str, 
        logical_form: str
    ) -> StructuralPattern:
        """Extract structure from COGS logical form."""
        # COGS logical form example: "* cat ( x _ 1 ) ; dog . agent ( x _ 2 , x _ 1 )"
        
        # Extract predicates and their argument patterns
        predicates = re.findall(r'(\w+)\s*\([^)]+\)', logical_form)
        
        # Replace specific entities with slots
        template = logical_form
        entities = re.findall(r'\*\s*(\w+)', logical_form)
        
        content_slots = list(set(entities))
        for i, entity in enumerate(content_slots):
            template = template.replace(f'* {entity}', f'* ENTITY_{i}')
            
        pattern_type = '_'.join(sorted(set(predicates)))
        
        return StructuralPattern(
            pattern_type=pattern_type,
            pattern_template=template,
            content_slots=content_slots,
            raw_text=sentence
        )
        
    def _extract_heuristic(self, sentence: str) -> StructuralPattern:
        """Heuristic extraction without logical form."""
        tokens = sentence.lower().split()
        
        # Simple heuristic: nouns are content, rest is structure
        # This is a fallback - logical form extraction is preferred
        content_slots = []
        structure_tokens = []
        
        # Very simple POS-like tagging based on position
        for i, token in enumerate(tokens):
            # Assume capitalized words or words after articles are nouns (content)
            if i > 0 and tokens[i-1] in {'a', 'an', 'the'}:
                content_slots.append(token)
                structure_tokens.append(f'NOUN_{len(content_slots)-1}')
            else:
                structure_tokens.append(token)
                
        return StructuralPattern(
            pattern_type='heuristic',
            pattern_template=' '.join(structure_tokens),
            content_slots=content_slots,
            raw_text=sentence
        )
        
    def are_structurally_equivalent(
        self, 
        pattern1: StructuralPattern, 
        pattern2: StructuralPattern
    ) -> bool:
        """Check structural equivalence for COGS patterns."""
        # For COGS, same predicate structure = equivalent
        return pattern1.pattern_type == pattern2.pattern_type


def get_structure_extractor(dataset_name: str):
    """Factory function to get appropriate extractor."""
    extractors = {
        'scan': SCANStructureExtractor,
        'cogs': COGSStructureExtractor,
    }
    
    if dataset_name not in extractors:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(extractors.keys())}")
        
    return extractors[dataset_name]()
```

### Step 4.2: Pair Generator (src/data/pair_generator.py)

```python
# src/data/pair_generator.py
"""
Automated Structural Pair Generator

Generates positive and negative pairs for SCL training WITHOUT human annotation.

Positive pairs: Same structure, different content
Negative pairs: Different structure

This is THE critical component for making SCI trainable at scale.
"""

import torch
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from .structure_extractor import (
    StructuralPattern,
    SCANStructureExtractor,
    COGSStructureExtractor,
    get_structure_extractor
)


@dataclass
class StructuralPair:
    """A pair of examples with structural relationship."""
    example1_idx: int
    example2_idx: int
    is_positive: bool  # True if same structure
    structure_type: str


class StructuralPairGenerator:
    """
    Generates training pairs for SCL.
    
    Strategy:
    1. Extract structural patterns from all examples
    2. Group examples by structural pattern
    3. Generate positive pairs within groups
    4. Generate negative pairs across groups
    """
    
    def __init__(
        self,
        dataset_name: str,
        pairs_per_example: int = 5,
        negative_ratio: float = 1.0
    ):
        self.extractor = get_structure_extractor(dataset_name)
        self.pairs_per_example = pairs_per_example
        self.negative_ratio = negative_ratio
        
        # Cache for extracted patterns
        self.patterns: List[StructuralPattern] = []
        self.pattern_groups: Dict[str, List[int]] = defaultdict(list)
        
    def process_dataset(self, examples: List[Dict]) -> None:
        """
        Process entire dataset to extract patterns and build groups.
        
        Args:
            examples: List of dicts with 'input' (and optionally 'logical_form')
        """
        self.patterns = []
        self.pattern_groups = defaultdict(list)
        
        for idx, example in enumerate(examples):
            # Extract pattern
            if 'logical_form' in example:
                pattern = self.extractor.extract(
                    example['input'], 
                    example['logical_form']
                )
            else:
                pattern = self.extractor.extract(example['input'])
                
            self.patterns.append(pattern)
            
            # Group by pattern template (structure)
            self.pattern_groups[pattern.pattern_template].append(idx)
            
        print(f"Processed {len(examples)} examples into {len(self.pattern_groups)} structural groups")
        
    def generate_batch_pairs(
        self, 
        batch_indices: List[int]
    ) -> torch.Tensor:
        """
        Generate pair labels for a batch.
        
        Args:
            batch_indices: Indices of examples in the batch
            
        Returns:
            pair_labels: [B, B] tensor where 1 = same structure, 0 = different
        """
        B = len(batch_indices)
        pair_labels = torch.zeros(B, B)
        
        for i, idx_i in enumerate(batch_indices):
            pattern_i = self.patterns[idx_i]
            
            for j, idx_j in enumerate(batch_indices):
                if i == j:
                    continue
                    
                pattern_j = self.patterns[idx_j]
                
                # Check structural equivalence
                if self.extractor.are_structurally_equivalent(pattern_i, pattern_j):
                    pair_labels[i, j] = 1.0
                    
        return pair_labels
    
    def generate_positive_pairs(
        self, 
        batch_indices: List[int],
        max_pairs_per_anchor: int = 3
    ) -> List[StructuralPair]:
        """
        Generate explicit positive pairs for a batch.
        
        Used for ensuring sufficient positive pairs when batch is small.
        """
        pairs = []
        
        for idx in batch_indices:
            pattern = self.patterns[idx]
            template = pattern.pattern_template
            
            # Find other examples with same structure
            same_structure_indices = [
                other_idx for other_idx in self.pattern_groups[template]
                if other_idx != idx and other_idx in batch_indices
            ]
            
            # Sample positive pairs
            n_pairs = min(max_pairs_per_anchor, len(same_structure_indices))
            if n_pairs > 0:
                sampled = random.sample(same_structure_indices, n_pairs)
                for other_idx in sampled:
                    pairs.append(StructuralPair(
                        example1_idx=idx,
                        example2_idx=other_idx,
                        is_positive=True,
                        structure_type=pattern.pattern_type
                    ))
                    
        return pairs
    
    def create_augmented_batch(
        self,
        base_indices: List[int],
        target_positives_per_example: int = 2
    ) -> List[int]:
        """
        Augment batch to ensure sufficient positive pairs.
        
        If batch doesn't have enough positive pairs, add examples
        from the same structural groups.
        """
        augmented = list(base_indices)
        
        for idx in base_indices:
            pattern = self.patterns[idx]
            template = pattern.pattern_template
            
            # Count existing positives in batch
            existing_positives = sum(
                1 for other in augmented 
                if other != idx and self.patterns[other].pattern_template == template
            )
            
            # Add more if needed
            needed = target_positives_per_example - existing_positives
            if needed > 0:
                # Find candidates not already in batch
                candidates = [
                    other_idx for other_idx in self.pattern_groups[template]
                    if other_idx not in augmented
                ]
                
                if candidates:
                    to_add = random.sample(candidates, min(needed, len(candidates)))
                    augmented.extend(to_add)
                    
        return augmented


class SCANPairGenerator(StructuralPairGenerator):
    """
    Specialized pair generator for SCAN with content substitution.
    
    Can generate new examples with same structure but substituted content.
    """
    
    def __init__(self, **kwargs):
        super().__init__(dataset_name='scan', **kwargs)
        
        # Available actions for substitution
        self.actions = ['walk', 'run', 'jump', 'look']
        
    def generate_synthetic_positive(
        self, 
        pattern: StructuralPattern
    ) -> Optional[str]:
        """
        Generate a synthetic example with same structure but different content.
        """
        if not pattern.content_slots:
            return None
            
        # Choose different actions
        new_content = []
        for original in pattern.content_slots:
            alternatives = [a for a in self.actions if a != original]
            if alternatives:
                new_content.append(random.choice(alternatives))
            else:
                new_content.append(original)
                
        # Generate new command
        return self.extractor.generate_structural_variant(pattern, new_content)
```

### Step 4.3: SCAN Dataset (src/data/scan_dataset.py)

```python
# src/data/scan_dataset.py
"""
SCAN Dataset loader with structural pair generation.

SCAN is the PRIMARY training dataset for SCI because:
1. Has clear compositional structure
2. Known grammar enables exact structural matching
3. Standard benchmark for compositional generalization
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer

from .pair_generator import SCANPairGenerator, StructuralPairGenerator


class SCANDataset(Dataset):
    """
    SCAN dataset with structural pair generation for SCI training.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = 'length',  # length, template, addprim_jump, addprim_turn_left
        subset: str = 'train',  # train, test
        max_length: int = 128,
        pairs_per_example: int = 5,
        use_synthetic_pairs: bool = True
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.subset = subset
        self.max_length = max_length
        
        # Load dataset
        self.raw_data = self._load_scan(split, subset)
        
        # Initialize pair generator
        self.pair_generator = SCANPairGenerator(
            pairs_per_example=pairs_per_example
        )
        
        # Process dataset for pair generation
        examples = [{'input': ex['commands']} for ex in self.raw_data]
        self.pair_generator.process_dataset(examples)
        
        self.use_synthetic_pairs = use_synthetic_pairs
        
    def _load_scan(self, split: str, subset: str) -> List[Dict]:
        """Load SCAN dataset from HuggingFace."""
        # SCAN split names in HuggingFace datasets
        split_mapping = {
            'length': 'length_split',
            'template': 'template_around_right',
            'addprim_jump': 'addprim_jump_split',
            'addprim_turn_left': 'addprim_turn_left_split',
        }
        
        hf_split = split_mapping.get(split, split)
        
        try:
            dataset = load_dataset('scan', hf_split, split=subset)
            return list(dataset)
        except Exception as e:
            print(f"Error loading SCAN: {e}")
            # Fallback: try loading from local files
            return self._load_scan_local(split, subset)
            
    def _load_scan_local(self, split: str, subset: str) -> List[Dict]:
        """Fallback: load from local SCAN files."""
        import os
        
        # Expected format: data/scan/{split}/tasks_{subset}.txt
        path = f"data/scan/{split}/tasks_{subset}.txt"
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"SCAN data not found at {path}")
            
        examples = []
        with open(path, 'r') as f:
            for line in f:
                if 'IN:' in line and 'OUT:' in line:
                    parts = line.strip().split('OUT:')
                    command = parts[0].replace('IN:', '').strip()
                    actions = parts[1].strip()
                    examples.append({
                        'commands': command,
                        'actions': actions
                    })
                    
        return examples
        
    def __len__(self) -> int:
        return len(self.raw_data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.raw_data[idx]
        
        # Format: "IN: {command} OUT: {actions}"
        command = example['commands']
        actions = example['actions']
        
        # Create input-output pair
        # Instruction = command, Response = actions
        input_text = f"IN: {command} OUT:"
        target_text = f" {actions}"
        full_text = input_text + target_text
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels: -100 for instruction tokens, actual token IDs for response
        input_tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_length = input_tokens['input_ids'].shape[1]
        
        labels = tokenized['input_ids'].clone()
        labels[0, :input_length] = -100  # Mask instruction tokens
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'idx': idx,  # For pair generation
            'command': command,
            'actions': actions
        }
        
    def get_pair_labels(self, batch_indices: List[int]) -> torch.Tensor:
        """Get structural pair labels for a batch."""
        return self.pair_generator.generate_batch_pairs(batch_indices)


class SCANCollator:
    """
    Custom collator that generates pair labels for each batch.
    """
    
    def __init__(self, dataset: SCANDataset):
        self.dataset = dataset
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack tensors
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        
        # Get indices for pair generation
        indices = [ex['idx'] for ex in batch]
        
        # Generate pair labels
        pair_labels = self.dataset.get_pair_labels(indices)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pair_labels': pair_labels,
            'indices': indices
        }
```

---

## Phase 5: Training and Evaluation Pipeline

I'll continue with Part 3 which covers the training loop, evaluation, and verification tests.
