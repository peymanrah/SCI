"""
Main SCI Model: Integrates all SCI components with TinyLlama-1.1B.

This is the complete SCI architecture that combines:
1. Structural Encoder (SE) - Extracts structure-invariant patterns
2. Content Encoder (CE) - Extracts content independent of structure
3. Causal Binding Mechanism (CBM) - Binds content to structural slots
4. TinyLlama-1.1B decoder - Generates outputs conditioned on structure+content

CRITICAL IMPLEMENTATION NOTES:
- SE and CE must ONLY see instruction tokens (not response)
- CBM is injected into decoder at specified layers via hooks
- All components can be enabled/disabled for ablation studies
- Supports both training (with SCL loss) and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from sci.models.components.structural_encoder import StructuralEncoder
from sci.models.components.content_encoder import ContentEncoder
from sci.models.components.causal_binding import CausalBindingMechanism


class SCIModel(nn.Module):
    """
    Complete SCI Model integrating all components with TinyLlama.

    Architecture Flow:
    1. Input tokenization
    2. Instruction masking (CRITICAL: prevent data leakage)
    3. Structural Encoder → structural_slots [B, K, D]
    4. Content Encoder → content_repr [B, D]
    5. Causal Binding → bound_repr [B, K, D]
    6. Decoder with CBM injection → logits [B, N, V]

    Args:
        config: Configuration object containing all hyperparameters

    Example config structure:
        model:
            base_model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            structural_encoder:
                enabled: true
                num_slots: 8
                ...
            content_encoder:
                enabled: true
                ...
            causal_binding:
                enabled: true
                injection_layers: [6, 11, 16]
                ...
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Device placement
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load base TinyLlama model
        print(f"Loading base model: {config.model.base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_name,
            torch_dtype=torch.float16 if config.training.mixed_precision else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model dimensions from TinyLlama
        self.d_model = self.base_model.config.hidden_size  # 2048 for TinyLlama-1.1B
        self.vocab_size = self.base_model.config.vocab_size
        self.num_layers = self.base_model.config.num_hidden_layers  # 22 layers

        print(f"Model dimensions: d_model={self.d_model}, vocab_size={self.vocab_size}, num_layers={self.num_layers}")

        # HIGH #24: Add base_model compatibility checks
        assert hasattr(self.base_model, 'model'), \
            f"Base model must have 'model' attribute (got type: {type(self.base_model).__name__})"
        assert hasattr(self.base_model.model, 'layers'), \
            f"Base model must have transformer layers at model.layers"

        # Validate CBM injection layers are within bounds
        # Check both attribute name formats for compatibility
        cbm_enabled = getattr(config.model.causal_binding, 'enable_causal_intervention',
                             getattr(config.model.causal_binding, 'enabled', False))
        if cbm_enabled and len(config.model.causal_binding.injection_layers) > 0:
            max_injection_layer = max(config.model.causal_binding.injection_layers)
            assert max_injection_layer < self.num_layers, \
                f"CBM injection layer {max_injection_layer} exceeds model layers (max: {self.num_layers-1})"

        print(f"✓ Base model compatibility validated")

        # Get shared embedding layer from TinyLlama
        self.shared_embedding = self.base_model.get_input_embeddings()

        # Initialize SCI components based on config
        self._initialize_sci_components()

        # Register hooks for CBM injection
        self._register_cbm_hooks()

        # Storage for intermediate representations (used during forward pass)
        self.current_structural_slots = None
        self.current_structural_scores = None
        self.current_content_repr = None
        # CRITICAL #11: edge_weights initialization
        # Currently set to None (GNN feature not implemented)
        # When GNN is added, this will store graph edge weights [batch, num_slots, num_slots]
        self.current_edge_weights = None
        self.current_instruction_mask = None

    def _init_encoder_projection(self, encoder, name: str):
        """
        V8 FIX #5: Eagerly initialize input_projection for encoder.
        
        This avoids lazy initialization in forward() which can cause
        device/dtype mismatches in DDP/multi-GPU setups.
        
        Args:
            encoder: StructuralEncoder or ContentEncoder instance
            name: Name for logging
        """
        import torch.nn as nn
        
        # Get embedding dimension from shared embedding
        embedding_dim = self.shared_embedding.embedding_dim  # 2048 for TinyLlama
        encoder_d_model = encoder.d_model
        
        if embedding_dim != encoder_d_model:
            # Create projection layer with proper dimensions
            encoder.input_projection = nn.Linear(embedding_dim, encoder_d_model)
            print(f"  - {name} input projection: {embedding_dim} -> {encoder_d_model}")
        else:
            encoder.input_projection = nn.Identity()
            print(f"  - {name} input projection: Identity (dims match)")
        
        # Mark as initialized so forward() doesn't try to re-init
        encoder.embedding_dim = embedding_dim
        encoder._projection_initialized = True

    def _initialize_sci_components(self):
        """Initialize Structural Encoder, Content Encoder, and CBM."""

        # 1. Structural Encoder
        if self.config.model.structural_encoder.enabled:
            print("Initializing Structural Encoder...")
            self.structural_encoder = StructuralEncoder(self.config)
            # Share embedding with TinyLlama
            self.structural_encoder.embedding = self.shared_embedding
            # V8 FIX #5: Eagerly initialize input_projection to avoid DDP race
            self._init_encoder_projection(self.structural_encoder, "Structural Encoder")
            # Move to same device as base model
            self.structural_encoder = self.structural_encoder.to(self.device)
            print(f"  - Num slots: {self.structural_encoder.num_slots}")
            print(f"  - Abstraction layers at: {self.structural_encoder.injection_layers}")
        else:
            self.structural_encoder = None
            print("Structural Encoder DISABLED (ablation mode)")

        # 2. Content Encoder
        if self.config.model.content_encoder.enabled:
            print("Initializing Content Encoder...")
            self.content_encoder = ContentEncoder(self.config)
            # Share embedding with TinyLlama
            self.content_encoder.embedding = self.shared_embedding
            # V8 FIX #5: Eagerly initialize input_projection to avoid DDP race
            self._init_encoder_projection(self.content_encoder, "Content Encoder")
            # Move to same device as base model
            self.content_encoder = self.content_encoder.to(self.device)
            print(f"  - Num layers: {self.content_encoder.num_layers}")
            print(f"  - Pooling: {self.content_encoder.pooling}")
        else:
            self.content_encoder = None
            print("Content Encoder DISABLED (ablation mode)")

        # 3. Causal Binding Mechanism
        if self.config.model.causal_binding.enabled:
            print("Initializing Causal Binding Mechanism...")
            self.causal_binding = CausalBindingMechanism(self.config)
            # Move to same device as base model
            self.causal_binding = self.causal_binding.to(self.device)
            self.cbm_injection_layers = self.config.model.causal_binding.injection_layers
            print(f"  - Injection layers: {self.cbm_injection_layers}")
            print(f"  - Num heads: {self.causal_binding.num_heads}")
        else:
            self.causal_binding = None
            self.cbm_injection_layers = []
            print("Causal Binding Mechanism DISABLED (ablation mode)")

        # LOW #74: Log total model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n✓ Model initialization complete:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

    def _register_cbm_hooks(self):
        """
        Register forward hooks to inject CBM at specified decoder layers.

        The hooks intercept the hidden states at specific layers and apply
        the Causal Binding Mechanism to condition on structure+content.
        """
        if self.causal_binding is None or len(self.cbm_injection_layers) == 0:
            return

        # Get decoder layers from TinyLlama
        # TinyLlama uses: model.layers[0..21]
        decoder_layers = self.base_model.model.layers

        def make_hook(layer_idx):
            """Create a hook function for a specific layer."""
            def hook(module, input, output):
                # TinyLlama decoder layers return a plain tensor (not a tuple)
                # Output is always 3D: [batch_size, seq_len, d_model]
                hidden_states = output

                # Only apply CBM if we have structural and content representations
                if (self.current_structural_slots is not None and
                    self.current_content_repr is not None):

                    # Apply Causal Binding Mechanism
                    try:
                        # Ensure hidden_states is 3D
                        if hidden_states.dim() != 3:
                            print(f"Warning: Expected 3D hidden_states at layer {layer_idx}, got {hidden_states.dim()}D")
                            return output

                        batch_size, seq_len, d_model = hidden_states.shape
                        
                        # Device alignment: ensure CBM inputs are on same device as hidden states
                        target_device = hidden_states.device
                        structural_slots = self.current_structural_slots
                        content_repr = self.current_content_repr
                        edge_weights = self.current_edge_weights
                        
                        if structural_slots.device != target_device:
                            structural_slots = structural_slots.to(target_device)
                        if content_repr.device != target_device:
                            content_repr = content_repr.to(target_device)
                        if edge_weights is not None and edge_weights.device != target_device:
                            edge_weights = edge_weights.to(target_device)

                        # Step 1: Bind content to structural slots
                        bound_repr, _ = self.causal_binding.bind(
                            structural_slots=structural_slots,
                            content_repr=content_repr,
                            edge_weights=edge_weights,
                        )

                        # Step 2: Broadcast to sequence length
                        broadcast_repr = self.causal_binding.broadcast(
                            bound_slots=bound_repr,
                            seq_len=seq_len,
                        )

                        # Step 3: Inject into decoder hidden states
                        injected_hidden = self.causal_binding.inject(
                            decoder_hidden=hidden_states,
                            bound_repr=broadcast_repr,
                            layer_idx=layer_idx,
                        )

                        # Ensure output dtype matches input dtype (important for fp16)
                        if injected_hidden.dtype != hidden_states.dtype:
                            injected_hidden = injected_hidden.to(hidden_states.dtype)

                        # Return in the same format as received (plain tensor)
                        return injected_hidden

                    except Exception as e:
                        print(f"Warning: CBM injection failed at layer {layer_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        return output

                return output

            return hook

        # Register hooks at specified layers
        print("Registering CBM injection hooks...")
        for layer_idx in self.cbm_injection_layers:
            if layer_idx < len(decoder_layers):
                decoder_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                print(f"  - Registered hook at layer {layer_idx}")
            else:
                print(f"  - WARNING: Layer {layer_idx} exceeds num_layers={len(decoder_layers)}")

    def get_instruction_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract instruction mask from labels.

        CRITICAL: This prevents data leakage by ensuring SE and CE only see
        the instruction, not the response.

        Convention: labels has -100 for tokens not included in loss (instruction + padding)
        We need to distinguish instruction from padding using attention_mask.

        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] with -100 for instruction AND padding tokens
            attention_mask: [batch_size, seq_len] with 1 for valid tokens, 0 for padding

        Returns:
            instruction_mask: [batch_size, seq_len] with 1 for instruction, 0 for response/padding
        """
        # labels == -100 indicates instruction OR padding tokens
        # To get just instruction, we need: (labels == -100) AND (attention_mask == 1)
        is_masked = (labels == -100)
        
        if attention_mask is not None:
            # Exclude padding: instruction is where labels=-100 AND token is not padding
            instruction_mask = (is_masked & (attention_mask == 1)).long()
        else:
            # Fallback: assume no padding, so all -100 positions are instruction
            instruction_mask = is_masked.long()

        return instruction_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SCI components.

        Args:
            input_ids: [batch_size, seq_len] - Token IDs
            attention_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
            labels: [batch_size, seq_len] - Target labels with -100 for instruction
            instruction_mask: [batch_size, seq_len] - Optional explicit instruction mask (1=instruction, 0=response)
            return_dict: Whether to return dictionary
            output_hidden_states: Whether to return intermediate hidden states

        Returns:
            Dictionary containing:
                - logits: [batch_size, seq_len, vocab_size]
                - loss: Scalar LM loss (if labels provided)
                - structural_slots: [batch_size, num_slots, d_model] (if SE enabled)
                - content_repr: [batch_size, d_model] (if CE enabled)
                - structural_scores: List of structuralness scores (if SE enabled)
                - hidden_states: Decoder hidden states (if requested)
        """
        batch_size, seq_len = input_ids.shape

        # Move inputs to model device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if instruction_mask is not None:
            instruction_mask = instruction_mask.to(self.device)

        # ============================================================
        # STEP 1: Extract instruction mask (CRITICAL for no leakage)
        # ============================================================
        # BUG #90 FIX: Use explicit instruction_mask if provided,
        # otherwise derive from labels

        if instruction_mask is not None:
            # Use explicit instruction_mask from batch (preferred)
            pass  # Already set from parameter
        elif labels is not None:
            # Derive from labels, excluding padding using attention_mask
            instruction_mask = self.get_instruction_mask(input_ids, labels, attention_mask)
        else:
            # During inference, treat all tokens as instruction
            instruction_mask = attention_mask

        # Store for hooks
        self.current_instruction_mask = instruction_mask

        # ============================================================
        # STEP 2: Structural Encoder (SE)
        # ============================================================

        if self.structural_encoder is not None:
            # SE processes ONLY instruction tokens
            structural_slots, structural_scores = self.structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,  # CRITICAL
            )

            # Store for CBM injection
            self.current_structural_slots = structural_slots
            self.current_structural_scores = structural_scores
        else:
            structural_slots = None
            structural_scores = None
            self.current_structural_slots = None
            self.current_structural_scores = None

        # ============================================================
        # STEP 3: Content Encoder (CE)
        # ============================================================

        if self.content_encoder is not None:
            # CE processes ONLY instruction tokens
            content_repr = self.content_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,  # CRITICAL
            )

            # Store for CBM injection
            self.current_content_repr = content_repr
        else:
            content_repr = None
            self.current_content_repr = None

        # ============================================================
        # STEP 4: TinyLlama Decoder (with CBM injection via hooks)
        # ============================================================

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # ============================================================
        # STEP 5: Prepare outputs
        # ============================================================

        result = {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None,
            'structural_slots': structural_slots,
            'content_repr': content_repr,
            'structural_scores': structural_scores,
        }

        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states

        # Clear stored representations
        self.current_structural_slots = None
        self.current_content_repr = None
        self.current_structural_scores = None
        self.current_instruction_mask = None

        return result if return_dict else (result['logits'], result['loss'])

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate sequences using the SCI model.

        During generation, we treat the entire input as "instruction" since
        there is no response yet.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            max_length: Maximum generation length
            **kwargs: Additional generation arguments

        Returns:
            generated_ids: [batch_size, max_length]
        """
        # Extract structure and content from input
        batch_size, seq_len = input_ids.shape

        # For generation, treat all input as instruction
        instruction_mask = attention_mask

        # Encode structure and content
        if self.structural_encoder is not None:
            structural_slots, _ = self.structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )
            self.current_structural_slots = structural_slots

        if self.content_encoder is not None:
            content_repr = self.content_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )
            self.current_content_repr = content_repr

        # #70 FIX: Wrap in try/finally to clear representations even on exception
        try:
            # Generate using base model (CBM injected via hooks)
            generated_ids = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs,
            )
            return generated_ids
        finally:
            # Clear stored representations (always, even on exception)
            self.current_structural_slots = None
            self.current_content_repr = None

    def compute_orthogonality_loss(
        self,
        content_repr: torch.Tensor,
        structural_slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute orthogonality loss between content and structure.

        This ensures clean factorization: content ⊥ structure.

        Args:
            content_repr: [batch_size, d_model]
            structural_slots: [batch_size, num_slots, d_model]

        Returns:
            loss: Scalar orthogonality loss
        """
        # Pool structural slots
        structural_pooled = structural_slots.mean(dim=1)  # [batch_size, d_model]

        # Normalize both representations
        content_norm = F.normalize(content_repr, dim=-1)
        structure_norm = F.normalize(structural_pooled, dim=-1)

        # Cosine similarity (should be close to 0)
        cosine_sim = (content_norm * structure_norm).sum(dim=-1)  # [batch_size]

        # Loss is absolute cosine similarity
        loss = cosine_sim.abs().mean()

        return loss

    def get_structural_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Get structural representation for analysis/testing.

        Used for:
        - Testing structural invariance
        - Visualizing structure extraction
        - Computing structural similarity

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            structural_slots: [batch_size, num_slots, d_model]
            structural_scores: List of [batch_size, seq_len, d_model] scores
        """
        if self.structural_encoder is None:
            raise ValueError("Structural Encoder is not enabled")

        # Treat input as instruction
        instruction_mask = attention_mask

        structural_slots, structural_scores = self.structural_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            instruction_mask=instruction_mask,
        )

        return structural_slots, structural_scores

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        from dataclasses import asdict
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        import json
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            # Convert config to dict - use asdict for dataclasses
            if isinstance(self.config, dict):
                config_dict = self.config
            else:
                # SCIConfig is a dataclass, use asdict()
                config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2)

        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, config=None):
        """Load model from directory."""
        import os
        import json

        # Load config if not provided
        if config is None:
            config_path = os.path.join(load_directory, 'config.json')
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Convert dict to SCIConfig using _dict_to_config
            from sci.config.config_loader import _dict_to_config
            config = _dict_to_config(config_dict)

        # Initialize model
        model = cls(config)

        # Load state dict
        state_dict = torch.load(
            os.path.join(load_directory, 'pytorch_model.bin'),
            map_location='cpu'
        )
        model.load_state_dict(state_dict)

        print(f"Model loaded from {load_directory}")

        return model


if __name__ == "__main__":
    # Quick test
    print("Testing SCIModel...")

    from sci.config.config_loader import load_config

    # Load config
    config = load_config("configs/base_config.yaml")

    # Initialize model
    model = SCIModel(config)

    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    labels[:, :seq_len//2] = -100  # First half is instruction

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )

    print(f"✓ Logits shape: {outputs['logits'].shape}")
    print(f"✓ Loss: {outputs['loss'].item():.4f}")
    if outputs['structural_slots'] is not None:
        print(f"✓ Structural slots shape: {outputs['structural_slots'].shape}")
    if outputs['content_repr'] is not None:
        print(f"✓ Content repr shape: {outputs['content_repr'].shape}")

    print("\n✓ SCIModel test passed!")
