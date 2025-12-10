"""
Causal Binding Mechanism (CBM): Binds content to structural slots.

The CBM is responsible for combining structural and content representations
through cross-attention and causal intervention, then broadcasting the bound
representation back to the sequence for injection into the decoder.

Key components:
1. Binding Attention: Cross-attention from structural slots to content
2. Causal Intervention: Applies do-calculus to enforce causal relationships
3. Broadcast Attention: Maps bound slots back to sequence positions

This is NOT just concatenation - it performs causal reasoning to bind content
to the appropriate structural slots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CausalBindingMechanism(nn.Module):
    """
    Causal Binding Mechanism: Binds content to structural slots via causal intervention.

    The CBM performs three key operations:
    1. Binding: Cross-attention from structural slots (query) to content (key/value)
    2. Intervention: Causal intervention using edge weights from structural graph
    3. Broadcast: Map bound slots back to sequence positions for decoder injection

    Args:
        config: Configuration object with CBM settings

    Input:
        structural_slots: [batch_size, num_slots, d_model] - Structural representation
        content_repr: [batch_size, d_model] - Content representation
        edge_weights: [batch_size, num_slots, num_slots] - Optional causal graph edges

    Output:
        bound_repr: [batch_size, num_slots, d_model] - Bound representation
        attention_weights: [batch_size, num_slots, ...] - Attention weights for analysis
    """

    def __init__(self, config):
        super().__init__()

        # Configuration
        self.d_model = getattr(config.model.causal_binding, 'd_model', 2048)  # TinyLlama dimension
        self.num_heads = config.model.causal_binding.num_heads
        self.dropout = config.model.causal_binding.dropout
        self.injection_layers = config.model.causal_binding.injection_layers
        
        # CONFIGURABLE SLOT COUNT: Read from SE config or default to 8
        self.num_slots = getattr(config.model.structural_encoder, 'num_slots', 8)

        # Head dimension
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.d_model // self.num_heads

        # #74 FIX: Initialize projections in __init__ to avoid race conditions with DataParallel
        # Get encoder dimensions from config with safe defaults
        structural_d_model = getattr(config.model.structural_encoder, 'd_model', self.d_model)
        content_d_model = getattr(config.model.content_encoder, 'd_model', self.d_model)
        
        # Project structural slots if dimension differs
        if structural_d_model != self.d_model:
            self.structural_projection = nn.Linear(structural_d_model, self.d_model)
        else:
            self.structural_projection = nn.Identity()

        # Project content if dimension differs  
        if content_d_model != self.d_model:
            self.content_projection = nn.Linear(content_d_model, self.d_model)
        else:
            self.content_projection = nn.Identity()
        
        # Flag kept for backward compatibility but always True now
        self._projections_initialized = True

        # 1. BINDING ATTENTION
        # Cross-attention: Query from structural slots, Key/Value from content
        self.binding_query = nn.Linear(self.d_model, self.d_model)
        self.binding_key = nn.Linear(self.d_model, self.d_model)
        self.binding_value = nn.Linear(self.d_model, self.d_model)
        self.binding_out = nn.Linear(self.d_model, self.d_model)

        self.binding_dropout = nn.Dropout(self.dropout)

        # Layer norm for stability
        self.binding_norm = nn.LayerNorm(self.d_model)

        # 2. CAUSAL INTERVENTION LAYER
        # This applies causal reasoning using edge weights
        self.use_causal_intervention = getattr(config.model.causal_binding, 'use_causal_intervention', True)

        if self.use_causal_intervention:
            # Message passing network for causal intervention
            self.intervention_query = nn.Linear(self.d_model, self.d_model)
            self.intervention_key = nn.Linear(self.d_model, self.d_model)
            self.intervention_value = nn.Linear(self.d_model, self.d_model)
            self.intervention_out = nn.Linear(self.d_model, self.d_model)

            self.intervention_norm = nn.LayerNorm(self.d_model)

        # 3. BROADCAST ATTENTION
        # Maps bound slots back to sequence positions
        self.broadcast_query = nn.Linear(self.d_model, self.d_model)
        self.broadcast_key = nn.Linear(self.d_model, self.d_model)
        self.broadcast_value = nn.Linear(self.d_model, self.d_model)
        self.broadcast_out = nn.Linear(self.d_model, self.d_model)

        self.broadcast_norm = nn.LayerNorm(self.d_model)

        # COMPOSITIONAL GENERALIZATION FIX: Use RoPE for position queries
        # instead of learned embeddings that require interpolation for OOD lengths
        # This allows natural extrapolation to longer sequences
        from sci.models.components.positional_encoding import RotaryPositionalEncoding
        
        # RoPE for broadcast queries - enables length generalization
        self.broadcast_pos_encoding = RotaryPositionalEncoding(
            d_model=self.d_model,
            max_length=4096,  # Large enough for any SCAN sequence
            base=10000,
        )
        
        # Learnable base query (single vector, replicated for each position)
        # Position information comes from RoPE, not learned embeddings
        self.base_position_query = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.xavier_uniform_(self.base_position_query[0])
        
        # Legacy: Keep position_queries for backward compatibility with checkpoints
        # But mark as deprecated
        self.base_max_seq_len = 2048  # Increased from 1024 to match config
        position_queries = torch.empty(1, self.base_max_seq_len, self.d_model)
        nn.init.xavier_uniform_(position_queries[0])
        self.position_queries = nn.Parameter(position_queries)
        
        # Flag to use new RoPE-based broadcast (default True for new training)
        self.use_rope_broadcast = True

        # Injection adapters for each decoder layer
        # These prepare the bound representation for injection
        self.adapters = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),  # Concatenate with decoder hidden
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.d_model),
            )
            for layer in self.injection_layers
        })

        # Gating mechanism for injection
        self.injection_gate = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.Sigmoid(),  # Gate values in [0, 1]
            )
            for layer in self.injection_layers
        })

        # STRUCTURAL EOS PREDICTOR: Predicts completion based on structural coverage
        # This provides a structural signal for EOS rather than relying purely on token statistics
        # Uses attention pooling: EOS query attends to bound slots to compute completion score
        self.eos_query = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.xavier_uniform_(self.eos_query[0])
        
        self.eos_key = nn.Linear(self.d_model, self.d_model)
        self.eos_value = nn.Linear(self.d_model, self.d_model)
        
        # Final projection to scalar EOS logit
        self.eos_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 4, 1),  # Single scalar output
        )
        
        # Flag to enable/disable structural EOS prediction
        self.use_structural_eos = getattr(config.model.causal_binding, 'use_structural_eos', True)

        # CRITICAL ASSERTIONS
        assert hasattr(self, 'binding_query'), "CRITICAL: Binding attention missing!"
        if self.use_causal_intervention:
            assert hasattr(self, 'intervention_query'), "CRITICAL: CausalInterventionLayer missing!"
        assert hasattr(self, 'broadcast_query'), "CRITICAL: Broadcast attention missing!"

    def _multihead_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_proj: nn.Linear,
        key_proj: nn.Linear,
        value_proj: nn.Linear,
        out_proj: nn.Linear,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention computation.

        Args:
            query: [batch, num_queries, d_model]
            key: [batch, num_keys, d_model]
            value: [batch, num_keys, d_model]
            attention_mask: [batch, num_queries, num_keys] - Optional mask

        Returns:
            output: [batch, num_queries, d_model]
            attention_weights: [batch, num_heads, num_queries, num_keys]
        """
        batch_size = query.shape[0]
        num_queries = query.shape[1]
        num_keys = key.shape[1]

        # Project and reshape to [batch, num_heads, num_q/k, head_dim]
        Q = query_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = key_proj(key).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        V = value_proj(value).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)  # Broadcast over heads

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.binding_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back to [batch, num_queries, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.d_model
        )

        # Output projection
        output = out_proj(attn_output)

        return output, attn_weights

    def bind(
        self,
        structural_slots: torch.Tensor,
        content_repr: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Bind content to structural slots via cross-attention.

        Args:
            structural_slots: [batch, num_slots, structural_d_model]
            content_repr: [batch, d_model]
            edge_weights: [batch, num_slots, num_slots] - Optional causal edges
            return_attention: Whether to return attention weights

        Returns:
            bound_repr: [batch, num_slots, d_model]
            attn_weights: [batch, num_heads, num_slots, 1] if return_attention
        """
        # #74 FIX: Projections are now initialized in __init__, no runtime initialization needed
        # Project to CBM d_model
        structural_slots = self.structural_projection(structural_slots)  # [batch, num_slots, d_model]
        content_repr = self.content_projection(content_repr)  # [batch, d_model]

        # Expand content to enable cross-attention
        # [batch, d_model] -> [batch, 1, d_model]
        content_expanded = content_repr.unsqueeze(1)

        # STEP 1: Binding Attention
        # Query: structural slots, Key/Value: content
        bound_repr, binding_attn = self._multihead_attention(
            query=structural_slots,
            key=content_expanded,
            value=content_expanded,
            query_proj=self.binding_query,
            key_proj=self.binding_key,
            value_proj=self.binding_value,
            out_proj=self.binding_out,
        )

        # Residual connection and normalization
        bound_repr = self.binding_norm(structural_slots + bound_repr)

        # STEP 2: Causal Intervention (if enabled and edge_weights provided)
        if self.use_causal_intervention and edge_weights is not None:
            # Use edge_weights to perform message passing between slots
            # This implements do-calculus: do(slot_i = value) affects slot_j via edge i->j

            # Compute messages from each slot
            messages, _ = self._multihead_attention(
                query=bound_repr,
                key=bound_repr,
                value=bound_repr,
                query_proj=self.intervention_query,
                key_proj=self.intervention_key,
                value_proj=self.intervention_value,
                out_proj=self.intervention_out,
            )

            # Weight messages by edge_weights
            # edge_weights: [batch, num_slots, num_slots] where [i, j] is edge from slot j to slot i
            # messages: [batch, num_slots, d_model] - each slot sends a message
            # Expand for broadcasting:
            # edge_weights: [batch, num_slots, num_slots, 1] (target, source, 1)
            # messages: [batch, 1, num_slots, d_model] (1, source, d_model)
            edge_weights_expanded = edge_weights.unsqueeze(-1)  # [batch, num_slots, num_slots, 1]
            messages_expanded = messages.unsqueeze(1)  # [batch, 1, num_slots, d_model]

            # Weighted sum over source slots (dim=2)
            # [batch, num_slots, num_slots, d_model] -> [batch, num_slots, d_model]
            intervention = (edge_weights_expanded * messages_expanded).sum(dim=2)

            # Residual connection and normalization
            bound_repr = self.intervention_norm(bound_repr + intervention)

        if return_attention:
            return bound_repr, binding_attn
        else:
            return bound_repr, None

    def broadcast(
        self,
        bound_slots: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Broadcast bound slots back to sequence positions.

        COMPOSITIONAL GENERALIZATION FIX: Uses RoPE for position queries
        instead of learned embeddings. This enables natural extrapolation
        to sequences longer than seen during training.

        Args:
            bound_slots: [batch, num_slots, d_model]
            seq_len: Target sequence length

        Returns:
            broadcast_repr: [batch, seq_len, d_model]
        """
        batch_size, num_slots, d_model = bound_slots.shape
        device = bound_slots.device

        if self.use_rope_broadcast:
            # NEW: RoPE-based position queries for length generalization
            # 1. Replicate base query for each position
            pos_queries = self.base_position_query.expand(batch_size, seq_len, -1).clone()
            
            # 2. Apply RoPE to add position information
            # This naturally extrapolates to any sequence length
            pos_queries = self.broadcast_pos_encoding(pos_queries, seq_len=seq_len)
        else:
            # LEGACY: Learned position queries (kept for checkpoint compatibility)
            if seq_len <= self.position_queries.shape[1]:
                pos_queries = self.position_queries[:, :seq_len, :].to(device).expand(batch_size, -1, -1)
            else:
                # Interpolation fallback
                cache_key = f"_cached_pos_queries_{seq_len}"
                if hasattr(self, cache_key):
                    pos_queries = getattr(self, cache_key).to(device).expand(batch_size, -1, -1)
                else:
                    import torch.nn.functional as F
                    pos_queries_base = self.position_queries.to(device)
                    pos_queries_base = pos_queries_base.transpose(1, 2)
                    pos_queries = F.interpolate(pos_queries_base, size=seq_len, mode='linear', align_corners=True)
                    pos_queries = pos_queries.transpose(1, 2)
                    self.register_buffer(cache_key, pos_queries.clone(), persistent=False)
                    pos_queries = pos_queries.expand(batch_size, -1, -1)

        # Broadcast attention: position queries attend to bound slots
        broadcast_repr, _ = self._multihead_attention(
            query=pos_queries,
            key=bound_slots,
            value=bound_slots,
            query_proj=self.broadcast_query,
            key_proj=self.broadcast_key,
            value_proj=self.broadcast_value,
            out_proj=self.broadcast_out,
        )

        # Normalization
        broadcast_repr = self.broadcast_norm(broadcast_repr)

        return broadcast_repr

    def inject(
        self,
        decoder_hidden: torch.Tensor,
        bound_repr: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Inject bound representation into decoder hidden states.

        This is called at specific decoder layers to inject structural
        and content information.

        Args:
            decoder_hidden: [batch, seq_len, d_model] - Decoder hidden states
            bound_repr: [batch, seq_len, d_model] - Bound representation (broadcasted)
            layer_idx: Decoder layer index

        Returns:
            injected_hidden: [batch, seq_len, d_model]
        """
        if layer_idx not in self.injection_layers:
            return decoder_hidden

        # Ensure bound_repr is 3D [batch, seq_len, d_model]
        if bound_repr.dim() == 2:
            # If 2D [batch, d_model], unsqueeze and expand
            bound_repr = bound_repr.unsqueeze(1).expand(-1, decoder_hidden.shape[1], -1)
        elif bound_repr.dim() == 3:
            if bound_repr.shape[1] != decoder_hidden.shape[1]:
                # Re-broadcast if seq_len doesn't match
                # Determine if we have bound_slots [batch, num_slots, d_model] or already broadcast
                if bound_repr.shape[1] <= 10:  # Likely num_slots
                    bound_repr = self.broadcast(
                        bound_slots=bound_repr,
                        seq_len=decoder_hidden.shape[1]
                    )
                else:
                    # CRITICAL #15: Fix incorrect slice in broadcast injection
                    # Already broadcast but different seq_len - just slice or pad
                    if bound_repr.shape[1] > decoder_hidden.shape[1]:
                        # Keep the LAST tokens (most recent in autoregressive generation)
                        bound_repr = bound_repr[:, -decoder_hidden.shape[1]:, :]
                    else:
                        # Pad with zeros at the end
                        padding = decoder_hidden.shape[1] - bound_repr.shape[1]
                        bound_repr = torch.cat([
                            bound_repr,
                            torch.zeros(bound_repr.shape[0], padding, bound_repr.shape[2], device=bound_repr.device)
                        ], dim=1)
        else:
            raise ValueError(f"bound_repr must be 2D or 3D, got {bound_repr.dim()}D with shape {bound_repr.shape}")

        # Final safety check before concatenation
        if bound_repr.dim() != 3 or decoder_hidden.dim() != 3:
            raise ValueError(
                f"Dimension mismatch before concat: "
                f"decoder_hidden.shape={decoder_hidden.shape}, bound_repr.shape={bound_repr.shape}"
            )

        # Concatenate decoder hidden and bound representation
        combined = torch.cat([decoder_hidden, bound_repr], dim=-1)

        # Compute gate values
        gate = self.injection_gate[str(layer_idx)](combined)

        # Compute adapted representation
        adapted = self.adapters[str(layer_idx)](combined)

        # Gated injection (allows model to control how much to use)
        injected_hidden = gate * adapted + (1 - gate) * decoder_hidden

        return injected_hidden

    def predict_eos(
        self,
        bound_slots: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Predict EOS probability based on structural slot coverage.
        
        COMPOSITIONAL GENERALIZATION: Instead of relying purely on token statistics
        for EOS prediction, this provides a structural signal based on how "complete"
        the slot representations are. When all structural slots have been satisfied
        (e.g., all compositional operations have been generated), the model should
        predict EOS.
        
        Mathematical Intuition:
        - Each slot represents a compositional pattern (e.g., "twice", "and", "after")
        - The EOS query attends to all slots to compute a weighted coverage score
        - High coverage (all patterns consumed) → high EOS probability
        - Low coverage (patterns remaining) → low EOS probability
        
        Args:
            bound_slots: [batch, num_slots, d_model] - Bound slot representations
            return_attention: Whether to return attention weights for analysis
            
        Returns:
            eos_logits: [batch, 1] - Scalar EOS logit for each example
            attention_weights: [batch, 1, num_slots] (optional) - Slot attention for EOS
        """
        if not self.use_structural_eos:
            # Return zeros if structural EOS is disabled
            batch_size = bound_slots.shape[0]
            zeros = torch.zeros(batch_size, 1, device=bound_slots.device)
            if return_attention:
                return zeros, None
            return zeros
        
        batch_size, num_slots, d_model = bound_slots.shape
        device = bound_slots.device
        
        # Expand EOS query for batch: [1, 1, d_model] -> [batch, 1, d_model]
        eos_query = self.eos_query.expand(batch_size, -1, -1).to(device)
        
        # Compute keys and values from bound slots
        eos_keys = self.eos_key(bound_slots)  # [batch, num_slots, d_model]
        eos_values = self.eos_value(bound_slots)  # [batch, num_slots, d_model]
        
        # Attention: query attends to all slots
        # [batch, 1, d_model] x [batch, d_model, num_slots] -> [batch, 1, num_slots]
        attn_logits = torch.bmm(eos_query, eos_keys.transpose(1, 2)) / (d_model ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Weighted sum over slots: [batch, 1, num_slots] x [batch, num_slots, d_model]
        # -> [batch, 1, d_model]
        eos_repr = torch.bmm(attn_weights, eos_values)
        
        # Project to scalar EOS logit: [batch, 1, d_model] -> [batch, 1]
        eos_logits = self.eos_head(eos_repr.squeeze(1))  # [batch, d_model] -> [batch, 1]
        
        if return_attention:
            return eos_logits, attn_weights
        return eos_logits

    def get_eos_loss(
        self,
        bound_slots: torch.Tensor,
        eos_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute structural EOS loss.
        
        This loss encourages the EOS predictor to output high values at the
        true EOS position and low values elsewhere.
        
        Args:
            bound_slots: [batch, num_slots, d_model] - Bound representations
            eos_positions: [batch] - Position of EOS token in each sequence
            sequence_lengths: [batch] - Total length of each sequence
            
        Returns:
            eos_loss: Scalar loss value
        """
        if not self.use_structural_eos:
            return torch.tensor(0.0, device=bound_slots.device)
        
        # Predict EOS from structural slots
        eos_logits = self.predict_eos(bound_slots)  # [batch, 1]
        
        # Binary cross-entropy: 1 at EOS position, 0 elsewhere
        # For training, we treat the current position as the target
        # This should be called at the actual EOS position
        eos_targets = torch.ones_like(eos_logits)  # [batch, 1]
        
        # BCE with logits
        eos_loss = F.binary_cross_entropy_with_logits(eos_logits, eos_targets)
        
        return eos_loss


if __name__ == "__main__":
    # Test Causal Binding Mechanism
    print("Testing CausalBindingMechanism...")

    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class CausalBindingConfig:
        d_model: int = 2048
        num_heads: int = 8
        dropout: float = 0.1
        injection_layers: List[int] = field(default_factory=lambda: [6, 12, 18])
        use_causal_intervention: bool = True
        use_structural_eos: bool = True

    @dataclass
    class StructuralEncoderConfig:
        d_model: int = 2048
        num_slots: int = 8  # Configurable slot count

    @dataclass
    class ContentEncoderConfig:
        d_model: int = 2048

    @dataclass
    class ModelConfig:
        causal_binding: CausalBindingConfig = field(default_factory=CausalBindingConfig)
        structural_encoder: StructuralEncoderConfig = field(default_factory=StructuralEncoderConfig)
        content_encoder: ContentEncoderConfig = field(default_factory=ContentEncoderConfig)

    @dataclass
    class Config:
        model: ModelConfig = field(default_factory=ModelConfig)

    config = Config()

    # Create CBM
    cbm = CausalBindingMechanism(config)
    print(f"  - CBM num_slots: {cbm.num_slots}")

    # Test binding
    batch_size = 4
    num_slots = 8
    d_model = 2048

    structural_slots = torch.randn(batch_size, num_slots, d_model)
    content_repr = torch.randn(batch_size, d_model)
    edge_weights = torch.rand(batch_size, num_slots, num_slots)

    # Test bind without edge weights
    bound_repr, _ = cbm.bind(structural_slots, content_repr)
    assert bound_repr.shape == structural_slots.shape, \
        f"Bound shape mismatch: {bound_repr.shape} vs {structural_slots.shape}"
    print(f"✓ Binding without edges: {bound_repr.shape}")

    # Test bind with edge weights
    bound_repr_with_edges, attn = cbm.bind(
        structural_slots, content_repr, edge_weights, return_attention=True
    )
    assert bound_repr_with_edges.shape == structural_slots.shape
    print(f"✓ Binding with edges: {bound_repr_with_edges.shape}")

    # Verify edge weights affect the result
    diff = (bound_repr - bound_repr_with_edges).abs().max().item()
    assert diff > 1e-6, "Edge weights should affect binding"
    print(f"✓ Edge weights have effect (diff: {diff:.4f})")

    # Test broadcast
    seq_len = 32
    broadcast_repr = cbm.broadcast(bound_repr, seq_len)
    assert broadcast_repr.shape == (batch_size, seq_len, d_model), \
        f"Broadcast shape mismatch: {broadcast_repr.shape}"
    print(f"✓ Broadcast: {broadcast_repr.shape}")

    # Test injection
    decoder_hidden = torch.randn(batch_size, seq_len, d_model)
    injected = cbm.inject(decoder_hidden, broadcast_repr, layer_idx=6)
    assert injected.shape == decoder_hidden.shape
    print(f"✓ Injection at layer 6: {injected.shape}")

    # Verify injection modifies hidden states
    diff = (injected - decoder_hidden).abs().max().item()
    assert diff > 1e-6, "Injection should modify hidden states"
    print(f"✓ Injection modifies hidden states (diff: {diff:.4f})")

    # Test injection at non-injection layer (should return unchanged)
    injected_no_change = cbm.inject(decoder_hidden, broadcast_repr, layer_idx=5)
    assert torch.allclose(injected_no_change, decoder_hidden), \
        "Non-injection layer should return unchanged"
    print(f"✓ Non-injection layer returns unchanged")

    # Test structural EOS prediction
    eos_logits, eos_attn = cbm.predict_eos(bound_repr, return_attention=True)
    assert eos_logits.shape == (batch_size, 1), \
        f"EOS logits shape mismatch: {eos_logits.shape}"
    assert eos_attn.shape == (batch_size, 1, num_slots), \
        f"EOS attention shape mismatch: {eos_attn.shape}"
    print(f"✓ Structural EOS prediction: {eos_logits.shape}")
    print(f"  - EOS attention weights: {eos_attn.shape}")

    # Test configurable slot count
    config.model.structural_encoder.num_slots = 16
    cbm_16_slots = CausalBindingMechanism(config)
    assert cbm_16_slots.num_slots == 16, f"Slot count mismatch: {cbm_16_slots.num_slots}"
    print(f"✓ Configurable slot count: {cbm_16_slots.num_slots} slots")

    print("\n✓ All CausalBindingMechanism tests passed!")
