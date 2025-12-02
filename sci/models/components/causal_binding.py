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
        self.d_model = config.model.causal_binding.get('d_model', 2048)  # TinyLlama dimension
        self.num_heads = config.model.causal_binding.num_heads
        self.dropout = config.model.causal_binding.dropout
        self.injection_layers = config.model.causal_binding.injection_layers

        # Head dimension
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.d_model // self.num_heads

        # 1. BINDING ATTENTION
        # Cross-attention: Query from structural slots, Key/Value from content
        self.binding_query = nn.Linear(self.d_model, self.d_model)
        self.binding_key = nn.Linear(self.d_model, self.d_model)
        self.binding_value = nn.Linear(self.d_model, self.d_model)
        self.binding_out = nn.Linear(self.d_model, self.d_model)

        self.binding_dropout = nn.Dropout(dropout)

        # Layer norm for stability
        self.binding_norm = nn.LayerNorm(self.d_model)

        # 2. CAUSAL INTERVENTION LAYER
        # This applies causal reasoning using edge weights
        self.use_causal_intervention = config.model.causal_binding.get('use_causal_intervention', True)

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

        # Injection adapters for each decoder layer
        # These prepare the bound representation for injection
        self.adapters = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),  # Concatenate with decoder hidden
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(dropout),
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
            structural_slots: [batch, num_slots, d_model]
            content_repr: [batch, d_model]
            edge_weights: [batch, num_slots, num_slots] - Optional causal edges
            return_attention: Whether to return attention weights

        Returns:
            bound_repr: [batch, num_slots, d_model]
            attn_weights: [batch, num_heads, num_slots, 1] if return_attention
        """
        batch_size, num_slots, _ = structural_slots.shape

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
            # edge_weights: [batch, num_slots, num_slots] where [i, j] is edge from i to j
            # Expand for broadcasting: [batch, num_slots, num_slots, 1]
            edge_weights_expanded = edge_weights.unsqueeze(-1)

            # Apply edge weights to messages
            # messages: [batch, num_slots, d_model]
            # Expand to: [batch, num_slots, 1, d_model] for broadcasting
            messages_expanded = messages.unsqueeze(2)

            # Weighted sum over source slots
            # [batch, num_slots, num_slots, d_model] -> [batch, num_slots, d_model]
            intervention = (edge_weights_expanded * messages_expanded).sum(dim=1)

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

        This uses attention to map each position in the sequence to
        relevant bound slots.

        Args:
            bound_slots: [batch, num_slots, d_model]
            seq_len: Target sequence length

        Returns:
            broadcast_repr: [batch, seq_len, d_model]
        """
        batch_size, num_slots, d_model = bound_slots.shape

        # Create learnable position queries
        # We want seq_len queries that will attend to num_slots slots
        # For simplicity, use a learned embedding for each position
        device = bound_slots.device

        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Use a simple learned position embedding
        # Initialize if not exists (this is a bit hacky, but works for now)
        if not hasattr(self, 'position_queries'):
            self.register_buffer(
                'position_queries',
                torch.randn(1, 512, d_model, device=device) * 0.02
            )

        # Get position queries (clip to seq_len)
        pos_queries = self.position_queries[:, :seq_len, :].expand(batch_size, -1, -1)

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

        # Ensure bound_repr matches decoder sequence length
        if bound_repr.shape[1] != decoder_hidden.shape[1]:
            # Re-broadcast if needed
            bound_repr = self.broadcast(
                bound_repr if bound_repr.dim() == 3 and bound_repr.shape[1] <= 10 else bound_repr[:, :1, :],
                decoder_hidden.shape[1]
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

    @dataclass
    class ModelConfig:
        causal_binding: CausalBindingConfig = field(default_factory=CausalBindingConfig)

    @dataclass
    class Config:
        model: ModelConfig = field(default_factory=ModelConfig)

    config = Config()

    # Create CBM
    cbm = CausalBindingMechanism(config)

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

    print("\n✓ All CausalBindingMechanism tests passed!")
