"""
Structural Encoder: Extracts structural patterns from sequences.

The Structural Encoder uses AbstractionLayer to suppress content information
while preserving structural patterns. It pools variable-length sequences into
a fixed number of structural slots using Slot Attention.

Key components:
1. Shared embeddings with TinyLlama
2. Rotary positional encoding
3. AbstractionLayer injection at layers [3, 6, 9]
4. Transformer encoder (12 layers)
5. Slot Attention pooling (8 slots)

LOW #81: Slot Attention Citation
---------------------------------
Slot Attention is based on:
    Locatello et al., "Object-Centric Learning with Slot Attention"
    NeurIPS 2020
    https://arxiv.org/abs/2006.15055

We use slot attention to pool variable-length sequences into a fixed number
of structural slots, enabling permutation-invariant structural representations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from sci.models.components.abstraction_layer import AbstractionLayer
from sci.models.components.positional_encoding import RotaryPositionalEncoding
from sci.models.components.slot_attention import SlotAttention


class StructuralEncoder(nn.Module):
    """
    Structural Encoder: Encodes structural patterns invariant to content.

    This encoder learns to represent the abstract structural pattern of an input
    sequence (e.g., "X twice" pattern) while being invariant to the specific
    content that fills the pattern (e.g., "walk" vs "run").

    Args:
        config: Configuration object with structural encoder settings

    Input:
        input_ids: [batch_size, seq_len] - Token IDs (from shared embedding)
        attention_mask: [batch_size, seq_len] - 1 for valid, 0 for padding
        instruction_mask: [batch_size, seq_len] - 1 for instruction, 0 for response
            (CRITICAL for preventing data leakage)

    Output:
        structural_slots: [batch_size, num_slots, d_model] - Structural representations
        structural_scores: List of [batch_size, seq_len, d_model] - Structuralness scores
            from each AbstractionLayer
    """

    def __init__(self, config):
        super().__init__()

        # Configuration
        self.d_model = config.model.structural_encoder.d_model
        self.num_layers = config.model.structural_encoder.num_layers
        self.num_slots = config.model.structural_encoder.num_slots
        self.num_heads = config.model.structural_encoder.num_heads
        self.dim_feedforward = config.model.structural_encoder.dim_feedforward
        self.dropout = config.model.structural_encoder.dropout

        # AbstractionLayer injection configuration
        self.injection_layers = config.model.structural_encoder.abstraction_layer.injection_layers
        # Alias for tests
        self.abstraction_layer_positions = self.injection_layers

        # Shared embedding (will be set from TinyLlama)
        self.embedding = None  # Set externally

        # Positional encoding (RoPE for length generalization)
        self.pos_encoding = RotaryPositionalEncoding(
            d_model=self.d_model,
            max_length=config.model.position_encoding.max_length,
            base=config.model.position_encoding.base,
        )

        # Input projection (from embedding dim to encoder d_model)
        # Default to identity projection, will be replaced if dimensions differ
        self.embedding_dim = self.d_model
        self.input_projection = nn.Identity()
        self._projection_initialized = False

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for stability
            )
            for _ in range(self.num_layers)
        ])

        # Alias for compatibility with tests
        self.encoder = self.layers

        # AbstractionLayer modules at specific depths
        # CRITICAL: These are THE KEY INNOVATION
        self.abstraction_layers = nn.ModuleDict({
            str(layer_idx): AbstractionLayer(
                d_model=self.d_model,
                hidden_multiplier=config.model.structural_encoder.abstraction_layer.hidden_multiplier,
                residual_init=config.model.structural_encoder.abstraction_layer.residual_init,
                dropout=config.model.structural_encoder.abstraction_layer.dropout,
            )
            for layer_idx in self.injection_layers
        })

        # Slot Attention for pooling to fixed representation
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            d_model=self.d_model,
            num_iterations=3,  # Iterative refinement
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(self.d_model)

        # CRITICAL ASSERTION: Verify AbstractionLayer exists
        assert len(self.abstraction_layers) > 0, "CRITICAL: AbstractionLayer missing!"
        for layer_idx in self.injection_layers:
            assert str(layer_idx) in self.abstraction_layers, \
                f"CRITICAL: AbstractionLayer missing at layer {layer_idx}!"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of Structural Encoder.

        CRITICAL: If instruction_mask is provided, the encoder will ONLY
        attend to instruction tokens, preventing data leakage from response tokens.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] - General attention mask
            instruction_mask: [batch_size, seq_len] - Instruction vs response mask

        Returns:
            structural_slots: [batch_size, num_slots, d_model]
            structural_scores_all: List of score tensors from each AbstractionLayer
        """
        # HIGH #38: Add tensor size checks
        assert input_ids.dim() == 2, \
            f"input_ids must be 2D [batch, seq_len], got {input_ids.dim()}D"

        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape, \
                f"attention_mask shape {attention_mask.shape} != input_ids shape {input_ids.shape}"

        if instruction_mask is not None:
            assert instruction_mask.shape == input_ids.shape, \
                f"instruction_mask shape {instruction_mask.shape} != input_ids shape {input_ids.shape}"

        batch_size, seq_len = input_ids.shape

        # Get embeddings
        # CRITICAL: Use shared embeddings
        assert self.embedding is not None, "CRITICAL: Shared embedding not set!"
        hidden_states = self.embedding(input_ids)  # [batch, seq_len, embedding_dim]

        # Create input projection if dimensions differ and not yet initialized
        if not self._projection_initialized:
            self.embedding_dim = hidden_states.shape[-1]
            if self.embedding_dim != self.d_model:
                # Get model's device (in case it was moved after __init__)
                model_device = next(self.parameters()).device
                self.input_projection = nn.Linear(self.embedding_dim, self.d_model).to(
                    device=model_device,
                    dtype=hidden_states.dtype
                )
                print(f"Created input projection: {self.embedding_dim} -> {self.d_model} on {model_device}")
            self._projection_initialized = True

        # Project to encoder dimension
        hidden_states = self.input_projection(hidden_states)  # [batch, seq_len, d_model]

        # Apply rotary positional encoding
        hidden_states = self.pos_encoding(hidden_states)

        # CRITICAL: Apply instruction mask to prevent data leakage
        if instruction_mask is not None:
            # Zero out response tokens
            # instruction_mask: 1 for instruction, 0 for response
            mask_expanded = instruction_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            hidden_states = hidden_states * mask_expanded

        # Create attention mask for transformer
        # Combine general mask and instruction mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        if instruction_mask is not None:
            # Only attend to instruction tokens
            attention_mask = attention_mask * instruction_mask.float()

        # Convert to transformer format (0 = attend, -inf = ignore)
        # Transformer expects [batch, seq_len] with True for masked positions
        transformer_mask = (attention_mask == 0)

        # Collect structural scores from each AbstractionLayer
        structural_scores_all = []

        # Pass through transformer layers with AbstractionLayer injection
        for i, layer in enumerate(self.layers):
            # Apply transformer layer
            hidden_states = layer(hidden_states, src_key_padding_mask=transformer_mask)

            # CRITICAL: Re-apply instruction mask after each layer to prevent leakage
            # This ensures response tokens remain zeroed even after residual connections
            if instruction_mask is not None:
                mask_expanded = instruction_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask_expanded

            # Inject AbstractionLayer at specific layers
            if str(i) in self.abstraction_layers:
                hidden_states, scores = self.abstraction_layers[str(i)](
                    hidden_states, attention_mask
                )
                structural_scores_all.append(scores)

                # Re-apply mask after abstraction layer as well
                if instruction_mask is not None:
                    mask_expanded = instruction_mask.unsqueeze(-1).float()
                    hidden_states = hidden_states * mask_expanded

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # CRITICAL: Re-apply mask after final norm
        if instruction_mask is not None:
            mask_expanded = instruction_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask_expanded

        # Pool to structural slots using Slot Attention
        structural_slots = self.slot_attention(hidden_states, attention_mask)

        return structural_slots, structural_scores_all

    def get_structural_statistics(
        self,
        structural_scores_all: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute statistics about structural representations.

        Useful for monitoring training and debugging.
        """
        if not structural_scores_all:
            return {}

        # Average scores across all AbstractionLayer instances
        all_scores = torch.stack(structural_scores_all, dim=0)  # [num_layers, batch, seq, d_model]
        avg_scores = all_scores.mean(dim=0)  # [batch, seq, d_model]

        # Get statistics from the last AbstractionLayer
        last_scores = structural_scores_all[-1]

        stats = {}

        # Overall statistics
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            valid_scores = avg_scores * mask_expanded
            num_valid = mask_expanded.sum()

            stats["mean_score"] = (valid_scores.sum() / num_valid).item()
            stats["high_structural_ratio"] = ((valid_scores > 0.7).float().sum() / num_valid).item()
            stats["low_structural_ratio"] = ((valid_scores < 0.3).float().sum() / num_valid).item()
        else:
            stats["mean_score"] = avg_scores.mean().item()
            stats["high_structural_ratio"] = (avg_scores > 0.7).float().mean().item()
            stats["low_structural_ratio"] = (avg_scores < 0.3).float().mean().item()

        # Per-layer statistics
        for i, (layer_idx, scores) in enumerate(zip(self.injection_layers, structural_scores_all)):
            layer_mean = scores.mean().item()
            stats[f"layer_{layer_idx}_mean_score"] = layer_mean

        return stats


if __name__ == "__main__":
    # Test Structural Encoder
    print("Testing StructuralEncoder...")

    # Create minimal config
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class AbstractionLayerConfig:
        hidden_multiplier: int = 2
        residual_init: float = 0.1
        dropout: float = 0.1
        temperature: float = 0.1
        injection_layers: List[int] = field(default_factory=lambda: [3, 6, 9])

    @dataclass
    class StructuralEncoderConfig:
        d_model: int = 512
        num_layers: int = 12
        num_slots: int = 8
        num_heads: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.1
        abstraction_layer: AbstractionLayerConfig = field(default_factory=AbstractionLayerConfig)

    @dataclass
    class PositionEncodingConfig:
        type: str = "rotary"
        max_length: int = 512
        base: int = 10000

    @dataclass
    class ModelConfig:
        structural_encoder: StructuralEncoderConfig = field(default_factory=StructuralEncoderConfig)
        position_encoding: PositionEncodingConfig = field(default_factory=PositionEncodingConfig)

    @dataclass
    class Config:
        model: ModelConfig = field(default_factory=ModelConfig)

    config = Config()

    # Create encoder
    encoder = StructuralEncoder(config)

    # Create dummy shared embedding
    vocab_size = 1000
    encoder.embedding = nn.Embedding(vocab_size, config.model.structural_encoder.d_model)

    # Test forward pass
    batch_size = 4
    seq_len = 20

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Test without instruction mask
    structural_slots, structural_scores = encoder(input_ids, attention_mask)

    # Check outputs
    expected_shape = (batch_size, config.model.structural_encoder.num_slots, config.model.structural_encoder.d_model)
    assert structural_slots.shape == expected_shape, \
        f"Structural slots shape mismatch: {structural_slots.shape} vs {expected_shape}"

    assert len(structural_scores) == len(config.model.structural_encoder.abstraction_layer.injection_layers), \
        f"Wrong number of structural scores: {len(structural_scores)}"

    print(f"✓ Structural slots shape: {structural_slots.shape}")
    print(f"✓ Number of AbstractionLayer outputs: {len(structural_scores)}")

    # Test with instruction mask (data leakage prevention)
    instruction_mask = torch.ones(batch_size, seq_len)
    instruction_mask[:, 10:] = 0  # First 10 tokens are instruction, rest is response

    structural_slots_masked, _ = encoder(input_ids, attention_mask, instruction_mask)

    # Verify that changing response tokens doesn't affect output
    input_ids_modified = input_ids.clone()
    input_ids_modified[:, 10:] = torch.randint(0, vocab_size, (batch_size, seq_len - 10))

    structural_slots_modified, _ = encoder(input_ids_modified, attention_mask, instruction_mask)

    # Outputs should be identical because response is masked
    diff = (structural_slots_masked - structural_slots_modified).abs().max().item()
    assert diff < 1e-5, f"CRITICAL: Encoder sees response tokens! Diff: {diff}"
    print(f"✓ Data leakage prevention verified (diff: {diff:.2e})")

    # Test statistics
    stats = encoder.get_structural_statistics(structural_scores, attention_mask)
    print(f"✓ Statistics: mean_score={stats['mean_score']:.3f}, " +
          f"high_ratio={stats['high_structural_ratio']:.3f}")

    print("\n✓ All StructuralEncoder tests passed!")
