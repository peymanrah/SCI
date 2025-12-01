"""
Content Encoder: Extracts content information independently of structure.

The Content Encoder learns to represent the semantic content of the input
(e.g., "walk", "run", "jump") while being orthogonal to structural patterns.

Key features:
1. Shares embeddings with TinyLlama and Structural Encoder
2. Lightweight (2 layers) since it refines shared embeddings
3. Enforces orthogonality to structural representations
4. Mean pooling for sequence-level content representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sci.models.components.positional_encoding import RotaryPositionalEncoding


class ContentEncoder(nn.Module):
    """
    Content Encoder: Encodes content independently of structure.

    This encoder learns content representations that are orthogonal to
    structural representations, ensuring clean factorization.

    Args:
        config: Configuration object with content encoder settings

    Input:
        input_ids: [batch_size, seq_len] - Token IDs
        attention_mask: [batch_size, seq_len] - Attention mask
        instruction_mask: [batch_size, seq_len] - Instruction vs response mask

    Output:
        content_repr: [batch_size, d_model] - Content representation
    """

    def __init__(self, config):
        super().__init__()

        # Configuration
        self.d_model = config.model.content_encoder.d_model
        self.num_layers = config.model.content_encoder.num_layers
        self.num_heads = config.model.content_encoder.num_heads
        self.dim_feedforward = config.model.content_encoder.dim_feedforward
        self.dropout = config.model.content_encoder.dropout
        self.pooling = config.model.content_encoder.pooling

        # Shared embedding (will be set from TinyLlama)
        # CRITICAL: Must share with Structural Encoder
        self.embedding = None  # Set externally

        # Positional encoding (RoPE for consistency)
        self.pos_encoding = RotaryPositionalEncoding(
            d_model=self.d_model,
            max_length=config.model.position_encoding.max_length,
            base=config.model.position_encoding.base,
        )

        # Input projection
        self.input_projection = nn.Linear(self.d_model, self.d_model)

        # Lightweight transformer layers (only 2 layers)
        # REASONING: Content is already well-represented in shared embeddings
        # We just need to refine it
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.num_layers)
        ])

        # Pooling layer
        if self.pooling == "mean":
            self.pool = self._mean_pooling
        elif self.pooling == "max":
            self.pool = self._max_pooling
        else:
            self.pool = self._mean_pooling

        # Final normalization
        self.final_norm = nn.LayerNorm(self.d_model)

        # Orthogonal projection (optional, can also use loss-based orthogonality)
        # This projects content to be orthogonal to structure
        self.use_orthogonal_projection = config.model.content_encoder.get("use_orthogonal_projection", False)
        if self.use_orthogonal_projection:
            self.orthogonal_proj = nn.Linear(self.d_model, self.d_model)

    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling over valid tokens."""
        # Expand mask to match hidden dimension
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]

        # Sum over sequence dimension
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # [batch, d_model]

        # Divide by number of valid tokens
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
        pooled = sum_hidden / sum_mask  # [batch, d_model]

        return pooled

    def _max_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max pooling over valid tokens."""
        # Mask invalid positions with -inf
        mask_expanded = attention_mask.unsqueeze(-1).float()
        hidden_masked = hidden_states.clone()
        hidden_masked[mask_expanded == 0] = float('-inf')

        # Max over sequence dimension
        pooled, _ = hidden_masked.max(dim=1)  # [batch, d_model]

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of Content Encoder.

        CRITICAL: If instruction_mask is provided, only instruction tokens
        are encoded to prevent data leakage.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            instruction_mask: [batch_size, seq_len]

        Returns:
            content_repr: [batch_size, d_model]
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        # CRITICAL: Use shared embeddings
        assert self.embedding is not None, "CRITICAL: Shared embedding not set!"
        hidden_states = self.embedding(input_ids)

        # Project to encoder dimension
        hidden_states = self.input_projection(hidden_states)

        # Apply rotary positional encoding
        hidden_states = self.pos_encoding(hidden_states)

        # CRITICAL: Apply instruction mask to prevent data leakage
        if instruction_mask is not None:
            mask_expanded = instruction_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask_expanded

        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        if instruction_mask is not None:
            attention_mask = attention_mask * instruction_mask.float()

        # Convert to transformer format
        transformer_mask = (attention_mask == 0)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=transformer_mask)

        # Pool to sequence-level representation
        content_repr = self.pool(hidden_states, attention_mask)

        # Apply orthogonal projection if enabled
        if self.use_orthogonal_projection:
            content_repr = self.orthogonal_proj(content_repr)

        # Final normalization
        content_repr = self.final_norm(content_repr)

        return content_repr


def compute_orthogonality_loss(
    structural_repr: torch.Tensor,
    content_repr: torch.Tensor,
    lambda_ortho: float = 0.1,
) -> torch.Tensor:
    """
    Compute orthogonality loss between structural and content representations.

    This loss encourages structural and content representations to be orthogonal,
    ensuring clean factorization.

    Args:
        structural_repr: [batch_size, num_slots, d_model] or [batch_size, d_model]
        content_repr: [batch_size, d_model]
        lambda_ortho: Loss weight

    Returns:
        loss: Scalar orthogonality loss
    """
    # If structural_repr has slots, pool to [batch, d_model]
    if structural_repr.dim() == 3:
        structural_repr = structural_repr.mean(dim=1)

    # Normalize representations
    structural_norm = F.normalize(structural_repr, dim=-1)
    content_norm = F.normalize(content_repr, dim=-1)

    # Compute inner product (should be close to 0 for orthogonal vectors)
    # Shape: [batch_size, batch_size] (pairwise dot products)
    inner_product = torch.matmul(structural_norm, content_norm.T)

    # Frobenius norm squared (sum of squared elements)
    ortho_loss = torch.norm(inner_product, p='fro') ** 2

    # Normalize by batch size
    ortho_loss = ortho_loss / structural_repr.shape[0]

    return lambda_ortho * ortho_loss


if __name__ == "__main__":
    # Test Content Encoder
    print("Testing ContentEncoder...")

    from dataclasses import dataclass, field

    @dataclass
    class ContentEncoderConfig:
        d_model: int = 512
        num_layers: int = 2
        num_heads: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.1
        pooling: str = "mean"

    @dataclass
    class PositionEncodingConfig:
        type: str = "rotary"
        max_length: int = 512
        base: int = 10000

    @dataclass
    class ModelConfig:
        content_encoder: ContentEncoderConfig = field(default_factory=ContentEncoderConfig)
        position_encoding: PositionEncodingConfig = field(default_factory=PositionEncodingConfig)

    @dataclass
    class Config:
        model: ModelConfig = field(default_factory=ModelConfig)

    config = Config()

    # Create encoder
    encoder = ContentEncoder(config)

    # Create dummy shared embedding
    vocab_size = 1000
    encoder.embedding = nn.Embedding(vocab_size, config.model.content_encoder.d_model)

    # Test forward pass
    batch_size = 4
    seq_len = 20

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Test without instruction mask
    content_repr = encoder(input_ids, attention_mask)

    # Check output shape
    expected_shape = (batch_size, config.model.content_encoder.d_model)
    assert content_repr.shape == expected_shape, \
        f"Content repr shape mismatch: {content_repr.shape} vs {expected_shape}"

    print(f"✓ Content representation shape: {content_repr.shape}")

    # Test with instruction mask (data leakage prevention)
    instruction_mask = torch.ones(batch_size, seq_len)
    instruction_mask[:, 10:] = 0  # First 10 tokens are instruction

    content_repr_masked = encoder(input_ids, attention_mask, instruction_mask)

    # Verify that changing response tokens doesn't affect output
    input_ids_modified = input_ids.clone()
    input_ids_modified[:, 10:] = torch.randint(0, vocab_size, (batch_size, seq_len - 10))

    content_repr_modified = encoder(input_ids_modified, attention_mask, instruction_mask)

    diff = (content_repr_masked - content_repr_modified).abs().max().item()
    assert diff < 1e-5, f"CRITICAL: Content encoder sees response tokens! Diff: {diff}"
    print(f"✓ Data leakage prevention verified (diff: {diff:.2e})")

    # Test orthogonality loss
    structural_repr = torch.randn(batch_size, 8, 512)  # Dummy structural representation
    ortho_loss = compute_orthogonality_loss(structural_repr, content_repr)

    assert ortho_loss.dim() == 0, "Orthogonality loss should be scalar"
    print(f"✓ Orthogonality loss computed: {ortho_loss.item():.6f}")

    # Test that orthogonal representations give lower loss
    structural_flat = structural_repr.mean(dim=1)
    # Create orthogonal content
    content_orthogonal = content_repr - (content_repr * F.normalize(structural_flat, dim=-1)).sum(dim=-1, keepdim=True) * F.normalize(structural_flat, dim=-1)
    content_orthogonal = F.normalize(content_orthogonal, dim=-1)

    ortho_loss_low = compute_orthogonality_loss(structural_repr, content_orthogonal, lambda_ortho=0.1)
    print(f"✓ Orthogonal content loss (should be lower): {ortho_loss_low.item():.6f}")

    assert ortho_loss_low < ortho_loss, "Orthogonal vectors should have lower loss"

    print("\n✓ All ContentEncoder tests passed!")
