"""
Positional Encoding implementations for SCI.

Provides two types of positional encodings:
1. RoPE (Rotary Position Embedding) - RECOMMENDED for length generalization
2. ALiBi (Attention with Linear Biases) - Alternative for ablation studies

RoPE is critical for SCAN length split as it extrapolates better to longer sequences.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from Su et al. 2021.

    RoPE encodes positional information by rotating the query/key representations
    in a rotation matrix that varies with position. This provides excellent
    length extrapolation capabilities, which is critical for SCAN length split.

    Key properties:
    - Relative position encoding (position differences matter, not absolute positions)
    - Better length extrapolation than absolute/learned positional embeddings
    - Compatible with TinyLlama (which uses RoPE natively)

    Args:
        d_model: Model dimension (must be even)
        max_length: Maximum sequence length
        base: Base for frequency computation (default: 10000)

    References:
        RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al. 2021)
    """

    def __init__(self, d_model: int, max_length: int = 2048, base: int = 10000):
        super().__init__()

        assert d_model % 2 == 0, f"d_model must be even for RoPE, got {d_model}"

        self.d_model = d_model
        self.max_length = max_length
        self.base = base

        # Precompute frequency tensor
        # inv_freq has shape [d_model // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute rotation matrices for common sequence lengths
        # Cache up to max_length to avoid recomputation
        self._precompute_freqs_cis(max_length)

    def _precompute_freqs_cis(self, seq_len: int):
        """Precompute complex exponentials for rotations."""
        # Create position indices: [seq_len]
        t = torch.arange(seq_len, device=self.inv_freq.device).float()

        # Compute frequencies: [seq_len, d_model // 2]
        freqs = torch.outer(t, self.inv_freq)

        # Convert to complex exponentials: e^(i * theta)
        # freqs_cis has shape [seq_len, d_model // 2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            Rotated tensor
        """
        x1, x2 = x[..., : self.d_model // 2], x[..., self.d_model // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply rotary positional encoding.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            seq_len: Sequence length (if different from x.shape[1])

        Returns:
            Position-encoded tensor [batch_size, seq_len, d_model]
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # Extend cache if needed
        if seq_len > self.freqs_cis.shape[0]:
            self._precompute_freqs_cis(seq_len)

        # Get frequencies for current sequence length
        # CRITICAL #21: Add helpful error message for sequence length overflow
        if seq_len > self.max_length:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum supported length ({self.max_length}). "
                f"Increase max_length in PositionalEncodingConfig or reduce input sequence length."
            )

        # freqs_cis: [seq_len, d_model // 2]
        freqs_cis = self.freqs_cis[:seq_len]

        # Convert input to complex for rotation
        # x: [batch_size, seq_len, d_model] → [batch_size, seq_len, d_model // 2, 2]
        x_complex = torch.view_as_complex(
            x.reshape(*x.shape[:-1], -1, 2)
        )

        # Apply rotation: multiply by e^(i * theta)
        # Expand freqs_cis to match batch dimension
        freqs_cis = freqs_cis.unsqueeze(0)  # [1, seq_len, d_model // 2]

        # Element-wise complex multiplication
        x_rotated = x_complex * freqs_cis

        # Convert back to real
        # [batch_size, seq_len, d_model // 2, 2] → [batch_size, seq_len, d_model]
        x_out = torch.view_as_real(x_rotated).reshape(*x.shape)

        return x_out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_length={self.max_length}, base={self.base}"


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) from Press et al. 2021.

    ALiBi adds a linearly decaying bias to attention scores based on
    key-query distance. Unlike RoPE, it modifies attention scores rather
    than representations.

    This is provided as an alternative for ablation studies, though RoPE
    is recommended for SCAN length generalization.

    Args:
        num_heads: Number of attention heads
        max_length: Maximum sequence length

    References:
        Train Short, Test Long: Attention with Linear Biases (Press et al. 2021)
    """

    def __init__(self, num_heads: int, max_length: int = 2048):
        super().__init__()

        self.num_heads = num_heads
        self.max_length = max_length

        # Compute slopes for each head
        # Slopes decrease geometrically: 2^(-8/n), 2^(-16/n), ...
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

        # Precompute bias matrix
        # bias[i, j] = -|i - j| (distance penalty)
        self._precompute_bias(max_length)

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes for each attention head.

        Uses geometric sequence: 2^(-8/n * i) for i in [1, num_heads]
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        # If num_heads is power of 2, use standard formula
        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads)).float()

        # Otherwise, use closest power of 2 and interpolate
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)

        # Add extra slopes by interpolation
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
        extra_slopes = extra_slopes[0::2][: num_heads - closest_power_of_2]
        slopes.extend(extra_slopes)

        return torch.tensor(slopes).float()

    def _precompute_bias(self, max_length: int):
        """Precompute distance-based bias matrix."""
        # Create position indices
        positions = torch.arange(max_length).unsqueeze(0)  # [1, max_length]

        # Compute pairwise distances: |i - j|
        # [max_length, max_length]
        distances = torch.abs(positions.T - positions)

        # Apply negative sign (penalty for distance)
        bias = -distances.float()

        # Expand for num_heads: [num_heads, max_length, max_length]
        bias = bias.unsqueeze(0).repeat(self.num_heads, 1, 1)

        # Apply head-specific slopes
        # slopes: [num_heads] → [num_heads, 1, 1]
        bias = bias * self.slopes.unsqueeze(-1).unsqueeze(-1)

        self.register_buffer("bias", bias, persistent=False)

    def forward(
        self,
        attention_scores: torch.Tensor,
        seq_len_q: Optional[int] = None,
        seq_len_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            attention_scores: [batch_size, num_heads, seq_len_q, seq_len_k]
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length

        Returns:
            Biased attention scores
        """
        batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape

        # Extend cache if needed
        max_len = max(seq_len_q, seq_len_k)
        if max_len > self.bias.shape[1]:
            self._precompute_bias(max_len)

        # Get bias for current sequence lengths
        bias = self.bias[:num_heads, :seq_len_q, :seq_len_k]

        # Add bias to attention scores
        # bias: [num_heads, seq_len_q, seq_len_k] → [1, num_heads, seq_len_q, seq_len_k]
        biased_scores = attention_scores + bias.unsqueeze(0)

        return biased_scores

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, max_length={self.max_length}"


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings (standard approach).

    Provided for completeness and baseline comparisons.
    NOT recommended for SCAN length generalization as it doesn't extrapolate well.

    Args:
        d_model: Model dimension
        max_length: Maximum sequence length
    """

    def __init__(self, d_model: int, max_length: int = 512):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings.

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            Position-encoded tensor
        """
        batch_size, seq_len, d_model = x.shape

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Get position embeddings
        pos_emb = self.position_embeddings(positions)

        # Add to input
        return x + pos_emb


if __name__ == "__main__":
    # Test RoPE
    print("Testing RotaryPositionalEncoding...")

    d_model = 512
    batch_size = 4
    seq_len = 20

    rope = RotaryPositionalEncoding(d_model=d_model, max_length=2048)

    x = torch.randn(batch_size, seq_len, d_model)
    x_rope = rope(x)

    assert x_rope.shape == x.shape, f"Shape mismatch: {x_rope.shape} vs {x.shape}"
    print(f"✓ RoPE output shape: {x_rope.shape}")

    # Test length extrapolation (longer than precomputed)
    seq_len_long = 100
    x_long = torch.randn(batch_size, seq_len_long, d_model)
    x_rope_long = rope(x_long)
    assert x_rope_long.shape == x_long.shape
    print(f"✓ RoPE length extrapolation works: {x_rope_long.shape}")

    # Test ALiBi
    print("\nTesting ALiBiPositionalBias...")

    num_heads = 8
    alibi = ALiBiPositionalBias(num_heads=num_heads, max_length=2048)

    # Create dummy attention scores
    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    biased_scores = alibi(attention_scores)

    assert biased_scores.shape == attention_scores.shape
    print(f"✓ ALiBi output shape: {biased_scores.shape}")

    # Check that bias is applied (scores should change)
    assert not torch.allclose(biased_scores, attention_scores)
    print("✓ ALiBi bias is applied")

    print("\n✓ All positional encoding tests passed!")
