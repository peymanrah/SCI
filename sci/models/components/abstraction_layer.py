"""
AbstractionLayer: THE KEY INNOVATION of SCI

This module learns to distinguish structural tokens from content tokens by outputting
"structuralness" scores in the range [0, 1]:
- High score (>0.7): Structural word (e.g., "twice", "and", "left")
- Low score (<0.3): Content word (e.g., "walk", "jump", "run")

The layer uses these scores to suppress content information while preserving structural patterns,
enabling structural representations that are invariant to content substitution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AbstractionLayer(nn.Module):
    """
    AbstractionLayer: Learns to detect and preserve structural patterns.

    This is the core innovation that enables SCI to separate structure from content.
    The layer computes token-wise "structuralness" scores and applies them to suppress
    content while preserving structural information.

    Architecture:
        1. Structural detector: MLP with sigmoid output → scores in [0, 1]
        2. Structural masking: Multiply hidden states by scores
        3. Residual connection: Controlled by learnable gate parameter

    Args:
        d_model: Hidden dimension
        hidden_multiplier: Multiplier for MLP hidden size (default: 2)
        residual_init: Initial value for residual gate (default: 0.1)
        dropout: Dropout probability (default: 0.1)

    Input:
        hidden_states: [batch_size, seq_len, d_model]

    Output:
        output: [batch_size, seq_len, d_model] - Structurally-masked representation
        structural_scores: [batch_size, seq_len, d_model] - Per-token structuralness scores
    """

    def __init__(
        self,
        d_model: int,
        hidden_multiplier: int = 2,
        residual_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = d_model * hidden_multiplier

        # Structural detector: MLP that outputs structuralness scores
        # Uses Sigmoid to ensure scores are in [0, 1]
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model),
            nn.Sigmoid(),  # CRITICAL: Maps to [0, 1] for interpretability
        )

        # Residual gate: Learnable parameter controlling how much original info to preserve
        # Initialize small to encourage learning strong structural masking
        self.residual_gate = nn.Parameter(torch.tensor(residual_init))

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)

        # For gradient clipping if needed
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of AbstractionLayer.

        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] - Optional mask (1 for valid, 0 for padding)

        Returns:
            output: [batch_size, seq_len, d_model] - Structurally-masked representation
            structural_scores: [batch_size, seq_len, d_model] - Structuralness scores
        """
        # Compute structuralness scores for each token
        # Shape: [batch_size, seq_len, d_model]
        structural_scores = self.structural_detector(hidden_states)

        # Apply structural masking: suppress low-scoring (content) dimensions
        structural_repr = hidden_states * structural_scores

        # Apply attention mask if provided (zero out padding tokens)
        if attention_mask is not None:
            # Expand mask to match hidden dimension
            # attention_mask: [batch_size, seq_len] → [batch_size, seq_len, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            structural_repr = structural_repr * mask_expanded
            structural_scores = structural_scores * mask_expanded

        # Residual connection: blend structural repr with original
        # This allows gradient flow and prevents complete information loss
        output = (1 - self.residual_gate) * structural_repr + self.residual_gate * hidden_states

        # Normalize for stability
        output = self.layer_norm(output)

        return output, structural_scores

    def get_structural_statistics(
        self, structural_scores: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> dict:
        """
        Compute statistics about structural scores (for analysis/debugging).

        Args:
            structural_scores: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] - Optional mask

        Returns:
            Dictionary with statistics:
                - mean_score: Average structuralness across all tokens
                - high_structural_ratio: Fraction of scores > 0.7
                - low_structural_ratio: Fraction of scores < 0.3
                - token_structural_scores: Average score per token position
        """
        # CRITICAL #9: Add eps=1e-8 protection against division by zero
        eps = 1e-8

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            valid_scores = structural_scores * mask_expanded
            # Count valid elements: mask_expanded has shape [batch, seq, 1], need to account for d_model
            num_valid = mask_expanded.sum() * structural_scores.size(-1)
        else:
            valid_scores = structural_scores
            num_valid = structural_scores.numel()

        # Average score across all dimensions and tokens (with epsilon protection)
        mean_score = valid_scores.sum() / (num_valid + eps)

        # Count high/low structural scores (with epsilon protection)
        high_structural = (valid_scores > 0.7).float().sum() / (num_valid + eps)
        low_structural = (valid_scores < 0.3).float().sum() / (num_valid + eps)

        # Average structural score per token (avg across d_model dimension)
        token_scores = structural_scores.mean(dim=-1)  # [batch_size, seq_len]

        return {
            "mean_score": mean_score.item(),
            "high_structural_ratio": high_structural.item(),
            "low_structural_ratio": low_structural.item(),
            "token_structural_scores": token_scores.detach(),
            "residual_gate": self.residual_gate.item(),
        }

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"d_model={self.d_model}, hidden_dim={self.hidden_dim}, "
            f"residual_gate={self.residual_gate.item():.4f}"
        )


class MultiHeadAbstractionLayer(nn.Module):
    """
    Multi-head variant of AbstractionLayer (EXPERIMENTAL).

    Uses multiple independent structural detectors to capture different
    types of structural patterns (e.g., syntactic vs semantic structure).

    This is an advanced variant that can be used in ablation studies.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        hidden_multiplier: int = 2,
        residual_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # Create multiple structural detectors (one per head)
        self.abstraction_heads = nn.ModuleList(
            [
                AbstractionLayer(
                    d_model=self.head_dim,
                    hidden_multiplier=hidden_multiplier,
                    residual_init=residual_init,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

        # Combine heads
        self.combine = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-head abstraction.

        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
            structural_scores: [batch_size, seq_len, d_model, num_heads]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Split into heads
        # [batch_size, seq_len, d_model] → [batch_size, seq_len, num_heads, head_dim]
        hidden_states_heads = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply abstraction layer to each head
        head_outputs = []
        head_scores = []

        for i, head_layer in enumerate(self.abstraction_heads):
            # Get head input: [batch_size, seq_len, head_dim]
            head_input = hidden_states_heads[:, :, i, :]

            # Apply abstraction
            head_out, head_score = head_layer(head_input, attention_mask)

            head_outputs.append(head_out)
            head_scores.append(head_score)

        # Concatenate heads
        # [num_heads, batch_size, seq_len, head_dim] → [batch_size, seq_len, d_model]
        output = torch.cat(head_outputs, dim=-1)

        # Combine and normalize
        output = self.combine(output)
        output = self.layer_norm(output)

        # Stack scores for analysis
        # [num_heads, batch_size, seq_len, head_dim] → [batch_size, seq_len, d_model, num_heads]
        structural_scores = torch.stack(head_scores, dim=-1)

        return output, structural_scores


if __name__ == "__main__":
    # Quick test
    print("Testing AbstractionLayer...")

    d_model = 512
    batch_size = 4
    seq_len = 20

    # Create layer
    layer = AbstractionLayer(d_model=d_model, hidden_multiplier=2)

    # Create dummy input
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    output, scores = layer(hidden_states, attention_mask)

    # Check shapes
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    assert scores.shape == (batch_size, seq_len, d_model), f"Scores shape mismatch: {scores.shape}"

    # Check score range [0, 1]
    assert scores.min() >= 0.0 and scores.max() <= 1.0, f"Scores out of range: [{scores.min():.3f}, {scores.max():.3f}]"

    # Get statistics
    stats = layer.get_structural_statistics(scores, attention_mask)
    print(f"Mean score: {stats['mean_score']:.3f}")
    print(f"High structural ratio: {stats['high_structural_ratio']:.3f}")
    print(f"Low structural ratio: {stats['low_structural_ratio']:.3f}")
    print(f"Residual gate: {stats['residual_gate']:.4f}")

    print("✓ AbstractionLayer tests passed!")
