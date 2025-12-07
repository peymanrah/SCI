"""
Slot Attention mechanism for structural representation.

Slot Attention (Locatello et al. 2020) is used in the Structural Encoder to pool
variable-length sequences into a fixed number of "slots" that represent different
structural aspects of the input.

For example, with 8 slots, different slots might specialize in detecting:
- Slot 1: Repetition patterns ("twice", "thrice")
- Slot 2: Sequential patterns ("and", "after")
- Slot 3: Directional modifiers ("left", "right")
- etc.

References:
    Object-Centric Learning with Slot Attention (Locatello et al. 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SlotAttention(nn.Module):
    """
    Slot Attention mechanism for pooling sequences into structural slots.

    Iteratively refines a set of slot representations by attending to the input
    sequence. Each slot learns to capture a different aspect of the structural pattern.

    Args:
        num_slots: Number of structural slots (default: 8)
        d_model: Model dimension
        num_iterations: Number of refinement iterations (default: 3)
        epsilon: Small value for numerical stability (default: 1e-8)
        hidden_dim: Hidden dimension for MLP (default: None, uses d_model)

    Input:
        inputs: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len] - Optional mask for padding

    Output:
        slots: [batch_size, num_slots, d_model] - Slot representations
    """

    def __init__(
        self,
        num_slots: int = 8,
        d_model: int = 512,
        num_iterations: int = 3,
        epsilon: float = 1e-8,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        if hidden_dim is None:
            hidden_dim = d_model

        # Learnable slot initialization
        # Initialize with small random values
        self.slots_mu = nn.Parameter(torch.randn(1, 1, d_model))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, d_model))

        # Also create slot_queries for compatibility with tests
        # These are initialized slot representations that get refined
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.1)

        # Layer norm for inputs and slots
        self.norm_inputs = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)

        # Attention: slots attend to inputs
        # Query: slots, Key/Value: inputs
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # GRU for slot updates
        self.gru = nn.GRUCell(d_model, d_model)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Layer norm for MLP
        self.norm_mlp = nn.LayerNorm(d_model)

    def initialize_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize slots with slot_queries (learnable parameters).

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            slots: [batch_size, num_slots, d_model]
        """
        # Use slot_queries as the base initialization
        # This ensures slot_queries gets gradients during training
        slots = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Add noise for initialization (only in training mode)
        if self.training:
            # Add small Gaussian noise based on slots_log_sigma
            sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
            slots = slots + sigma * torch.randn_like(slots)

        return slots

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Slot Attention to pool inputs into slots.

        Args:
            inputs: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] - 1 for valid, 0 for padding

        Returns:
            slots: [batch_size, num_slots, d_model]
        """
        batch_size, seq_len, d_model = inputs.shape

        # Normalize inputs
        inputs = self.norm_inputs(inputs)

        # Initialize slots
        slots = self.initialize_slots(batch_size, inputs.device)

        # Compute keys and values (constant across iterations)
        k = self.to_k(inputs)  # [batch_size, seq_len, d_model]
        v = self.to_v(inputs)  # [batch_size, seq_len, d_model]

        # Prepare attention mask for softmax
        if attention_mask is not None:
            # Convert mask to additive form: 0 for valid, -inf for padding
            # [batch_size, seq_len] → [batch_size, 1, seq_len]
            attn_mask = (1.0 - attention_mask.float()) * -1e9
            attn_mask = attn_mask.unsqueeze(1)
        else:
            attn_mask = None

        # Iterative refinement
        for iteration in range(self.num_iterations):
            slots_prev = slots

            # Normalize slots
            slots = self.norm_slots(slots)

            # Compute queries from slots
            q = self.to_q(slots)  # [batch_size, num_slots, d_model]

            # Compute attention scores
            # [batch_size, num_slots, d_model] x [batch_size, d_model, seq_len]
            # → [batch_size, num_slots, seq_len]
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / (d_model ** 0.5)

            # Apply attention mask if provided
            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask

            # Softmax over sequence dimension (each slot attends to all inputs)
            attn = F.softmax(attn_logits, dim=-1)  # [batch_size, num_slots, seq_len]

            # Weighted sum of values
            # [batch_size, num_slots, seq_len] x [batch_size, seq_len, d_model]
            # → [batch_size, num_slots, d_model]
            updates = torch.bmm(attn, v)

            # Update slots with GRU
            # Flatten for GRU: [batch_size * num_slots, d_model]
            slots_flat = slots.view(batch_size * self.num_slots, d_model)
            updates_flat = updates.view(batch_size * self.num_slots, d_model)

            slots_flat = self.gru(updates_flat, slots_flat)

            # Reshape back: [batch_size, num_slots, d_model]
            slots = slots_flat.view(batch_size, self.num_slots, d_model)

            # MLP refinement with residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

    def extra_repr(self) -> str:
        return (
            f"num_slots={self.num_slots}, d_model={self.d_model}, "
            f"num_iterations={self.num_iterations}"
        )


class SimpleSlotAttention(nn.Module):
    """
    Simplified Slot Attention without iterative refinement.

    Uses single-step attention pooling instead of iterative GRU updates.
    Faster but potentially less expressive than full Slot Attention.

    This can be used for ablation studies or when computational efficiency is critical.
    """

    def __init__(
        self,
        num_slots: int = 8,
        d_model: int = 512,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.d_model = d_model

        # Learnable slot queries
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model))

        # Attention layers
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Layer norms
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply simplified slot attention.

        Args:
            inputs: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len]

        Returns:
            slots: [batch_size, num_slots, d_model]
        """
        batch_size = inputs.shape[0]

        # Expand slot queries for batch
        queries = self.slot_queries.expand(batch_size, -1, -1)

        # Compute Q, K, V
        q = self.to_q(queries)  # [batch_size, num_slots, d_model]
        k = self.to_k(inputs)   # [batch_size, seq_len, d_model]
        v = self.to_v(inputs)   # [batch_size, seq_len, d_model]

        # Attention scores
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5)

        # Apply mask if provided
        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask.float()) * -1e9
            attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            attn_logits = attn_logits + attn_mask

        # Softmax
        attn = F.softmax(attn_logits, dim=-1)

        # Weighted sum
        slots = torch.bmm(attn, v)  # [batch_size, num_slots, d_model]

        # Project and normalize
        slots = self.out_proj(slots)
        slots = self.norm(slots)

        return slots


if __name__ == "__main__":
    # Test Slot Attention
    print("Testing SlotAttention...")

    batch_size = 4
    seq_len = 20
    d_model = 512
    num_slots = 8

    slot_attn = SlotAttention(num_slots=num_slots, d_model=d_model, num_iterations=3)

    # Create dummy input
    inputs = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    slots = slot_attn(inputs, attention_mask)

    # Check shape
    expected_shape = (batch_size, num_slots, d_model)
    assert slots.shape == expected_shape, f"Shape mismatch: {slots.shape} vs {expected_shape}"
    print(f"✓ Slot Attention output shape: {slots.shape}")

    # Test without mask
    slots_no_mask = slot_attn(inputs)
    assert slots_no_mask.shape == expected_shape
    print("✓ Slot Attention works without mask")

    # Test simple variant
    print("\nTesting SimpleSlotAttention...")
    simple_slot_attn = SimpleSlotAttention(num_slots=num_slots, d_model=d_model)
    slots_simple = simple_slot_attn(inputs, attention_mask)

    assert slots_simple.shape == expected_shape
    print(f"✓ Simple Slot Attention output shape: {slots_simple.shape}")

    print("\n✓ All Slot Attention tests passed!")
