"""
Tests for Structural Encoder

The Structural Encoder is responsible for extracting structure-invariant
representations using slot attention and abstraction layers.

Required by: SCI_ENGINEERING_STANDARDS.md Section 2.2
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from sci.models.components.structural_encoder import StructuralEncoder


@pytest.fixture
def structural_encoder_config():
    """Create configuration for structural encoder testing."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            d_model=2048,  # TinyLlama dimension
            base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            structural_encoder=SimpleNamespace(
                enabled=True,
                num_slots=8,
                num_layers=12,
                d_model=512,
                num_heads=8,
                dim_feedforward=2048,
                dropout=0.1,
                slot_attention=SimpleNamespace(
                    num_iterations=3,
                    epsilon=1e-8,
                    num_heads=8,
                    hidden_dim=512,
                    dropout=0.1,
                ),
                abstraction_layer=SimpleNamespace(
                    hidden_multiplier=2,
                    residual_init=0.1,
                    dropout=0.1,
                    injection_layers=[3, 6, 9],
                ),
                gnn=SimpleNamespace(
                    enabled=False,
                    num_layers=2,
                    num_heads=4,
                    dropout=0.1,
                ),
            ),
            position_encoding=SimpleNamespace(
                type="rotary",
                max_length=512,
                base=10000,
            ),
        ),
        training=SimpleNamespace(
            fp16=False,
        ),
    )
    return config


@pytest.fixture
def structural_encoder(structural_encoder_config):
    """Create structural encoder for testing."""
    encoder = StructuralEncoder(structural_encoder_config)

    # Set shared embedding (required for structural encoder to work)
    vocab_size = 32000
    d_model = structural_encoder_config.model.structural_encoder.d_model
    encoder.embedding = nn.Embedding(vocab_size, d_model)

    # Set to eval mode for deterministic behavior (disables dropout)
    encoder.eval()

    return encoder


class TestStructuralEncoderArchitecture:
    """Test structural encoder architecture and components."""

    def test_output_shape(self, structural_encoder, structural_encoder_config):
        """
        Test that structural encoder outputs correct shape.

        Expected: [batch_size, num_slots, d_model]
        """
        batch_size = 4
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            structural_slots, structural_scores = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        num_slots = structural_encoder_config.model.structural_encoder.num_slots
        d_model = structural_encoder_config.model.structural_encoder.d_model

        # Check structural slots shape
        expected_shape = (batch_size, num_slots, d_model)
        assert structural_slots.shape == expected_shape, (
            f"Structural slots shape {structural_slots.shape} != {expected_shape}"
        )

        print(f"✓ Structural encoder output shape correct: {structural_slots.shape}")

    def test_slot_queries_learnable(self, structural_encoder):
        """
        Test that slot queries are learnable parameters.

        Critical: Slots must be learned, not fixed.
        """
        # Check that slot queries exist and require gradients
        assert hasattr(structural_encoder.slot_attention, 'slot_queries'), (
            "Slot queries not found in slot attention module"
        )

        slot_queries = structural_encoder.slot_attention.slot_queries
        assert isinstance(slot_queries, torch.nn.Parameter), (
            "Slot queries should be nn.Parameter"
        )
        assert slot_queries.requires_grad, (
            "Slot queries should be learnable (requires_grad=True)"
        )

        # Check shape: [num_slots, d_model]
        num_slots = structural_encoder.num_slots
        d_model = structural_encoder.d_model
        expected_shape = (num_slots, d_model)

        assert slot_queries.shape == expected_shape, (
            f"Slot queries shape {slot_queries.shape} != {expected_shape}"
        )

        print(f"✓ Slot queries are learnable parameters: {slot_queries.shape}")

    def test_instruction_only_attention(self, structural_encoder):
        """
        CRITICAL TEST: Verify structural encoder only attends to instruction tokens.

        This ensures no data leakage from response tokens.
        """
        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Create instruction mask (only first half is instruction)
        instruction_len = 16
        instruction_mask = torch.zeros(batch_size, seq_len)
        instruction_mask[:, :instruction_len] = 1

        with torch.no_grad():
            structural_slots, _ = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        # Now test with different response but same instruction
        input_ids_2 = input_ids.clone()
        input_ids_2[:, instruction_len:] = torch.randint(0, vocab_size, (batch_size, seq_len - instruction_len))

        with torch.no_grad():
            structural_slots_2, _ = structural_encoder(
                input_ids=input_ids_2,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        # Structural representations should be very similar (same instruction)
        diff = (structural_slots - structural_slots_2).abs().max().item()

        # Allow small numerical differences
        assert diff < 1e-3, (
            f"Structural representations differ by {diff:.6f} despite same instruction. "
            "This suggests structural encoder is attending to response tokens!"
        )

        print(f"✓ Structural encoder only attends to instruction (diff: {diff:.6e})")

    def test_abstraction_layers_inject_at_correct_positions(
        self, structural_encoder, structural_encoder_config
    ):
        """
        Test that abstraction layers are injected at specified positions.
        """
        injection_layers = structural_encoder_config.model.structural_encoder.abstraction_layer.injection_layers

        # Check that abstraction layers exist
        assert hasattr(structural_encoder, 'abstraction_layers'), (
            "Abstraction layers not found in structural encoder"
        )

        # Check number of abstraction layers matches config
        assert len(structural_encoder.abstraction_layers) == len(injection_layers), (
            f"Expected {len(injection_layers)} abstraction layers, "
            f"got {len(structural_encoder.abstraction_layers)}"
        )

        # Check that abstraction layers are at correct positions
        for idx, layer_idx in enumerate(injection_layers):
            assert layer_idx in structural_encoder.abstraction_layer_positions, (
                f"Abstraction layer not found at position {layer_idx}"
            )

        print(f"✓ Abstraction layers inject at positions: {injection_layers}")

    def test_structural_scores_range(self, structural_encoder):
        """
        Test that structural scores are in [0, 1] range.

        Abstraction layers output sigmoid-activated scores.
        """
        batch_size = 4
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            _, structural_scores = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        # Check that scores exist
        assert structural_scores is not None, "Structural scores not returned"
        assert len(structural_scores) > 0, "No structural scores computed"

        # Check that all scores are in [0, 1]
        for i, scores in enumerate(structural_scores):
            assert scores.min() >= 0.0, (
                f"Structural scores at layer {i} have values < 0: {scores.min():.6f}"
            )
            assert scores.max() <= 1.0, (
                f"Structural scores at layer {i} have values > 1: {scores.max():.6f}"
            )

        print(f"✓ Structural scores in [0, 1] range across {len(structural_scores)} layers")


class TestStructuralEncoderFunctionality:
    """Test structural encoder functionality."""

    def test_handles_variable_length_sequences(self, structural_encoder):
        """
        Test that structural encoder handles variable-length sequences correctly.
        """
        batch_size = 2
        vocab_size = 32000

        for seq_len in [16, 32, 64, 128]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            instruction_mask = torch.ones(batch_size, seq_len)

            with torch.no_grad():
                try:
                    structural_slots, _ = structural_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        instruction_mask=instruction_mask,
                    )

                    # Check output shape is consistent (batch_size, num_slots, d_model)
                    assert structural_slots.shape[0] == batch_size
                    assert structural_slots.shape[1] == structural_encoder.num_slots
                    assert structural_slots.shape[2] == structural_encoder.d_model

                except Exception as e:
                    pytest.fail(f"Failed with seq_len={seq_len}: {e}")

        print("✓ Handles variable sequence lengths [16, 32, 64, 128]")

    def test_gradient_flow(self, structural_encoder):
        """
        Test that gradients flow through structural encoder.

        Critical for training.
        """
        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        # Forward pass with gradient tracking
        structural_slots, _ = structural_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            instruction_mask=instruction_mask,
        )

        # Compute dummy loss
        loss = structural_slots.sum()

        # Backward pass
        loss.backward()

        # Check that slot queries have gradients
        assert structural_encoder.slot_attention.slot_queries.grad is not None, (
            "Slot queries have no gradients"
        )

        # Check that encoder layers have gradients
        has_grad = any(
            p.grad is not None
            for p in structural_encoder.encoder.parameters()
            if p.requires_grad
        )
        assert has_grad, "Encoder layers have no gradients"

        print("✓ Gradients flow through structural encoder")

    def test_deterministic_with_eval_mode(self, structural_encoder):
        """
        Test that structural encoder is deterministic in eval mode.
        """
        structural_encoder.eval()

        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        # Two forward passes with same input
        with torch.no_grad():
            structural_slots_1, _ = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

            structural_slots_2, _ = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        # Outputs should be identical in eval mode
        assert torch.allclose(structural_slots_1, structural_slots_2, atol=1e-6), (
            "Structural encoder is non-deterministic in eval mode"
        )

        print("✓ Structural encoder is deterministic in eval mode")

    def test_slot_attention_convergence(self, structural_encoder):
        """
        Test that slot attention iterates and converges.
        """
        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            structural_slots, _ = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        # Check that slots are not all identical (indicating they specialized)
        slot_variance = structural_slots.var(dim=1).mean().item()

        assert slot_variance > 1e-6, (
            f"Slots have very low variance ({slot_variance:.2e}). "
            "They may not be specializing correctly."
        )

        print(f"✓ Slot attention converges (slot variance: {slot_variance:.6f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
