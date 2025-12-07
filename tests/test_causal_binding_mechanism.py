"""
Tests for Causal Binding Mechanism (CBM)

The CBM binds content to structural slots via cross-attention and causal
intervention, then broadcasts back to sequence positions for decoder injection.

Required by: SCI_ENGINEERING_STANDARDS.md Section 2.4
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from sci.models.components.causal_binding import CausalBindingMechanism


@pytest.fixture
def cbm_config():
    """Create configuration for CBM testing."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            d_model=2048,
            causal_binding=SimpleNamespace(
                d_model=2048,
                num_heads=8,
                dropout=0.1,
                injection_layers=[6, 11, 16],
                use_causal_intervention=True,
                injection_method="gated",
                gate_init=0.1,
            ),
        ),
    )
    return config


@pytest.fixture
def cbm(cbm_config):
    """Create causal binding mechanism for testing."""
    return CausalBindingMechanism(cbm_config)


class TestCausalBindingArchitecture:
    """Test CBM architecture and components."""

    def test_binding_attention_shape(self, cbm):
        """
        CRITICAL TEST: Verify binding attention outputs correct shape.

        Binding should map structural slots + content -> bound slots.
        Expected: [batch, num_slots, d_model]
        """
        batch_size = 4
        num_slots = 8
        d_model = 2048

        structural_slots = torch.randn(batch_size, num_slots, d_model)
        content_repr = torch.randn(batch_size, d_model)

        with torch.no_grad():
            bound_repr, attn_weights = cbm.bind(
                structural_slots=structural_slots,
                content_repr=content_repr,
                return_attention=True,
            )

        # Check bound representation shape
        expected_shape = (batch_size, num_slots, d_model)
        assert bound_repr.shape == expected_shape, (
            f"Bound repr shape {bound_repr.shape} != {expected_shape}"
        )

        # Check attention weights shape
        # Should be [batch, num_heads, num_slots, 1] (slots attend to content)
        assert attn_weights is not None, "Attention weights not returned"
        expected_attn_shape = (batch_size, cbm.num_heads, num_slots, 1)
        assert attn_weights.shape == expected_attn_shape, (
            f"Attention shape {attn_weights.shape} != {expected_attn_shape}"
        )

        print(f"✓ Binding attention shape correct: {bound_repr.shape}")
        print(f"✓ Attention weights shape: {attn_weights.shape}")

    def test_causal_intervention(self, cbm):
        """
        CRITICAL TEST: Verify causal intervention modifies bound representation.

        Edge weights should affect how slots influence each other via
        causal intervention (do-calculus).
        """
        batch_size = 4
        num_slots = 8
        d_model = 2048

        structural_slots = torch.randn(batch_size, num_slots, d_model)
        content_repr = torch.randn(batch_size, d_model)

        # Binding without edge weights
        with torch.no_grad():
            bound_no_edges, _ = cbm.bind(
                structural_slots=structural_slots,
                content_repr=content_repr,
                edge_weights=None,
            )

        # Binding with edge weights (causal intervention)
        edge_weights = torch.rand(batch_size, num_slots, num_slots)

        with torch.no_grad():
            bound_with_edges, _ = cbm.bind(
                structural_slots=structural_slots,
                content_repr=content_repr,
                edge_weights=edge_weights,
            )

        # Outputs should be different when edge weights are used
        diff = (bound_no_edges - bound_with_edges).abs().max().item()

        assert diff > 1e-6, (
            f"Edge weights didn't affect binding (diff={diff:.2e}). "
            "Causal intervention may not be working!"
        )

        # Test with different edge weights -> different outputs
        edge_weights_2 = torch.rand(batch_size, num_slots, num_slots)

        with torch.no_grad():
            bound_with_edges_2, _ = cbm.bind(
                structural_slots=structural_slots,
                content_repr=content_repr,
                edge_weights=edge_weights_2,
            )

        diff_2 = (bound_with_edges - bound_with_edges_2).abs().max().item()

        assert diff_2 > 1e-6, (
            "Different edge weights produce same output. "
            "Causal intervention not using edge weights!"
        )

        print(f"✓ Causal intervention works (no edges vs edges diff: {diff:.6f})")
        print(f"✓ Different edge weights produce different outputs (diff: {diff_2:.6f})")

    def test_broadcast_to_sequence(self, cbm):
        """
        CRITICAL TEST: Verify broadcast maps slots back to sequence positions.

        Broadcast should map [batch, num_slots, d_model] -> [batch, seq_len, d_model].
        """
        batch_size = 4
        num_slots = 8
        d_model = 2048

        bound_slots = torch.randn(batch_size, num_slots, d_model)

        # Test with different sequence lengths
        for seq_len in [16, 32, 64, 128]:
            with torch.no_grad():
                try:
                    broadcast_repr = cbm.broadcast(
                        bound_slots=bound_slots,
                        seq_len=seq_len,
                    )

                    # Check shape
                    expected_shape = (batch_size, seq_len, d_model)
                    assert broadcast_repr.shape == expected_shape, (
                        f"Broadcast shape {broadcast_repr.shape} != {expected_shape}"
                    )

                except Exception as e:
                    pytest.fail(f"Broadcast failed with seq_len={seq_len}: {e}")

        print("✓ Broadcast to sequence works for lengths [16, 32, 64, 128]")


class TestCausalBindingFunctionality:
    """Test CBM functionality."""

    def test_injection_modifies_hidden_states(self, cbm):
        """
        CRITICAL TEST: Verify injection actually modifies decoder hidden states.

        This is the core of CBM - it must inject bound representations.
        """
        batch_size = 4
        seq_len = 32
        d_model = 2048

        decoder_hidden = torch.randn(batch_size, seq_len, d_model)
        bound_repr = torch.randn(batch_size, seq_len, d_model)

        # Inject at an injection layer
        injection_layer = cbm.injection_layers[0]

        with torch.no_grad():
            injected = cbm.inject(
                decoder_hidden=decoder_hidden,
                bound_repr=bound_repr,
                layer_idx=injection_layer,
            )

        # Check shape preserved
        assert injected.shape == decoder_hidden.shape, (
            f"Injection changed shape: {injected.shape} != {decoder_hidden.shape}"
        )

        # Check that hidden states were modified
        diff = (injected - decoder_hidden).abs().max().item()

        assert diff > 1e-6, (
            f"Injection didn't modify hidden states (diff={diff:.2e}). "
            "CBM injection may not be working!"
        )

        print(f"✓ Injection modifies hidden states (max diff: {diff:.6f})")

    def test_no_injection_at_non_injection_layers(self, cbm):
        """
        Verify that CBM doesn't inject at non-injection layers.
        """
        batch_size = 4
        seq_len = 32
        d_model = 2048

        decoder_hidden = torch.randn(batch_size, seq_len, d_model)
        bound_repr = torch.randn(batch_size, seq_len, d_model)

        # Find a non-injection layer
        non_injection_layer = 0
        while non_injection_layer in cbm.injection_layers:
            non_injection_layer += 1

        with torch.no_grad():
            injected = cbm.inject(
                decoder_hidden=decoder_hidden,
                bound_repr=bound_repr,
                layer_idx=non_injection_layer,
            )

        # Should return unchanged
        assert torch.allclose(injected, decoder_hidden, atol=1e-8), (
            f"Non-injection layer {non_injection_layer} modified hidden states!"
        )

        print(f"✓ No injection at non-injection layer {non_injection_layer}")

    def test_injection_at_all_specified_layers(self, cbm):
        """
        Verify injection works at ALL specified injection layers.
        """
        batch_size = 2
        seq_len = 32
        d_model = 2048

        decoder_hidden = torch.randn(batch_size, seq_len, d_model)
        bound_repr = torch.randn(batch_size, seq_len, d_model)

        for layer_idx in cbm.injection_layers:
            with torch.no_grad():
                try:
                    injected = cbm.inject(
                        decoder_hidden=decoder_hidden,
                        bound_repr=bound_repr,
                        layer_idx=layer_idx,
                    )

                    # Check modification
                    diff = (injected - decoder_hidden).abs().max().item()
                    assert diff > 1e-6, (
                        f"Layer {layer_idx}: No modification (diff={diff:.2e})"
                    )

                except Exception as e:
                    pytest.fail(f"Injection failed at layer {layer_idx}: {e}")

        print(f"✓ Injection works at all layers: {cbm.injection_layers}")

    def test_gating_mechanism(self, cbm):
        """
        Test that gating mechanism controls injection amount.

        Gate values should be in [0, 1] and control blend.
        """
        batch_size = 2
        seq_len = 32
        d_model = 2048

        decoder_hidden = torch.randn(batch_size, seq_len, d_model)
        bound_repr = torch.randn(batch_size, seq_len, d_model)

        injection_layer = cbm.injection_layers[0]

        # Get gate values by inspecting the computation
        combined = torch.cat([decoder_hidden, bound_repr], dim=-1)

        with torch.no_grad():
            gate = cbm.injection_gate[str(injection_layer)](combined)

        # Check gate values in [0, 1]
        assert gate.min() >= 0.0, f"Gate values < 0: {gate.min()}"
        assert gate.max() <= 1.0, f"Gate values > 1: {gate.max()}"

        print(f"✓ Gate values in [0, 1]: min={gate.min():.3f}, max={gate.max():.3f}")

    def test_gradient_flow(self, cbm):
        """
        Test that gradients flow through CBM components.

        Critical for training.
        """
        batch_size = 2
        num_slots = 8
        d_model = 2048

        structural_slots = torch.randn(batch_size, num_slots, d_model, requires_grad=True)
        content_repr = torch.randn(batch_size, d_model, requires_grad=True)

        # Forward pass with binding
        bound_repr, _ = cbm.bind(
            structural_slots=structural_slots,
            content_repr=content_repr,
        )

        # Compute dummy loss
        loss = bound_repr.sum()

        # Backward pass
        loss.backward()

        # Check that inputs have gradients
        assert structural_slots.grad is not None, "Structural slots have no gradients"
        assert content_repr.grad is not None, "Content repr has no gradients"

        # Check that CBM parameters have gradients
        has_grad = any(
            p.grad is not None
            for p in cbm.parameters()
            if p.requires_grad
        )
        assert has_grad, "CBM has no parameter gradients"

        print("✓ Gradients flow through CBM")

    def test_binding_is_deterministic_in_eval(self, cbm):
        """
        Test that binding is deterministic in eval mode.
        """
        cbm.eval()

        batch_size = 2
        num_slots = 8
        d_model = 2048

        structural_slots = torch.randn(batch_size, num_slots, d_model)
        content_repr = torch.randn(batch_size, d_model)

        # Two forward passes with same input
        with torch.no_grad():
            bound_1, _ = cbm.bind(structural_slots, content_repr)
            bound_2, _ = cbm.bind(structural_slots, content_repr)

        # Should be identical in eval mode
        assert torch.allclose(bound_1, bound_2, atol=1e-6), (
            "CBM is non-deterministic in eval mode"
        )

        print("✓ CBM is deterministic in eval mode")

    def test_handles_variable_batch_sizes(self, cbm):
        """
        Test that CBM handles variable batch sizes.
        """
        num_slots = 8
        d_model = 2048

        for batch_size in [1, 2, 4, 8]:
            structural_slots = torch.randn(batch_size, num_slots, d_model)
            content_repr = torch.randn(batch_size, d_model)

            with torch.no_grad():
                try:
                    bound_repr, _ = cbm.bind(structural_slots, content_repr)

                    expected_shape = (batch_size, num_slots, d_model)
                    assert bound_repr.shape == expected_shape, (
                        f"Batch {batch_size}: shape {bound_repr.shape} != {expected_shape}"
                    )

                except Exception as e:
                    pytest.fail(f"CBM failed with batch_size={batch_size}: {e}")

        print("✓ Handles variable batch sizes [1, 2, 4, 8]")

    def test_handles_variable_num_slots(self, cbm):
        """
        Test that CBM handles different numbers of slots.
        """
        batch_size = 2
        d_model = 2048

        for num_slots in [4, 8, 16]:
            structural_slots = torch.randn(batch_size, num_slots, d_model)
            content_repr = torch.randn(batch_size, d_model)

            with torch.no_grad():
                try:
                    bound_repr, _ = cbm.bind(structural_slots, content_repr)

                    expected_shape = (batch_size, num_slots, d_model)
                    assert bound_repr.shape == expected_shape, (
                        f"Slots {num_slots}: shape {bound_repr.shape} != {expected_shape}"
                    )

                except Exception as e:
                    pytest.fail(f"CBM failed with num_slots={num_slots}: {e}")

        print("✓ Handles variable num_slots [4, 8, 16]")

    def test_end_to_end_pipeline(self, cbm):
        """
        Test complete CBM pipeline: bind -> broadcast -> inject.
        """
        batch_size = 2
        num_slots = 8
        seq_len = 32
        d_model = 2048

        # Step 1: Bind
        structural_slots = torch.randn(batch_size, num_slots, d_model)
        content_repr = torch.randn(batch_size, d_model)
        edge_weights = torch.rand(batch_size, num_slots, num_slots)

        with torch.no_grad():
            bound_repr, _ = cbm.bind(
                structural_slots=structural_slots,
                content_repr=content_repr,
                edge_weights=edge_weights,
            )

        # Step 2: Broadcast
        with torch.no_grad():
            broadcast_repr = cbm.broadcast(
                bound_slots=bound_repr,
                seq_len=seq_len,
            )

        # Step 3: Inject
        decoder_hidden = torch.randn(batch_size, seq_len, d_model)

        with torch.no_grad():
            injected = cbm.inject(
                decoder_hidden=decoder_hidden,
                bound_repr=broadcast_repr,
                layer_idx=cbm.injection_layers[0],
            )

        # Verify final shape
        assert injected.shape == decoder_hidden.shape, (
            f"End-to-end pipeline shape mismatch: {injected.shape}"
        )

        # Verify modification occurred
        diff = (injected - decoder_hidden).abs().max().item()
        assert diff > 1e-6, (
            f"End-to-end pipeline didn't modify hidden states (diff={diff:.2e})"
        )

        print(f"✓ End-to-end pipeline works (bind -> broadcast -> inject)")
        print(f"✓ Final modification: {diff:.6f}")


class TestCausalBindingConfiguration:
    """Test CBM configuration options."""

    def test_causal_intervention_can_be_disabled(self, cbm_config):
        """
        Test that causal intervention can be disabled (ablation mode).
        """
        # Disable causal intervention
        cbm_config.model.causal_binding.use_causal_intervention = False

        cbm_no_intervention = CausalBindingMechanism(cbm_config)

        # Should not have intervention layers
        assert not hasattr(cbm_no_intervention, 'intervention_query') or \
               cbm_no_intervention.use_causal_intervention is False, (
            "Causal intervention should be disabled"
        )

        print("✓ Causal intervention can be disabled")

    def test_injection_layers_configurable(self, cbm_config):
        """
        Test that injection layers are configurable.
        """
        # Test with different injection layers
        for injection_layers in [[6], [6, 11], [6, 11, 16], [0, 5, 10, 15, 20]]:
            cbm_config.model.causal_binding.injection_layers = injection_layers

            cbm_custom = CausalBindingMechanism(cbm_config)

            assert cbm_custom.injection_layers == injection_layers, (
                f"Injection layers mismatch: {cbm_custom.injection_layers} != {injection_layers}"
            )

            # Check adapters exist for all layers
            for layer in injection_layers:
                assert str(layer) in cbm_custom.adapters, (
                    f"Adapter missing for layer {layer}"
                )
                assert str(layer) in cbm_custom.injection_gate, (
                    f"Gate missing for layer {layer}"
                )

        print("✓ Injection layers are configurable")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
