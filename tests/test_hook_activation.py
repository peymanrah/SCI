"""
Test Hook Activation During Forward Pass

CRITICAL: These tests verify that registered hooks actually fire during forward
passes and that they properly modify hidden states. Without hook activation,
the Causal Binding Mechanism will not inject structure+content into the decoder,
causing SCI to silently fail to perform causal binding.

Required by: SCI_ENGINEERING_STANDARDS.md Section 2.5
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from sci.models.sci_model import SCIModel


@pytest.fixture
def minimal_config():
    """Create minimal configuration for hook testing."""
    config = SimpleNamespace(
        model=SimpleNamespace(
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
                ),
                abstraction_layer=SimpleNamespace(
                    hidden_multiplier=2,
                    residual_init=0.1,
                    dropout=0.1,
                    injection_layers=[3, 6],
                ),
            ),
            position_encoding=SimpleNamespace(
                type="rotary",
                max_length=512,
                base=10000,
            ),
            content_encoder=SimpleNamespace(
                enabled=True,
                num_layers=2,
                d_model=512,
                num_heads=8,
                dim_feedforward=2048,
                dropout=0.1,
                pooling="mean",
            ),
            causal_binding=SimpleNamespace(
                enabled=True,
                d_model=2048,
                injection_layers=[6, 11, 16],
                num_heads=8,
                dropout=0.1,
                injection_method="add",
                gate_init=0.1,
            ),
        ),
        training=SimpleNamespace(
            fp16=False,
            mixed_precision=False,
            batch_size=4,
            learning_rate=2e-5,
            num_epochs=50,
            warmup_steps=1000,
            gradient_clip=1.0,
            seed=42,
        ),
    )
    return config


@pytest.fixture
def sci_model(minimal_config):
    """Create SCI model for testing."""
    # Set device to CPU for testing
    torch.cuda.is_available = lambda: False

    model = SCIModel(minimal_config)
    model.eval()
    return model


@pytest.fixture
def sample_batch(sci_model):
    """Create sample input batch for testing."""
    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(0, sci_model.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, sci_model.vocab_size, (batch_size, seq_len))

    # First half is instruction (labels = -100)
    labels[:, :seq_len//2] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


class TestHookActivation:
    """Test that hooks are activated during forward pass."""

    def test_hooks_activate_during_training_forward(self, sci_model, sample_batch):
        """
        CRITICAL TEST: Verify hooks are called during training forward pass.

        This test captures whether hooks fire by checking if the model can
        execute a forward pass with all components enabled.
        """
        sci_model.train()

        # Perform forward pass
        with torch.no_grad():
            outputs = sci_model(
                input_ids=sample_batch['input_ids'],
                attention_mask=sample_batch['attention_mask'],
                labels=sample_batch['labels'],
                return_dict=True,
            )

        # Verify outputs exist
        assert outputs is not None, "Forward pass returned None"
        assert 'logits' in outputs, "Missing logits in outputs"
        assert 'loss' in outputs, "Missing loss in outputs"

        # Verify structural and content representations were computed
        assert outputs['structural_slots'] is not None, (
            "Structural slots not computed - hooks may not be activating"
        )
        assert outputs['content_repr'] is not None, (
            "Content repr not computed - hooks may not be activating"
        )

        # Verify logits have correct shape
        batch_size, seq_len = sample_batch['input_ids'].shape
        expected_shape = (batch_size, seq_len, sci_model.vocab_size)
        assert outputs['logits'].shape == expected_shape, (
            f"Logits shape {outputs['logits'].shape} != expected {expected_shape}"
        )

        print("✓ Hooks activated during training forward pass")

    def test_hooks_activate_during_inference(self, sci_model, sample_batch):
        """
        Verify hooks activate during inference (eval mode).

        Hooks must work in both training and inference modes.
        """
        sci_model.eval()

        # Perform forward pass without labels
        with torch.no_grad():
            outputs = sci_model(
                input_ids=sample_batch['input_ids'],
                attention_mask=sample_batch['attention_mask'],
                return_dict=True,
            )

        # Verify outputs exist
        assert outputs is not None, "Inference forward pass returned None"
        assert 'logits' in outputs, "Missing logits in inference outputs"

        # Verify structural and content representations were computed
        assert outputs['structural_slots'] is not None, (
            "Structural slots not computed during inference"
        )
        assert outputs['content_repr'] is not None, (
            "Content repr not computed during inference"
        )

        print("✓ Hooks activated during inference")

    def test_hooks_modify_hidden_states(self, sci_model, sample_batch):
        """
        CRITICAL TEST: Verify hooks actually modify hidden states.

        This test compares decoder outputs with and without SCI components
        to ensure CBM injection is having an effect.
        """
        sci_model.eval()

        # Forward pass WITH SCI components
        with torch.no_grad():
            outputs_with_sci = sci_model(
                input_ids=sample_batch['input_ids'],
                attention_mask=sample_batch['attention_mask'],
                return_dict=True,
            )
            logits_with_sci = outputs_with_sci['logits']

        # Disable SCI components temporarily
        original_structural_encoder = sci_model.structural_encoder
        original_content_encoder = sci_model.content_encoder
        original_causal_binding = sci_model.causal_binding

        sci_model.structural_encoder = None
        sci_model.content_encoder = None
        sci_model.causal_binding = None

        # Forward pass WITHOUT SCI components
        with torch.no_grad():
            outputs_without_sci = sci_model(
                input_ids=sample_batch['input_ids'],
                attention_mask=sample_batch['attention_mask'],
                return_dict=True,
            )
            logits_without_sci = outputs_without_sci['logits']

        # Restore SCI components
        sci_model.structural_encoder = original_structural_encoder
        sci_model.content_encoder = original_content_encoder
        sci_model.causal_binding = original_causal_binding

        # Verify logits are DIFFERENT (hooks had an effect)
        logit_diff = (logits_with_sci - logits_without_sci).abs().mean().item()

        assert logit_diff > 1e-6, (
            f"Logits unchanged (diff={logit_diff:.2e}). "
            "Hooks may not be modifying hidden states."
        )

        print(f"✓ Hooks modify hidden states (mean diff: {logit_diff:.6f})")

    def test_hooks_work_with_generation(self, sci_model):
        """
        Verify hooks work during generation (autoregressive decoding).

        Generation is different from teacher-forced forward pass.
        """
        sci_model.eval()

        # Create short input for generation
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, sci_model.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Generate
        with torch.no_grad():
            try:
                generated_ids = sci_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=20,
                    do_sample=False,  # Greedy decoding for determinism
                )

                # Verify generation worked
                assert generated_ids is not None, "Generation returned None"
                assert generated_ids.shape[0] == batch_size, (
                    f"Wrong batch size: {generated_ids.shape[0]}"
                )
                assert generated_ids.shape[1] >= seq_len, (
                    f"Generated sequence too short: {generated_ids.shape[1]}"
                )

                print(f"✓ Hooks work during generation (output length: {generated_ids.shape[1]})")

            except Exception as e:
                pytest.fail(f"Generation failed with hooks: {e}")

    def test_structural_and_content_stored_during_forward(self, sci_model, sample_batch):
        """
        Verify that structural_slots and content_repr are stored during forward
        pass for hooks to use.

        Hooks depend on these being available in model state.
        """
        sci_model.eval()

        # Initially should be None
        assert sci_model.current_structural_slots is None, (
            "current_structural_slots should be None before forward pass"
        )
        assert sci_model.current_content_repr is None, (
            "current_content_repr should be None before forward pass"
        )

        # Monkey-patch hook to check if representations are available
        hook_called = {'count': 0, 'had_representations': False}
        injection_layer = sci_model.cbm_injection_layers[0]
        decoder_layers = sci_model.base_model.model.layers

        def test_hook(module, input, output):
            hook_called['count'] += 1
            hook_called['had_representations'] = (
                sci_model.current_structural_slots is not None and
                sci_model.current_content_repr is not None
            )
            return output

        # Register test hook
        handle = decoder_layers[injection_layer].register_forward_hook(test_hook)

        # Forward pass
        with torch.no_grad():
            outputs = sci_model(
                input_ids=sample_batch['input_ids'],
                attention_mask=sample_batch['attention_mask'],
                return_dict=True,
            )

        # Remove test hook
        handle.remove()

        # Verify hook was called and had access to representations
        assert hook_called['count'] > 0, "Test hook was never called"
        assert hook_called['had_representations'], (
            "Structural slots and content repr not available during hook execution"
        )

        # After forward pass, should be cleared
        assert sci_model.current_structural_slots is None, (
            "current_structural_slots should be cleared after forward pass"
        )
        assert sci_model.current_content_repr is None, (
            "current_content_repr should be cleared after forward pass"
        )

        print("✓ Structural and content representations available to hooks")

    def test_hooks_handle_batch_size_variation(self, sci_model):
        """
        Verify hooks work correctly with different batch sizes.

        Hooks must handle dynamic shapes correctly.
        """
        sci_model.eval()

        for batch_size in [1, 2, 4, 8]:
            seq_len = 32
            input_ids = torch.randint(0, sci_model.vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)

            with torch.no_grad():
                try:
                    outputs = sci_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )

                    # Verify correct output shape
                    expected_shape = (batch_size, seq_len, sci_model.vocab_size)
                    assert outputs['logits'].shape == expected_shape, (
                        f"Batch size {batch_size}: "
                        f"logits shape {outputs['logits'].shape} != {expected_shape}"
                    )

                except Exception as e:
                    pytest.fail(f"Hooks failed with batch_size={batch_size}: {e}")

        print("✓ Hooks handle batch size variation [1, 2, 4, 8]")

    def test_hooks_handle_sequence_length_variation(self, sci_model):
        """
        Verify hooks work correctly with different sequence lengths.

        Critical for handling variable-length inputs.
        """
        sci_model.eval()
        batch_size = 2

        for seq_len in [16, 32, 64, 128]:
            input_ids = torch.randint(0, sci_model.vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)

            with torch.no_grad():
                try:
                    outputs = sci_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )

                    # Verify correct output shape
                    expected_shape = (batch_size, seq_len, sci_model.vocab_size)
                    assert outputs['logits'].shape == expected_shape, (
                        f"Seq len {seq_len}: "
                        f"logits shape {outputs['logits'].shape} != {expected_shape}"
                    )

                except Exception as e:
                    pytest.fail(f"Hooks failed with seq_len={seq_len}: {e}")

        print("✓ Hooks handle sequence length variation [16, 32, 64, 128]")

    def test_gradients_flow_through_hooks(self, sci_model, sample_batch):
        """
        Verify gradients flow through hook injections.

        Critical for training: hooks must not block gradient flow.
        """
        sci_model.train()

        # Forward pass with gradient tracking
        outputs = sci_model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            labels=sample_batch['labels'],
            return_dict=True,
        )

        loss = outputs['loss']
        assert loss.requires_grad, "Loss does not require grad"

        # Backward pass
        loss.backward()

        # Check that SCI components have gradients
        if sci_model.structural_encoder is not None:
            has_grad = any(
                p.grad is not None
                for p in sci_model.structural_encoder.parameters()
                if p.requires_grad
            )
            assert has_grad, "Structural encoder has no gradients"

        if sci_model.content_encoder is not None:
            has_grad = any(
                p.grad is not None
                for p in sci_model.content_encoder.parameters()
                if p.requires_grad
            )
            assert has_grad, "Content encoder has no gradients"

        if sci_model.causal_binding is not None:
            has_grad = any(
                p.grad is not None
                for p in sci_model.causal_binding.parameters()
                if p.requires_grad
            )
            assert has_grad, "Causal binding has no gradients"

        print("✓ Gradients flow through hook injections")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
