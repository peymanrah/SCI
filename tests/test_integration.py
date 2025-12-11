"""
Integration Tests for Complete SCI System

These tests verify that all SCI components work together correctly in
end-to-end scenarios, simulating actual training and inference workflows.

Required by: SCI_ENGINEERING_STANDARDS.md Section 2.7
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

# Note: These tests will use simplified configs to avoid loading full model
# Real integration will be tested with actual training scripts


@pytest.fixture
def minimal_sci_config():
    """Create minimal SCI configuration for integration testing."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            d_model=2048,
            num_decoder_layers=22,
            position_encoding=SimpleNamespace(
                type="rotary",
                max_length=512,
                base=10000,
            ),
            structural_encoder=SimpleNamespace(
                enabled=True,
                num_slots=8,
                num_layers=2,  # Reduced for testing
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
                    injection_layers=[1],  # Just one layer for testing
                ),
                gnn=SimpleNamespace(
                    enabled=False,
                ),
            ),
            content_encoder=SimpleNamespace(
                enabled=True,
                num_layers=1,  # Reduced for testing
                d_model=512,
                num_heads=8,
                dim_feedforward=2048,
                dropout=0.1,
                pooling="mean",
                use_orthogonal_projection=False,
            ),
            causal_binding=SimpleNamespace(
                enabled=True,
                d_model=2048,
                num_heads=8,
                dropout=0.1,
                injection_layers=[6, 11, 16],
                use_causal_intervention=True,
                injection_method="gated",
                gate_init=0.1,
            ),
        ),
        training=SimpleNamespace(
            fp16=False,
            batch_size=2,
            learning_rate=2e-5,
            num_epochs=50,
            warmup_steps=1000,
            gradient_clip=1.0,
            seed=42,
        ),
    )
    return config


class TestComponentIntegration:
    """Test integration between SCI components."""

    def test_structural_and_content_encoders_work_together(self, minimal_sci_config):
        """
        Test that structural and content encoders can process same input.

        They should share embeddings and produce complementary representations.
        """
        from sci.models.components.structural_encoder import StructuralEncoder
        from sci.models.components.content_encoder import ContentEncoder

        # Create encoders
        structural_encoder = StructuralEncoder(minimal_sci_config)
        content_encoder = ContentEncoder(minimal_sci_config)

        # Create shared embedding
        vocab_size = 32000
        d_model = minimal_sci_config.model.structural_encoder.d_model
        shared_embedding = nn.Embedding(vocab_size, d_model)

        # Share embeddings
        structural_encoder.embedding = shared_embedding
        content_encoder.embedding = shared_embedding

        # Create test input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            # Process with both encoders
            structural_slots, _, _ = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

            content_repr = content_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        # Check outputs have correct shapes
        assert structural_slots.shape == (batch_size, 8, d_model)
        assert content_repr.shape == (batch_size, d_model)

        print("✓ Structural and content encoders work together")

    def test_cbm_binds_structural_and_content(self, minimal_sci_config):
        """
        Test that CBM can bind structural and content representations.
        """
        from sci.models.components.causal_binding import CausalBindingMechanism

        cbm = CausalBindingMechanism(minimal_sci_config)

        batch_size = 2
        num_slots = 8
        # Use encoder d_model (512) for inputs, CBM projects to decoder d_model (2048)
        encoder_d_model = minimal_sci_config.model.structural_encoder.d_model  # 512
        decoder_d_model = minimal_sci_config.model.causal_binding.d_model  # 2048

        # Create dummy representations at ENCODER dimension
        # CBM will project these to decoder dimension internally
        structural_slots = torch.randn(batch_size, num_slots, encoder_d_model)
        content_repr = torch.randn(batch_size, encoder_d_model)

        with torch.no_grad():
            # Bind
            bound_repr, _ = cbm.bind(
                structural_slots=structural_slots,
                content_repr=content_repr,
            )

            # Broadcast
            seq_len = 32
            broadcast_repr = cbm.broadcast(
                bound_slots=bound_repr,
                seq_len=seq_len,
            )

        # Check final shape ready for injection (at decoder dimension)
        assert broadcast_repr.shape == (batch_size, seq_len, decoder_d_model)

        print("✓ CBM binds and broadcasts representations")

    def test_end_to_end_forward_pass_simulation(self, minimal_sci_config):
        """
        CRITICAL TEST: Simulate end-to-end forward pass with all components.

        This tests the complete SCI pipeline without loading full TinyLlama.
        """
        from sci.models.components.structural_encoder import StructuralEncoder
        from sci.models.components.content_encoder import ContentEncoder
        from sci.models.components.causal_binding import CausalBindingMechanism

        # Create all components
        structural_encoder = StructuralEncoder(minimal_sci_config)
        content_encoder = ContentEncoder(minimal_sci_config)
        cbm = CausalBindingMechanism(minimal_sci_config)

        # Create shared embedding
        vocab_size = 32000
        d_model_enc = minimal_sci_config.model.structural_encoder.d_model
        shared_embedding = nn.Embedding(vocab_size, d_model_enc)

        structural_encoder.embedding = shared_embedding
        content_encoder.embedding = shared_embedding

        # Create test input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Simulate instruction mask (first half is instruction)
        instruction_mask = torch.zeros(batch_size, seq_len)
        instruction_mask[:, :16] = 1

        with torch.no_grad():
            # STEP 1: Structural Encoding
            structural_slots, structural_scores, edge_weights = structural_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

            # STEP 2: Content Encoding
            content_repr = content_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

            # STEP 3: Causal Binding (now with edge_weights for causal intervention!)
            bound_repr, _ = cbm.bind(
                structural_slots=structural_slots,
                content_repr=content_repr,
                edge_weights=edge_weights,  # Enable causal intervention
            )

            # STEP 4: Broadcast
            broadcast_repr = cbm.broadcast(
                bound_slots=bound_repr,
                seq_len=seq_len,
            )

            # STEP 5: Simulate Injection (into dummy decoder hidden states)
            d_model_dec = 2048
            decoder_hidden = torch.randn(batch_size, seq_len, d_model_dec)

            injected_hidden = cbm.inject(
                decoder_hidden=decoder_hidden,
                bound_repr=broadcast_repr,
                layer_idx=cbm.injection_layers[0],
            )

        # Verify all steps completed successfully
        assert structural_slots is not None
        assert content_repr is not None
        assert bound_repr is not None
        assert broadcast_repr is not None
        assert injected_hidden is not None

        # Verify final shape
        assert injected_hidden.shape == (batch_size, seq_len, d_model_dec)

        # Verify injection modified hidden states
        diff = (injected_hidden - decoder_hidden).abs().max().item()
        assert diff > 1e-6, "Injection should modify hidden states"

        print("✓ End-to-end forward pass simulation successful")
        print(f"✓ All components integrated correctly")


class TestDataLeakageIntegration:
    """Test data leakage prevention in integrated system."""

    def test_instruction_mask_propagates_correctly(self, minimal_sci_config):
        """
        CRITICAL TEST: Verify instruction mask prevents data leakage across all components.
        """
        from sci.models.components.structural_encoder import StructuralEncoder
        from sci.models.components.content_encoder import ContentEncoder

        structural_encoder = StructuralEncoder(minimal_sci_config)
        content_encoder = ContentEncoder(minimal_sci_config)

        # Share embeddings
        vocab_size = 32000
        d_model = minimal_sci_config.model.structural_encoder.d_model
        shared_embedding = nn.Embedding(vocab_size, d_model)
        structural_encoder.embedding = shared_embedding
        content_encoder.embedding = shared_embedding

        # Set to eval mode for deterministic behavior (disables dropout)
        structural_encoder.eval()
        content_encoder.eval()

        batch_size = 2
        seq_len = 32

        # Create two inputs with same instruction, different response
        input_ids_1 = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_ids_2 = input_ids_1.clone()

        # Same instruction (first 16 tokens)
        instruction_len = 16
        input_ids_2[:, :instruction_len] = input_ids_1[:, :instruction_len]

        # Different response (last 16 tokens)
        input_ids_2[:, instruction_len:] = torch.randint(0, vocab_size, (batch_size, seq_len - instruction_len))

        attention_mask = torch.ones(batch_size, seq_len)

        # Instruction mask
        instruction_mask = torch.zeros(batch_size, seq_len)
        instruction_mask[:, :instruction_len] = 1

        with torch.no_grad():
            # Encode both inputs
            structural_1, _, _ = structural_encoder(input_ids_1, attention_mask, instruction_mask)
            structural_2, _, _ = structural_encoder(input_ids_2, attention_mask, instruction_mask)

            content_1 = content_encoder(input_ids_1, attention_mask, instruction_mask)
            content_2 = content_encoder(input_ids_2, attention_mask, instruction_mask)

        # Structural representations should be identical (same instruction)
        structural_diff = (structural_1 - structural_2).abs().max().item()
        assert structural_diff < 1e-5, (
            f"Structural encoder sees response tokens (diff={structural_diff:.6f})"
        )

        # Content representations should be identical (same instruction)
        content_diff = (content_1 - content_2).abs().max().item()
        assert content_diff < 1e-5, (
            f"Content encoder sees response tokens (diff={content_diff:.6f})"
        )

        print(f"✓ Data leakage prevention verified (structural diff: {structural_diff:.2e})")
        print(f"✓ Content encoder also leakage-free (content diff: {content_diff:.2e})")


class TestGradientFlow:
    """Test gradient flow through integrated system."""

    def test_gradients_flow_through_all_components(self, minimal_sci_config):
        """
        CRITICAL TEST: Verify gradients flow through complete pipeline.

        Essential for training.
        """
        from sci.models.components.structural_encoder import StructuralEncoder
        from sci.models.components.content_encoder import ContentEncoder
        from sci.models.components.causal_binding import CausalBindingMechanism

        # Create components in training mode
        structural_encoder = StructuralEncoder(minimal_sci_config).train()
        content_encoder = ContentEncoder(minimal_sci_config).train()
        cbm = CausalBindingMechanism(minimal_sci_config).train()

        # Share embeddings
        vocab_size = 32000
        d_model_enc = minimal_sci_config.model.structural_encoder.d_model
        shared_embedding = nn.Embedding(vocab_size, d_model_enc)
        structural_encoder.embedding = shared_embedding
        content_encoder.embedding = shared_embedding

        # Create input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        # Forward pass with gradient tracking
        structural_slots, _, _ = structural_encoder(input_ids, attention_mask, instruction_mask)
        content_repr = content_encoder(input_ids, attention_mask, instruction_mask)
        bound_repr, _ = cbm.bind(structural_slots, content_repr)

        # Compute dummy loss
        loss = bound_repr.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert shared_embedding.weight.grad is not None, "Shared embedding has no gradients"

        has_structural_grad = any(
            p.grad is not None for p in structural_encoder.parameters() if p.requires_grad
        )
        assert has_structural_grad, "Structural encoder has no gradients"

        has_content_grad = any(
            p.grad is not None for p in content_encoder.parameters() if p.requires_grad
        )
        assert has_content_grad, "Content encoder has no gradients"

        has_cbm_grad = any(
            p.grad is not None for p in cbm.parameters() if p.requires_grad
        )
        assert has_cbm_grad, "CBM has no gradients"

        print("✓ Gradients flow through all components")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
