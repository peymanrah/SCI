"""
Tests for Content Encoder

The Content Encoder extracts content information (semantic meaning) independently
of structural patterns. It must be orthogonal to the Structural Encoder and
share embeddings with the base model.

Required by: SCI_ENGINEERING_STANDARDS.md Section 2.3
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from sci.models.components.content_encoder import ContentEncoder, compute_orthogonality_loss


@pytest.fixture
def content_encoder_config():
    """Create configuration for content encoder testing."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            d_model=2048,
            base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            content_encoder=SimpleNamespace(
                enabled=True,
                num_layers=2,
                d_model=512,
                num_heads=8,
                dim_feedforward=2048,
                dropout=0.1,
                pooling="mean",
                use_orthogonal_projection=False,
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
def content_encoder(content_encoder_config):
    """Create content encoder with shared embedding for testing."""
    encoder = ContentEncoder(content_encoder_config)

    # Create dummy shared embedding
    vocab_size = 32000
    d_model = content_encoder_config.model.content_encoder.d_model
    encoder.embedding = nn.Embedding(vocab_size, d_model)

    return encoder


class TestContentEncoderArchitecture:
    """Test content encoder architecture and components."""

    def test_output_shape(self, content_encoder, content_encoder_config):
        """
        CRITICAL TEST: Verify content encoder outputs correct shape.

        Expected: [batch_size, d_model] (sequence-level representation)
        """
        batch_size = 4
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        instruction_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            content_repr = content_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        d_model = content_encoder_config.model.content_encoder.d_model
        expected_shape = (batch_size, d_model)

        assert content_repr.shape == expected_shape, (
            f"Content repr shape {content_repr.shape} != {expected_shape}"
        )

        print(f"✓ Content encoder output shape correct: {content_repr.shape}")

    def test_orthogonality_to_structure(self, content_encoder):
        """
        CRITICAL TEST: Verify content representations can be orthogonal to structural representations.

        This is the core SCI factorization: structure ⊥ content.
        """
        batch_size = 4
        seq_len = 32
        vocab_size = 32000
        d_model = 512

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            content_repr = content_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Create dummy structural representation
            structural_repr = torch.randn(batch_size, 8, d_model)

            # Compute orthogonality loss
            ortho_loss = compute_orthogonality_loss(
                structural_repr=structural_repr,
                content_repr=content_repr,
                lambda_ortho=1.0,
            )

        # Verify loss is computed correctly
        assert isinstance(ortho_loss, torch.Tensor), "Orthogonality loss should be tensor"
        assert ortho_loss.dim() == 0, "Orthogonality loss should be scalar"
        assert ortho_loss.item() >= 0, f"Orthogonality loss should be non-negative: {ortho_loss.item()}"

        # Test that orthogonal vectors give lower loss
        structural_flat = structural_repr.mean(dim=1)  # [batch, d_model]

        # Create orthogonal content via Gram-Schmidt
        content_normalized = F.normalize(content_repr, dim=-1)
        structural_normalized = F.normalize(structural_flat, dim=-1)

        # Project out structural component
        projection = (content_normalized * structural_normalized).sum(dim=-1, keepdim=True)
        content_orthogonal = content_normalized - projection * structural_normalized
        content_orthogonal = F.normalize(content_orthogonal, dim=-1) * torch.norm(content_repr, dim=-1, keepdim=True)

        ortho_loss_orthogonal = compute_orthogonality_loss(
            structural_repr=structural_repr,
            content_repr=content_orthogonal,
            lambda_ortho=1.0,
        )

        # Orthogonal content should have lower loss
        assert ortho_loss_orthogonal < ortho_loss, (
            f"Orthogonal content loss ({ortho_loss_orthogonal:.6f}) should be < "
            f"random content loss ({ortho_loss:.6f})"
        )

        print(f"✓ Orthogonality verified: random={ortho_loss:.6f}, orthogonal={ortho_loss_orthogonal:.6f}")

    def test_shared_embeddings(self, content_encoder):
        """
        CRITICAL TEST: Verify content encoder uses shared embeddings.

        Sharing embeddings with structural encoder and base model is critical
        for SCI architecture.
        """
        # Check that embedding is set
        assert content_encoder.embedding is not None, (
            "Content encoder embedding is None - shared embeddings not set!"
        )

        # Check that embedding is an nn.Embedding
        assert isinstance(content_encoder.embedding, nn.Embedding), (
            f"Embedding should be nn.Embedding, got {type(content_encoder.embedding)}"
        )

        # Check embedding shape
        vocab_size, d_model = content_encoder.embedding.weight.shape
        expected_d_model = content_encoder.d_model

        assert d_model == expected_d_model, (
            f"Embedding dimension {d_model} != content encoder d_model {expected_d_model}"
        )

        # Test that modifying embedding affects output
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            content_repr_1 = content_encoder(input_ids, attention_mask)

        # Modify shared embedding
        original_weight = content_encoder.embedding.weight.data.clone()
        content_encoder.embedding.weight.data += 1.0  # Add constant offset

        with torch.no_grad():
            content_repr_2 = content_encoder(input_ids, attention_mask)

        # Restore original
        content_encoder.embedding.weight.data = original_weight

        # Output should be different when embedding changes
        diff = (content_repr_1 - content_repr_2).abs().max().item()

        assert diff > 1e-6, (
            f"Changing embedding didn't affect output (diff={diff:.2e}). "
            "Embedding may not be properly shared!"
        )

        print(f"✓ Shared embeddings verified (output diff: {diff:.6f})")


class TestContentEncoderFunctionality:
    """Test content encoder functionality."""

    def test_data_leakage_prevention(self, content_encoder):
        """
        CRITICAL TEST: Verify content encoder only sees instruction tokens.

        This prevents data leakage from response tokens.
        """
        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        # Create instruction mask (first half is instruction)
        instruction_len = 16
        instruction_mask = torch.zeros(batch_size, seq_len)
        instruction_mask[:, :instruction_len] = 1

        # Create input with same instruction, different response
        input_ids_1 = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_ids_2 = input_ids_1.clone()

        # Same instruction
        input_ids_2[:, :instruction_len] = input_ids_1[:, :instruction_len]

        # Different response
        input_ids_2[:, instruction_len:] = torch.randint(0, vocab_size, (batch_size, seq_len - instruction_len))

        attention_mask = torch.ones(batch_size, seq_len)

        content_encoder.eval()  # Disable dropout for deterministic behavior
        with torch.no_grad():
            content_repr_1 = content_encoder(
                input_ids=input_ids_1,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

            content_repr_2 = content_encoder(
                input_ids=input_ids_2,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )

        # Content representations should be identical (same instruction)
        diff = (content_repr_1 - content_repr_2).abs().max().item()

        assert diff < 1e-5, (
            f"Content representations differ by {diff:.6f} despite identical instructions. "
            "This suggests content encoder is seeing response tokens (DATA LEAKAGE)!"
        )

        print(f"✓ Data leakage prevention verified (diff: {diff:.2e})")

    def test_pooling_methods(self, content_encoder_config):
        """
        Test that different pooling methods work correctly.
        """
        batch_size = 4
        seq_len = 32
        vocab_size = 32000

        for pooling_method in ["mean", "max"]:
            # Create encoder with specific pooling
            config = content_encoder_config
            config.model.content_encoder.pooling = pooling_method

            encoder = ContentEncoder(config)
            encoder.embedding = nn.Embedding(vocab_size, config.model.content_encoder.d_model)

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)

            with torch.no_grad():
                try:
                    content_repr = encoder(input_ids, attention_mask)

                    # Check output shape
                    expected_shape = (batch_size, config.model.content_encoder.d_model)
                    assert content_repr.shape == expected_shape, (
                        f"Pooling {pooling_method}: shape {content_repr.shape} != {expected_shape}"
                    )

                except Exception as e:
                    pytest.fail(f"Pooling method '{pooling_method}' failed: {e}")

        print("✓ Both pooling methods (mean, max) work correctly")

    def test_gradient_flow(self, content_encoder):
        """
        Test that gradients flow through content encoder.

        Critical for training.
        """
        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass with gradient tracking
        content_repr = content_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Compute dummy loss
        loss = content_repr.sum()

        # Backward pass
        loss.backward()

        # Check that encoder layers have gradients
        has_grad = False
        for param in content_encoder.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break

        assert has_grad, "Content encoder has no gradients"

        # Check that shared embedding has gradients
        assert content_encoder.embedding.weight.grad is not None, (
            "Shared embedding has no gradients"
        )

        print("✓ Gradients flow through content encoder")

    def test_handles_variable_length_sequences(self, content_encoder):
        """
        Test that content encoder handles variable-length sequences correctly.
        """
        batch_size = 2
        vocab_size = 32000

        for seq_len in [16, 32, 64, 128]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)

            with torch.no_grad():
                try:
                    content_repr = content_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    # Check output shape is consistent
                    expected_shape = (batch_size, content_encoder.d_model)
                    assert content_repr.shape == expected_shape, (
                        f"Seq len {seq_len}: shape {content_repr.shape} != {expected_shape}"
                    )

                except Exception as e:
                    pytest.fail(f"Failed with seq_len={seq_len}: {e}")

        print("✓ Handles variable sequence lengths [16, 32, 64, 128]")

    def test_deterministic_in_eval_mode(self, content_encoder):
        """
        Test that content encoder is deterministic in eval mode.
        """
        content_encoder.eval()

        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Two forward passes with same input
        with torch.no_grad():
            content_repr_1 = content_encoder(input_ids, attention_mask)
            content_repr_2 = content_encoder(input_ids, attention_mask)

        # Outputs should be identical in eval mode
        assert torch.allclose(content_repr_1, content_repr_2, atol=1e-6), (
            "Content encoder is non-deterministic in eval mode"
        )

        print("✓ Content encoder is deterministic in eval mode")

    def test_attention_mask_application(self, content_encoder):
        """
        Test that attention mask correctly masks padding tokens.
        """
        batch_size = 2
        seq_len = 32
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create attention mask with padding in second half
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 16:] = 0  # Second half is padding

        content_encoder.eval()  # Disable dropout for deterministic behavior
        with torch.no_grad():
            content_repr = content_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Modify padding tokens (should not affect output)
        input_ids_modified = input_ids.clone()
        input_ids_modified[:, 16:] = torch.randint(0, vocab_size, (batch_size, seq_len - 16))

        with torch.no_grad():
            content_repr_modified = content_encoder(
                input_ids=input_ids_modified,
                attention_mask=attention_mask,
            )

        # Outputs should be identical (padding was masked)
        diff = (content_repr - content_repr_modified).abs().max().item()

        assert diff < 1e-5, (
            f"Changing padding tokens affected output (diff={diff:.6f}). "
            "Attention mask may not be working correctly!"
        )

        print(f"✓ Attention mask correctly masks padding (diff: {diff:.2e})")

    def test_lightweight_architecture(self, content_encoder, content_encoder_config):
        """
        Verify content encoder is lightweight (only 2 layers).

        Content encoder should be smaller than structural encoder since it
        only refines shared embeddings.
        """
        num_layers = content_encoder_config.model.content_encoder.num_layers

        assert num_layers == 2, (
            f"Content encoder should have 2 layers, got {num_layers}"
        )

        # Check that encoder has correct number of layers
        assert len(content_encoder.layers) == num_layers, (
            f"Expected {num_layers} transformer layers, got {len(content_encoder.layers)}"
        )

        print(f"✓ Content encoder is lightweight ({num_layers} layers)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
