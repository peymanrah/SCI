"""
Tests for loss functions.

CRITICAL: SCL loss must only compute when positive pairs exist.
"""

import pytest
import torch

from sci.models.losses.scl_loss import StructuralContrastiveLoss, SimplifiedSCLLoss
from sci.models.losses.combined_loss import SCICombinedLoss


class TestStructuralContrastiveLoss:
    """Test SCL loss computation."""

    @pytest.fixture
    def scl_loss(self):
        return StructuralContrastiveLoss(temperature=0.07, lambda_weight=0.3)

    def test_loss_with_positive_pairs(self, scl_loss):
        """Test SCL loss with positive pairs."""
        batch_size = 8
        d_model = 512

        # Create structural representations
        struct_repr = torch.randn(batch_size, d_model)

        # Create pair labels (0,1 same; 2,3 same; etc.)
        pair_labels = torch.zeros(batch_size, batch_size)
        for i in range(0, batch_size, 2):
            if i + 1 < batch_size:
                pair_labels[i, i+1] = 1
                pair_labels[i+1, i] = 1

        loss = scl_loss(struct_repr, struct_repr, pair_labels)

        assert loss.requires_grad
        assert loss.item() > 0

    def test_loss_with_no_positive_pairs(self, scl_loss):
        """Test that loss is zero when no positive pairs."""
        batch_size = 4
        d_model = 512

        struct_repr = torch.randn(batch_size, d_model)
        pair_labels = torch.zeros(batch_size, batch_size)  # No positive pairs

        loss = scl_loss(struct_repr, struct_repr, pair_labels)

        assert loss.item() == 0.0

    def test_similar_pairs_have_lower_loss(self, scl_loss):
        """Test that similar positive pairs have lower loss."""
        batch_size = 4
        d_model = 512

        # Create very similar representations
        base_repr = torch.randn(1, d_model)
        similar_repr = base_repr.repeat(batch_size, 1) + 0.01 * torch.randn(batch_size, d_model)

        # Create dissimilar representations
        dissimilar_repr = torch.randn(batch_size, d_model)

        # Same pair labels
        pair_labels = torch.zeros(batch_size, batch_size)
        pair_labels[0, 1] = 1
        pair_labels[1, 0] = 1
        pair_labels[2, 3] = 1
        pair_labels[3, 2] = 1

        loss_similar = scl_loss(similar_repr, similar_repr, pair_labels)
        loss_dissimilar = scl_loss(dissimilar_repr, dissimilar_repr, pair_labels)

        assert loss_similar < loss_dissimilar

    def test_works_with_slot_representations(self, scl_loss):
        """Test SCL with 3D slot representations."""
        batch_size = 4
        num_slots = 8
        d_model = 512

        struct_repr = torch.randn(batch_size, num_slots, d_model)

        pair_labels = torch.zeros(batch_size, batch_size)
        pair_labels[0, 1] = 1
        pair_labels[1, 0] = 1

        loss = scl_loss(struct_repr, struct_repr, pair_labels)

        assert loss.requires_grad
        assert loss.item() >= 0


class TestCombinedLoss:
    """Test combined loss function."""

    @pytest.fixture
    def combined_loss(self):
        return SCICombinedLoss(
            scl_weight=0.3,
            ortho_weight=0.01,
            temperature=0.07,
        )

    def test_all_loss_components(self, combined_loss):
        """Test that all loss components are computed."""
        batch_size = 4
        num_slots = 8
        d_model = 512
        seq_len = 20
        vocab_size = 1000

        # Create model outputs
        model_outputs = {
            'logits': torch.randn(batch_size, seq_len, vocab_size),
            'loss': torch.tensor(2.5),  # LM loss from model
            'structural_slots': torch.randn(batch_size, num_slots, d_model),
            'content_repr': torch.randn(batch_size, d_model),
        }

        # Create pair labels
        pair_labels = torch.zeros(batch_size, batch_size)
        pair_labels[0, 1] = 1
        pair_labels[1, 0] = 1

        losses = combined_loss(model_outputs, pair_labels)

        # Check all components exist
        assert 'total_loss' in losses
        assert 'lm_loss' in losses
        assert 'scl_loss' in losses
        assert 'orthogonality_loss' in losses
        assert 'num_positive_pairs' in losses

        # Check values are reasonable
        assert losses['total_loss'].requires_grad
        assert losses['lm_loss'].item() > 0
        assert losses['scl_loss'].item() >= 0
        assert losses['orthogonality_loss'].item() >= 0
        assert losses['num_positive_pairs'] > 0

    def test_scl_warmup_override(self, combined_loss):
        """Test SCL weight warmup override."""
        model_outputs = {
            'logits': torch.randn(2, 10, 100),
            'loss': torch.tensor(1.0),
            'structural_slots': torch.randn(2, 8, 512),
            'content_repr': torch.randn(2, 512),
        }

        pair_labels = torch.zeros(2, 2)
        pair_labels[0, 1] = 1
        pair_labels[1, 0] = 1

        # Test with different warmup values
        losses_full = combined_loss(model_outputs, pair_labels, scl_weight_override=0.3)
        losses_warmup = combined_loss(model_outputs, pair_labels, scl_weight_override=0.1)

        assert losses_warmup['scl_weight_used'] == 0.1
        assert losses_full['scl_weight_used'] == 0.3

    def test_no_scl_when_no_positive_pairs(self, combined_loss):
        """Test that SCL loss is zero when no positive pairs."""
        model_outputs = {
            'logits': torch.randn(4, 10, 100),
            'loss': torch.tensor(2.0),
            'structural_slots': torch.randn(4, 8, 512),
            'content_repr': torch.randn(4, 512),
        }

        pair_labels = torch.zeros(4, 4)  # No positive pairs

        losses = combined_loss(model_outputs, pair_labels)

        assert losses['scl_loss'].item() == 0.0
        assert losses['num_positive_pairs'] == 0

    def test_orthogonality_loss_computation(self, combined_loss):
        """Test orthogonality loss computation."""
        batch_size = 4
        d_model = 512

        # Create orthogonal representations
        content = torch.randn(batch_size, d_model)
        structure = torch.randn(batch_size, 8, d_model)

        ortho_loss = combined_loss.compute_orthogonality_loss(content, structure)

        assert ortho_loss.requires_grad
        assert 0 <= ortho_loss.item() <= 1  # Cosine similarity range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
