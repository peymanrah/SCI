"""
Tests for AbstractionLayer - THE KEY INNOVATION.

CRITICAL: AbstractionLayer must produce structuralness scores in [0, 1].
"""

import pytest
import torch

from sci.models.components.abstraction_layer import AbstractionLayer, MultiHeadAbstractionLayer


class TestAbstractionLayer:
    """Test AbstractionLayer component."""

    @pytest.fixture
    def abstraction_layer(self):
        """Create AbstractionLayer for testing."""
        return AbstractionLayer(d_model=512, hidden_multiplier=2)

    def test_initialization(self, abstraction_layer):
        """Test layer initialization."""
        assert abstraction_layer.d_model == 512
        assert abstraction_layer.hidden_dim == 1024
        assert hasattr(abstraction_layer, 'structural_detector')
        assert hasattr(abstraction_layer, 'residual_gate')
        assert hasattr(abstraction_layer, 'layer_norm')

    def test_forward_shape(self, abstraction_layer):
        """Test forward pass output shapes."""
        batch_size = 4
        seq_len = 20
        d_model = 512

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)

        output, scores = abstraction_layer(hidden_states, attention_mask)

        assert output.shape == (batch_size, seq_len, d_model)
        assert scores.shape == (batch_size, seq_len, d_model)

    def test_structuralness_scores_range(self, abstraction_layer):
        """CRITICAL: Scores must be in [0, 1]."""
        batch_size = 4
        seq_len = 20
        d_model = 512

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)

        _, scores = abstraction_layer(hidden_states, attention_mask)

        # Check range
        assert scores.min() >= 0.0, f"Min score {scores.min()} < 0"
        assert scores.max() <= 1.0, f"Max score {scores.max()} > 1"

    def test_attention_mask_application(self, abstraction_layer):
        """Test that attention mask zeros out padding tokens."""
        batch_size = 2
        seq_len = 10
        d_model = 512

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 5:] = 0  # Mask last 5 tokens

        output, scores = abstraction_layer(hidden_states, attention_mask)

        # Check that masked positions have zero scores
        assert (scores[:, 5:, :] == 0).all(), "Masked positions should have zero scores"

    def test_gradient_flow(self, abstraction_layer):
        """Test that gradients flow properly."""
        hidden_states = torch.randn(2, 10, 512, requires_grad=True)
        attention_mask = torch.ones(2, 10)

        output, _ = abstraction_layer(hidden_states, attention_mask)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.abs().sum() > 0

    def test_statistics_computation(self, abstraction_layer):
        """Test structural statistics computation."""
        batch_size = 4
        seq_len = 20
        d_model = 512

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)

        _, scores = abstraction_layer(hidden_states, attention_mask)
        stats = abstraction_layer.get_structural_statistics(scores, attention_mask)

        assert 'mean_score' in stats
        assert 'high_structural_ratio' in stats
        assert 'low_structural_ratio' in stats
        assert 'residual_gate' in stats

        assert 0 <= stats['mean_score'] <= 1
        assert 0 <= stats['high_structural_ratio'] <= 1
        assert 0 <= stats['low_structural_ratio'] <= 1


class TestMultiHeadAbstractionLayer:
    """Test multi-head variant."""

    @pytest.fixture
    def multihead_layer(self):
        """Create multi-head AbstractionLayer."""
        return MultiHeadAbstractionLayer(d_model=512, num_heads=4)

    def test_initialization(self, multihead_layer):
        """Test multi-head initialization."""
        assert multihead_layer.d_model == 512
        assert multihead_layer.num_heads == 4
        assert multihead_layer.head_dim == 128
        assert len(multihead_layer.abstraction_heads) == 4

    def test_forward_shape(self, multihead_layer):
        """Test forward pass shapes."""
        batch_size = 2
        seq_len = 10
        d_model = 512

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)

        output, scores = multihead_layer(hidden_states, attention_mask)

        assert output.shape == (batch_size, seq_len, d_model)
        # scores: [batch, seq_len, head_dim, num_heads]
        assert scores.shape[0] == batch_size
        assert scores.shape[1] == seq_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
