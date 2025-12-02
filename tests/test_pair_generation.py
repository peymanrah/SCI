"""
Tests for structural pair generation.

CRITICAL: Pair generation must correctly identify same-structure examples.
"""

import pytest
import torch
import numpy as np

from sci.data.structure_extractors.scan_extractor import SCANStructureExtractor
from sci.data.pair_generators.scan_pair_generator import SCANPairGenerator


class TestSCANStructureExtractor:
    """Test structure extraction from SCAN commands."""

    @pytest.fixture
    def extractor(self):
        return SCANStructureExtractor()

    def test_simple_structure_extraction(self, extractor):
        """Test basic structure extraction."""
        template, content = extractor.extract_structure("walk twice")

        assert template == "ACTION_0 twice"
        assert content == ["walk"]

    def test_multiple_actions(self, extractor):
        """Test extraction with multiple actions."""
        template, content = extractor.extract_structure("walk left and run right")

        assert template == "ACTION_0 left and ACTION_1 right"
        assert content == ["walk", "run"]

    def test_same_structure_detection(self, extractor):
        """Test detection of same structure."""
        # Same structure
        assert extractor.are_same_structure("walk twice", "run twice")
        assert extractor.are_same_structure("walk and run", "jump and look")

        # Different structure
        assert not extractor.are_same_structure("walk twice", "walk left")
        assert not extractor.are_same_structure("walk", "walk twice")

    def test_grouping_by_structure(self, extractor):
        """Test grouping commands by structure."""
        commands = [
            "walk twice",
            "run twice",
            "jump left",
            "look left",
            "walk and run",
        ]

        groups = extractor.group_by_structure(commands)

        # Should have 3 unique structures
        assert len(groups) == 3

        # "walk twice" and "run twice" should be in same group
        template_twice = "ACTION_0 twice"
        assert template_twice in groups
        assert len(groups[template_twice]) == 2

        # "jump left" and "look left" should be in same group
        template_left = "ACTION_0 left"
        assert template_left in groups
        assert len(groups[template_left]) == 2


class TestSCANPairGenerator:
    """Test pair generation and caching."""

    @pytest.fixture
    def commands(self):
        """Sample commands for testing."""
        return [
            "walk twice",      # 0
            "run twice",       # 1 - same as 0
            "jump twice",      # 2 - same as 0,1
            "walk left",       # 3
            "run left",        # 4 - same as 3
            "look right",      # 5
            "jump and walk",   # 6
            "look and run",    # 7 - same as 6
        ]

    @pytest.fixture
    def generator(self):
        return SCANPairGenerator(cache_dir=".test_cache/pairs")

    def test_pair_matrix_generation(self, generator, commands):
        """Test pair matrix generation."""
        pair_matrix = generator.generate_pairs(
            commands,
            cache_name="test_pairs",
            force_regenerate=True,
        )

        # Check shape
        assert pair_matrix.shape == (len(commands), len(commands))

        # Check diagonal is zero (no self-pairs)
        assert (np.diag(pair_matrix) == 0).all()

        # Check symmetry
        assert (pair_matrix == pair_matrix.T).all()

        # Check specific pairs
        assert pair_matrix[0, 1] == 1, "walk twice and run twice should be positive"
        assert pair_matrix[0, 2] == 1, "walk twice and jump twice should be positive"
        assert pair_matrix[1, 2] == 1, "run twice and jump twice should be positive"
        assert pair_matrix[3, 4] == 1, "walk left and run left should be positive"
        assert pair_matrix[6, 7] == 1, "jump and walk / look and run should be positive"

        # Check negative pairs
        assert pair_matrix[0, 3] == 0, "walk twice and walk left should be negative"
        assert pair_matrix[0, 5] == 0, "walk twice and look right should be negative"

    def test_batch_pair_labels(self, generator, commands):
        """Test batch pair label extraction."""
        # Generate pairs first
        generator.generate_pairs(commands, cache_name="test_pairs", force_regenerate=True)

        # Get batch pair labels
        batch_indices = [0, 1, 3, 4]  # walk twice, run twice, walk left, run left
        pair_labels = generator.get_batch_pair_labels(batch_indices)

        assert pair_labels.shape == (4, 4)
        assert isinstance(pair_labels, torch.Tensor)

        # Check specific pairs in batch
        assert pair_labels[0, 1] == 1, "walk twice and run twice"
        assert pair_labels[2, 3] == 1, "walk left and run left"
        assert pair_labels[0, 2] == 0, "different structures"

    def test_pair_caching(self, generator, commands):
        """Test that pairs are cached and reloaded correctly."""
        # Generate pairs
        pair_matrix_1 = generator.generate_pairs(
            commands,
            cache_name="test_cache_pairs",
            force_regenerate=True,
        )

        # Create new generator and load from cache
        generator_2 = SCANPairGenerator(cache_dir=".test_cache/pairs")
        pair_matrix_2 = generator_2.generate_pairs(
            commands,
            cache_name="test_cache_pairs",
            force_regenerate=False,  # Should load from cache
        )

        # Should be identical
        assert np.array_equal(pair_matrix_1, pair_matrix_2)

    def test_structure_statistics(self, generator, commands):
        """Test structure statistics computation."""
        generator.generate_pairs(commands, cache_name="test_stats", force_regenerate=True)

        stats = generator.get_structure_statistics()

        assert 'num_unique_structures' in stats
        assert 'num_positive_pairs' in stats
        assert 'positive_ratio' in stats

        # We have 4 unique structures in test commands
        assert stats['num_unique_structures'] == 4

    def test_balanced_batch_sampling(self, generator, commands):
        """Test balanced batch sampling."""
        generator.generate_pairs(commands, cache_name="test_balanced", force_regenerate=True)

        # Sample balanced batch
        batch_indices = generator.sample_balanced_batch(
            batch_size=6,
            min_structures_per_batch=2,
            min_examples_per_structure=2,
        )

        assert len(batch_indices) == 6

        # Check that we have positive pairs in the batch
        batch_labels = generator.get_batch_pair_labels(batch_indices)
        num_positive = (batch_labels.sum() - batch_labels.diagonal().sum()).item()

        assert num_positive > 0, "Balanced batch should contain positive pairs"

    def teardown_method(self):
        """Clean up test cache."""
        import shutil
        import os
        if os.path.exists(".test_cache"):
            shutil.rmtree(".test_cache")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
