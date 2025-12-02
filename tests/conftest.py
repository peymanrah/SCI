"""
Pytest configuration and fixtures.
"""

import pytest
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def test_config():
    """Load minimal config for testing."""
    from sci.config.config_loader import load_config

    config = load_config("configs/sci_full.yaml")

    # Override for faster testing
    config.model.structural_encoder.num_layers = 2
    config.model.content_encoder.num_layers = 1
    config.training.max_epochs = 2
    config.training.batch_size = 4
    config.logging.use_wandb = False

    return config


@pytest.fixture
def cleanup_test_cache():
    """Cleanup test cache after tests."""
    yield

    # Cleanup
    import shutil
    if os.path.exists(".test_cache"):
        shutil.rmtree(".test_cache")
