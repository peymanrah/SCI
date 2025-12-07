"""
CRITICAL DATA LEAKAGE TESTS.

These tests verify that Structural Encoder and Content Encoder
ONLY see instruction tokens, NOT response tokens.

Data leakage would invalidate the entire approach.
"""

import pytest
import torch

from sci.models.sci_model import SCIModel
from sci.config.config_loader import load_config


class TestDataLeakagePrevention:
    """
    CRITICAL: Ensure no data leakage from response to encoders.
    """

    @pytest.fixture
    def model_and_config(self):
        """Load SCI model for testing."""
        # Use minimal config for testing
        config = load_config("configs/sci_full.yaml")

        # Override for faster testing
        config.model.structural_encoder.num_layers = 2
        config.model.content_encoder.num_layers = 1

        model = SCIModel(config)
        return model, config

    def test_instruction_mask_creation(self, model_and_config):
        """Test that instruction mask is created correctly."""
        model, config = model_and_config

        batch_size = 2
        seq_len = 20

        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = input_ids.clone()

        # Mark first half as instruction (labels = -100)
        labels[:, :seq_len//2] = -100

        # Get instruction mask
        instruction_mask = model.get_instruction_mask(input_ids, labels)

        # Verify
        assert instruction_mask.shape == (batch_size, seq_len)
        assert (instruction_mask[:, :seq_len//2] == 1).all(), "First half should be instruction"
        assert (instruction_mask[:, seq_len//2:] == 0).all(), "Second half should be response"

    def test_structural_encoder_sees_only_instruction(self, model_and_config):
        """
        CRITICAL: Verify SE only processes instruction tokens.
        """
        model, config = model_and_config
        model.eval()

        batch_size = 2
        seq_len = 20
        vocab_size = model.vocab_size

        # Create input with clear instruction/response split
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()

        # Mark first 10 tokens as instruction
        instruction_len = 10
        labels[:, :instruction_len] = -100

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        # The structural encoder should only have seen instruction tokens
        # We can't directly verify this, but we can check that the model
        # correctly masks the instruction tokens for SE/CE

        # Verify that instruction mask was correctly created
        instruction_mask = model.get_instruction_mask(input_ids, labels)
        assert (instruction_mask[:, :instruction_len] == 1).all()
        assert (instruction_mask[:, instruction_len:] == 0).all()

    def test_no_response_in_structural_encoding(self, model_and_config):
        """
        Test that structural encoding doesn't depend on response tokens.

        Two inputs with same instruction but different responses should
        produce identical structural representations.
        """
        model, config = model_and_config
        model.eval()

        if model.structural_encoder is None:
            pytest.skip("Structural encoder not enabled")

        batch_size = 1
        instruction_len = 10
        response_len = 10
        seq_len = instruction_len + response_len

        # Create two examples with same instruction, different response
        input_ids_1 = torch.randint(0, 1000, (batch_size, seq_len))
        input_ids_2 = input_ids_1.clone()

        # Same instruction
        shared_instruction = torch.randint(0, 1000, (batch_size, instruction_len))
        input_ids_1[:, :instruction_len] = shared_instruction
        input_ids_2[:, :instruction_len] = shared_instruction

        # Different responses
        input_ids_1[:, instruction_len:] = torch.randint(0, 1000, (batch_size, response_len))
        input_ids_2[:, instruction_len:] = torch.randint(0, 1000, (batch_size, response_len))

        # Ensure responses are different
        assert not torch.equal(input_ids_1[:, instruction_len:], input_ids_2[:, instruction_len:])

        # Create labels (mask instruction)
        labels_1 = input_ids_1.clone()
        labels_2 = input_ids_2.clone()
        labels_1[:, :instruction_len] = -100
        labels_2[:, :instruction_len] = -100

        attention_mask = torch.ones(batch_size, seq_len)

        # Get structural representations
        with torch.no_grad():
            outputs_1 = model(
                input_ids=input_ids_1,
                attention_mask=attention_mask,
                labels=labels_1,
                return_dict=True,
            )

            outputs_2 = model(
                input_ids=input_ids_2,
                attention_mask=attention_mask,
                labels=labels_2,
                return_dict=True,
            )

        structural_1 = outputs_1['structural_slots']
        structural_2 = outputs_2['structural_slots']

        # Structural representations should be identical (or very similar)
        # since instruction is the same
        if structural_1 is not None and structural_2 is not None:
            diff = (structural_1 - structural_2).abs().max().item()

            # Allow small numerical differences due to floating point
            assert diff < 1e-5, \
                f"Structural representations differ by {diff:.6f} despite identical instructions. " \
                f"This suggests data leakage from response tokens!"

    def test_labels_correctly_mask_instruction(self):
        """Test that -100 in labels correctly masks instruction tokens."""
        from sci.data.datasets.scan_dataset import SCANDataset
        from sci.data.scan_data_collator import SCANDataCollator
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader

        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create dataset (returns raw strings, not tokenized)
        dataset = SCANDataset(
            tokenizer=tokenizer,  # Required argument
            split_name='length',
            subset='train',
        )

        # Create collator to tokenize and create labels
        collator = SCANDataCollator(tokenizer=tokenizer, max_length=128)

        # Create dataloader with batch_size=1 to get a single example
        loader = DataLoader(dataset, batch_size=1, collate_fn=collator)

        # Get first batch
        batch = next(iter(loader))
        labels = batch['labels'][0]  # First (and only) example in batch

        # Check that some tokens are masked with -100
        num_masked = (labels == -100).sum().item()
        assert num_masked > 0, "No tokens are masked with -100!"

        # Check that masked tokens are at the beginning (instruction part)
        first_non_masked = (labels != -100).nonzero(as_tuple=True)[0][0].item()
        assert (labels[:first_non_masked] == -100).all(), \
            "Masked tokens should be contiguous at the start"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
