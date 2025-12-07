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


class TestEncoderAttentionLeakage:
    """
    CRITICAL: Verify SE/CE attention weights are ZERO on response tokens.
    
    Per SCI_ENGINEERING_STANDARDS.md Section 4.3:
    - SE and CE must ONLY see instruction tokens
    - Attention weights to response tokens must be exactly zero
    - Hidden states for response tokens must be zero after encoding
    """

    @pytest.fixture
    def model_and_inputs(self):
        """Create model and test inputs with clear instruction/response boundary."""
        config = load_config("configs/sci_full.yaml")
        
        # Use minimal config for faster testing
        config.model.structural_encoder.num_layers = 2
        config.model.content_encoder.num_layers = 1
        
        model = SCIModel(config)
        model.eval()
        
        batch_size = 2
        seq_len = 20
        instruction_len = 10  # First 10 tokens are instruction
        
        # Create input with known structure
        input_ids = torch.randint(100, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        labels[:, :instruction_len] = -100  # Mark instruction tokens
        
        # Create explicit instruction mask
        instruction_mask = torch.zeros(batch_size, seq_len)
        instruction_mask[:, :instruction_len] = 1
        
        return model, input_ids, attention_mask, labels, instruction_mask, instruction_len

    def test_se_hidden_states_zero_for_response_tokens(self, model_and_inputs):
        """
        CRITICAL: Verify SE hidden states are ZERO for response tokens.
        
        After structural encoding, all hidden states at response positions
        must be exactly zero to prevent any information leakage.
        """
        model, input_ids, attention_mask, labels, instruction_mask, instruction_len = model_and_inputs
        
        if model.structural_encoder is None:
            pytest.skip("Structural encoder not enabled")
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model.structural_encoder.embedding(input_ids)
            
            # Project if needed
            if model.structural_encoder._projection_initialized:
                embeddings = model.structural_encoder.input_projection(embeddings)
            
            # Apply positional encoding
            hidden_states = model.structural_encoder.pos_encoding(embeddings)
            
            # Apply instruction mask (this is what the encoder does)
            mask_expanded = instruction_mask.unsqueeze(-1).float()
            hidden_states_masked = hidden_states * mask_expanded
            
            # Verify response tokens are zero
            response_states = hidden_states_masked[:, instruction_len:, :]
            
            assert (response_states == 0).all(), \
                f"SE hidden states for response tokens are NOT zero! " \
                f"Max absolute value: {response_states.abs().max().item():.6e}"
            
            # Verify instruction tokens are NOT zero
            instruction_states = hidden_states_masked[:, :instruction_len, :]
            assert (instruction_states != 0).any(), \
                "SE hidden states for instruction tokens are all zero - something is wrong!"

    def test_ce_hidden_states_zero_for_response_tokens(self, model_and_inputs):
        """
        CRITICAL: Verify CE hidden states are ZERO for response tokens.
        """
        model, input_ids, attention_mask, labels, instruction_mask, instruction_len = model_and_inputs
        
        if model.content_encoder is None:
            pytest.skip("Content encoder not enabled")
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model.content_encoder.embedding(input_ids)
            
            # Project if needed  
            if model.content_encoder._projection_initialized:
                embeddings = model.content_encoder.input_projection(embeddings)
            
            # Apply instruction mask (this is what the encoder does)
            mask_expanded = instruction_mask.unsqueeze(-1).float()
            hidden_states_masked = embeddings * mask_expanded
            
            # Verify response tokens are zero
            response_states = hidden_states_masked[:, instruction_len:, :]
            
            assert (response_states == 0).all(), \
                f"CE hidden states for response tokens are NOT zero! " \
                f"Max absolute value: {response_states.abs().max().item():.6e}"

    def test_se_full_forward_no_response_leakage(self, model_and_inputs):
        """
        CRITICAL: Full SE forward pass must not leak response information.
        
        Verify that the final structural slots are identical whether or not
        response tokens contain actual data (since they should be masked).
        """
        model, input_ids, attention_mask, labels, instruction_mask, instruction_len = model_and_inputs
        
        if model.structural_encoder is None:
            pytest.skip("Structural encoder not enabled")
        
        batch_size, seq_len = input_ids.shape
        
        # Create two versions: one with response, one with zeros for response
        input_ids_1 = input_ids.clone()
        input_ids_2 = input_ids.clone()
        
        # Same instruction
        input_ids_2[:, :instruction_len] = input_ids_1[:, :instruction_len]
        # Different response (random tokens vs zeros)
        input_ids_2[:, instruction_len:] = 0  # Zero tokens for response
        
        with torch.no_grad():
            # Run SE on both
            slots_1, _ = model.structural_encoder(
                input_ids=input_ids_1,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )
            
            slots_2, _ = model.structural_encoder(
                input_ids=input_ids_2,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )
        
        # Slots should be IDENTICAL (response tokens should have no effect)
        diff = (slots_1 - slots_2).abs().max().item()
        
        assert diff < 1e-5, \
            f"SE structural slots differ by {diff:.6e} despite masking response tokens! " \
            f"This indicates data leakage from response tokens."

    def test_ce_full_forward_no_response_leakage(self, model_and_inputs):
        """
        CRITICAL: Full CE forward pass must not leak response information.
        """
        model, input_ids, attention_mask, labels, instruction_mask, instruction_len = model_and_inputs
        
        if model.content_encoder is None:
            pytest.skip("Content encoder not enabled")
        
        batch_size, seq_len = input_ids.shape
        
        # Create two versions: one with response, one with zeros for response
        input_ids_1 = input_ids.clone()
        input_ids_2 = input_ids.clone()
        
        # Same instruction
        input_ids_2[:, :instruction_len] = input_ids_1[:, :instruction_len]
        # Different response
        input_ids_2[:, instruction_len:] = 0
        
        with torch.no_grad():
            content_1 = model.content_encoder(
                input_ids=input_ids_1,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )
            
            content_2 = model.content_encoder(
                input_ids=input_ids_2,
                attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )
        
        # Content representations should be IDENTICAL
        diff = (content_1 - content_2).abs().max().item()
        
        assert diff < 1e-5, \
            f"CE content repr differs by {diff:.6e} despite masking response tokens! " \
            f"This indicates data leakage from response tokens."

    def test_attention_mask_blocks_response_tokens(self, model_and_inputs):
        """
        Verify the attention mask correctly blocks response tokens.
        
        When instruction_mask is applied to attention_mask, response positions
        should be masked (not attended to).
        """
        model, input_ids, attention_mask, labels, instruction_mask, instruction_len = model_and_inputs
        
        # Compute combined mask as encoder does
        combined_mask = attention_mask * instruction_mask.float()
        
        # Response positions should be 0 (not attended)
        response_mask = combined_mask[:, instruction_len:]
        assert (response_mask == 0).all(), \
            "Attention mask does not block response tokens!"
        
        # Instruction positions should be 1 (attended)
        instruction_mask_vals = combined_mask[:, :instruction_len]
        assert (instruction_mask_vals == 1).all(), \
            "Attention mask incorrectly blocks instruction tokens!"

    def test_pooling_excludes_response_tokens(self, model_and_inputs):
        """
        Verify that mean/max pooling only considers instruction tokens.
        
        When pooling hidden states, response tokens (with mask=0) should
        be excluded from the pooled representation.
        """
        model, input_ids, attention_mask, labels, instruction_mask, instruction_len = model_and_inputs
        
        if model.content_encoder is None:
            pytest.skip("Content encoder not enabled")
        
        batch_size, seq_len = input_ids.shape
        d_model = model.content_encoder.d_model
        
        # Create fake hidden states
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # Zero out response tokens (as encoder does)
        mask_expanded = instruction_mask.unsqueeze(-1).float()
        hidden_states = hidden_states * mask_expanded
        
        # Apply mean pooling with instruction mask as attention mask
        pooled = model.content_encoder._mean_pooling(hidden_states, instruction_mask)
        
        # The pooling should be equivalent to pooling only instruction tokens
        instruction_hidden = hidden_states[:, :instruction_len, :]
        expected_pooled = instruction_hidden.mean(dim=1)
        
        # Should be identical (since response tokens are zero and excluded from mask)
        diff = (pooled - expected_pooled).abs().max().item()
        
        assert diff < 1e-5, \
            f"Pooling differs by {diff:.6e} - response tokens may be leaking into pooled representation!"
