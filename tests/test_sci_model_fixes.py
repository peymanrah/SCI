
import torch
import pytest
from unittest.mock import MagicMock, patch
from sci.models.sci_model import SCIModel
from sci.models.components.structural_encoder import StructuralEncoder
from sci.models.components.causal_binding import CausalBindingMechanism

class MockConfig:
    def __init__(self):
        self.model = MagicMock()
        self.model.base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model.structural_encoder.enabled = True
        self.model.structural_encoder.d_model = 128
        self.model.structural_encoder.num_layers = 1
        self.model.structural_encoder.num_slots = 4
        self.model.structural_encoder.num_heads = 4
        self.model.structural_encoder.dim_feedforward = 256
        self.model.structural_encoder.dropout = 0.1
        self.model.structural_encoder.abstraction_layer.injection_layers = [0]
        self.model.structural_encoder.abstraction_layer.hidden_multiplier = 2
        self.model.structural_encoder.abstraction_layer.residual_init = 0.1
        self.model.structural_encoder.abstraction_layer.dropout = 0.1
        self.model.structural_encoder.slot_attention.num_iterations = 3
        self.model.structural_encoder.slot_attention.epsilon = 1e-8
        self.model.structural_encoder.use_edge_prediction = True
        
        self.model.content_encoder.enabled = True
        self.model.content_encoder.d_model = 128
        self.model.content_encoder.num_layers = 1
        self.model.content_encoder.num_heads = 4
        self.model.content_encoder.dim_feedforward = 256
        self.model.content_encoder.dropout = 0.1
        self.model.content_encoder.pooling = "mean"
        
        self.model.causal_binding.enabled = True
        self.model.causal_binding.injection_layers = [0]
        self.model.causal_binding.d_model = 128
        self.model.causal_binding.num_heads = 4
        self.model.causal_binding.dropout = 0.1
        self.model.causal_binding.use_causal_intervention = True
        self.model.causal_binding.use_structural_eos = True
        
        self.model.position_encoding.max_length = 512
        self.model.position_encoding.base = 10000
        
        self.training = MagicMock()
        self.training.mixed_precision = False

@pytest.fixture
def mock_sci_model():
    config = MockConfig()
    with patch('sci.models.sci_model.AutoModelForCausalLM') as mock_base_cls, \
         patch('sci.models.sci_model.AutoTokenizer') as mock_tokenizer_cls:
        
        mock_base = MagicMock()
        mock_base.config.hidden_size = 128
        mock_base.config.vocab_size = 1000
        mock_base.config.num_hidden_layers = 2
        mock_base.get_input_embeddings.return_value = torch.nn.Embedding(1000, 128)
        mock_base.model.layers = [MagicMock() for _ in range(2)]
        mock_base_cls.from_pretrained.return_value = mock_base
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        model = SCIModel(config)
        return model

def test_generate_unpacking_fix(mock_sci_model):
    """Test that generate correctly unpacks 3 values from structural_encoder."""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Mock structural encoder return values
    structural_slots = torch.randn(batch_size, 4, 128)
    structural_scores = [torch.randn(batch_size, seq_len, 128)]
    edge_weights = torch.randn(batch_size, 4, 4)
    
    mock_sci_model.structural_encoder = MagicMock(return_value=(structural_slots, structural_scores, edge_weights))
    mock_sci_model.content_encoder = MagicMock(return_value=torch.randn(batch_size, 128))
    mock_sci_model.base_model.generate.return_value = torch.randint(0, 1000, (batch_size, 20))
    
    # Run generate
    generated = mock_sci_model.generate(input_ids, attention_mask)
    
    # Verify structural_encoder was called
    mock_sci_model.structural_encoder.assert_called_once()
    
    # Verify edge_weights were stored (though cleared in finally block, we check logic flow)
    # Since we can't check internal state during execution easily without more mocking,
    # the fact that it didn't raise ValueError is the main test.
    assert generated is not None

def test_forward_eos_loss_call(mock_sci_model):
    """Test that forward calls get_eos_loss with correct arguments."""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Mock components
    mock_sci_model.structural_encoder = MagicMock(return_value=(
        torch.randn(batch_size, 4, 128), 
        [], 
        torch.randn(batch_size, 4, 4)
    ))
    mock_sci_model.content_encoder = MagicMock(return_value=torch.randn(batch_size, 128))
    
    # Mock CBM
    mock_sci_model.causal_binding.bind.return_value = (torch.randn(batch_size, 4, 128), None)
    mock_sci_model.causal_binding.get_eos_loss = MagicMock(return_value=torch.tensor(0.5))
    
    # Mock base model output
    mock_sci_model.base_model.return_value = MagicMock(
        logits=torch.randn(batch_size, seq_len, 1000),
        loss=torch.tensor(1.0)
    )
    
    # Run forward
    result = mock_sci_model(input_ids, attention_mask, labels=labels)
    
    # Verify get_eos_loss was called with correct args
    mock_sci_model.causal_binding.get_eos_loss.assert_called_once()
    call_kwargs = mock_sci_model.causal_binding.get_eos_loss.call_args.kwargs
    
    assert 'bound_slots' in call_kwargs
    assert 'eos_positions' in call_kwargs
    assert 'sequence_lengths' in call_kwargs
    assert 'labels' not in call_kwargs  # Should NOT pass labels anymore
    assert 'eos_token_id' not in call_kwargs  # Should NOT pass eos_token_id anymore
    
    assert result['structural_eos_loss'] is not None
