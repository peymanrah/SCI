"""Data Preparation Tests - Section 2.1"""
import pytest

def test_scan_loader():
    """Test SCAN data loading"""
    from sci.data.scan_loader import load_scan
    data = load_scan('length', 'train')
    assert len(data) > 0
    assert 'commands' in data[0]
    assert 'actions' in data[0]

def test_data_collator():
    """Test SCANDataCollator"""
    from sci.data.scan_data_collator import SCANDataCollator
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', cache_dir='.cache/models')
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    collator = SCANDataCollator(tokenizer)
    batch = collator([{'commands': 'jump twice', 'actions': 'JUMP JUMP'}])
    assert 'input_ids' in batch
    assert 'labels' in batch
    assert 'instruction_mask' in batch
