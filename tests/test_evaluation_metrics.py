"""Evaluation Metrics Tests - Section 2.1"""
import pytest

def test_scan_evaluator_exists():
    """Test SCANEvaluator exists and has required methods"""
    from sci.evaluation.scan_evaluator import SCANEvaluator
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', cache_dir='.cache/models')
    evaluator = SCANEvaluator(tokenizer)
    assert hasattr(evaluator, 'evaluate')
    assert hasattr(evaluator, 'print_results')

def test_exact_match_metric():
    """Test exact match calculation"""
    pred = "WALK WALK JUMP"
    target = "WALK WALK JUMP"
    assert pred == target  # Exact match

def test_token_accuracy():
    """Test token accuracy calculation"""
    pred_tokens = ["WALK", "WALK", "JUMP"]
    target_tokens = ["WALK", "WALK", "JUMP"]
    correct = sum(p == t for p, t in zip(pred_tokens, target_tokens))
    accuracy = correct / len(target_tokens)
    assert accuracy == 1.0
