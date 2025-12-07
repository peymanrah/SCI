"""
SCAN Evaluator with Exact Match Scoring

Required by: SCI_ENGINEERING_STANDARDS.md Section 4.2
"""

import torch
from tqdm import tqdm


class SCANEvaluator:
    """Evaluator for SCAN benchmark with exact match scoring."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate(self, model, test_dataloader, device='cuda'):
        """
        Evaluate model on SCAN dataset.

        Args:
            model: SCI model to evaluate
            test_dataloader: DataLoader with test data
            device: Device to run evaluation on

        Returns:
            dict: Evaluation results with metrics
        """
        model.eval()

        results = {
            'exact_match': 0,
            'token_accuracy': 0,
            'length_correct': 0,
            'total': 0,
            'errors': []
        }

        # HIGH #41/#70: Use torch.inference_mode for better performance
        with torch.inference_mode():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # CRITICAL #18: Use max_length instead of max_new_tokens to avoid context overflow
                # Calculate safe max_length: input_length + max_output_tokens
                max_output_tokens = 300  # SCAN length split max = 288 tokens
                max_total_length = min(input_ids.shape[1] + max_output_tokens, 2048)  # TinyLlama max = 2048

                # Generate predictions
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_total_length,  # Use max_length to prevent context overflow
                    do_sample=False,  # Greedy decoding
                    num_beams=1,  # No beam search
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # Decode predictions and targets
                # Remove input prefix from generated output
                generated_tokens = outputs[:, input_ids.shape[1]:]

                predictions = self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )

                targets = self.tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True
                )

                for pred, target, input_seq in zip(predictions, targets, input_ids):
                    results['total'] += 1

                    # Normalize whitespace
                    pred_normalized = ' '.join(pred.strip().split())
                    target_normalized = ' '.join(target.strip().split())

                    # EXACT MATCH (most important metric for SCAN)
                    if pred_normalized == target_normalized:
                        results['exact_match'] += 1
                    else:
                        # Log error for analysis
                        input_text = self.tokenizer.decode(input_seq, skip_special_tokens=True)
                        if len(results['errors']) < 100:  # Keep first 100 errors
                            results['errors'].append({
                                'input': input_text,
                                'prediction': pred_normalized,
                                'target': target_normalized
                            })

                    # Token accuracy
                    pred_tokens = pred_normalized.split()
                    target_tokens = target_normalized.split()

                    if len(pred_tokens) == len(target_tokens):
                        results['length_correct'] += 1
                        if len(target_tokens) > 0:
                            correct_tokens = sum(p == t for p, t in zip(pred_tokens, target_tokens))
                            results['token_accuracy'] += correct_tokens / len(target_tokens)
                    else:
                        # Length mismatch - partial credit for matching prefix
                        min_len = min(len(pred_tokens), len(target_tokens))
                        if min_len > 0 and len(target_tokens) > 0:
                            correct_tokens = sum(p == t for p, t in
                                               zip(pred_tokens[:min_len], target_tokens[:min_len]))
                            results['token_accuracy'] += correct_tokens / len(target_tokens)

        # Compute final metrics
        n = results['total']
        if n == 0:
            return {
                'exact_match': 0.0,
                'token_accuracy': 0.0,
                'length_accuracy': 0.0,
                'total_samples': 0,
                'num_errors': 0,
                'sample_errors': []
            }

        return {
            'exact_match': results['exact_match'] / n,
            'token_accuracy': results['token_accuracy'] / n,
            'length_accuracy': results['length_correct'] / n,
            'total_samples': n,
            'num_errors': len(results['errors']),
            'sample_errors': results['errors'][:10]  # First 10 errors for analysis
        }

    def print_results(self, results, split_name="Test"):
        """Print evaluation results in a formatted way."""
        print(f"\n{'='*70}")
        print(f"{split_name} Evaluation Results")
        print(f"{'='*70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Exact Match: {results['exact_match']*100:.2f}%")
        print(f"Token Accuracy: {results['token_accuracy']*100:.2f}%")
        print(f"Length Accuracy: {results['length_accuracy']*100:.2f}%")
        print(f"Number of errors: {results['num_errors']}")

        if results['sample_errors']:
            print(f"\nSample Errors (first {len(results['sample_errors'])}):")
            for i, error in enumerate(results['sample_errors'][:3], 1):
                print(f"\n  Error {i}:")
                print(f"    Input:      {error['input']}")
                print(f"    Predicted:  {error['prediction']}")
                print(f"    Target:     {error['target']}")

        print(f"{'='*70}\n")
