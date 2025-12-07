"""
SCAN Evaluator with Exact Match Scoring

Required by: SCI_ENGINEERING_STANDARDS.md Section 4.2

BUG FIX: Updated to handle concatenated sequence format where:
- input_ids = instruction + response (concatenated)
- labels = [-100 for instruction tokens] + [response tokens]
- instruction_mask = [1 for instruction tokens] + [0 for response tokens]

For evaluation, we need to:
1. Extract just the instruction tokens
2. Generate the response
3. Compare to expected response
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

        With the concatenated format, we need to extract instruction tokens
        and generate, then compare to expected response.

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
                
                # Get instruction_mask if available, else derive from labels
                instruction_mask = batch.get('instruction_mask')
                if instruction_mask is not None:
                    instruction_mask = instruction_mask.to(device)
                else:
                    # Derive from labels: instruction tokens have labels == -100
                    instruction_mask = (labels == -100).long()

                batch_size = input_ids.shape[0]

                # Process each example in batch
                for i in range(batch_size):
                    # Extract instruction tokens only (where instruction_mask == 1)
                    inst_mask_i = instruction_mask[i]
                    inst_length = inst_mask_i.sum().item()
                    
                    # Get instruction input_ids
                    instruction_input_ids = input_ids[i, :inst_length].unsqueeze(0)
                    instruction_attention_mask = attention_mask[i, :inst_length].unsqueeze(0)

                    # Extract target: tokens where labels != -100
                    labels_i = labels[i]
                    valid_label_mask = labels_i != -100
                    target_tokens = labels_i[valid_label_mask]

                    # CRITICAL #18: Use max_length to avoid context overflow
                    max_output_tokens = 300  # SCAN length split max = 288 tokens
                    max_total_length = min(inst_length + max_output_tokens, 2048)

                    # Generate prediction from instruction
                    outputs = model.generate(
                        input_ids=instruction_input_ids,
                        attention_mask=instruction_attention_mask,
                        max_length=max_total_length,
                        do_sample=False,  # Greedy decoding
                        num_beams=1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    # Extract generated tokens (remove instruction prefix)
                    generated_tokens = outputs[0, inst_length:]

                    # Decode
                    prediction = self.tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    ).strip()

                    target = self.tokenizer.decode(
                        target_tokens,
                        skip_special_tokens=True
                    ).strip()

                    input_text = self.tokenizer.decode(
                        instruction_input_ids[0],
                        skip_special_tokens=True
                    ).strip()

                    results['total'] += 1

                    # Normalize whitespace
                    pred_normalized = ' '.join(prediction.split())
                    target_normalized = ' '.join(target.split())

                    # EXACT MATCH (most important metric for SCAN)
                    if pred_normalized == target_normalized:
                        results['exact_match'] += 1
                    else:
                        # Log error for analysis
                        if len(results['errors']) < 100:
                            results['errors'].append({
                                'input': input_text,
                                'prediction': pred_normalized,
                                'target': target_normalized
                            })

                    # Token accuracy
                    pred_tokens = pred_normalized.split()
                    target_tokens_split = target_normalized.split()

                    if len(pred_tokens) == len(target_tokens_split):
                        results['length_correct'] += 1
                        if len(target_tokens_split) > 0:
                            correct_tokens = sum(p == t for p, t in zip(pred_tokens, target_tokens_split))
                            results['token_accuracy'] += correct_tokens / len(target_tokens_split)
                    else:
                        # Length mismatch - partial credit for matching prefix
                        min_len = min(len(pred_tokens), len(target_tokens_split))
                        if min_len > 0 and len(target_tokens_split) > 0:
                            correct_tokens = sum(p == t for p, t in
                                               zip(pred_tokens[:min_len], target_tokens_split[:min_len]))
                            results['token_accuracy'] += correct_tokens / len(target_tokens_split)

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
