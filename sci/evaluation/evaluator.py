"""
SCI Evaluator: Evaluation on SCAN benchmark with exact match metrics.

Key Features:
1. Exact match accuracy (standard for SCAN)
2. Token-level accuracy
3. Structural invariance metrics
4. Support for multiple evaluation datasets
5. Generation with greedy decoding or beam search
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

from sci.models.sci_model import SCIModel
from sci.data.datasets.scan_dataset import SCANDataset
from sci.data.scan_data_collator import SCANDataCollator


class SCIEvaluator:
    """
    Evaluator for SCI models on compositional generalization benchmarks.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(
        self,
        model: SCIModel,
        dataset_configs: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on one or more datasets.

        Args:
            model: SCIModel to evaluate
            dataset_configs: List of dataset configs. If None, uses config.evaluation.datasets

        Returns:
            Dictionary mapping dataset_name → metrics
        """
        model.eval()

        if dataset_configs is None:
            dataset_configs = self.config.evaluation.datasets

        results = {}

        for dataset_config in dataset_configs:
            dataset_name = f"{dataset_config['name']}_{dataset_config['split']}"
            print(f"\nEvaluating on {dataset_name}...")

            # Create dataset
            dataset = SCANDataset(
                tokenizer=model.tokenizer,
                split_name=dataset_config['split'],
                subset=dataset_config.get('subset', 'test'),
                max_length=self.config.evaluation.get('max_generation_length', 128),
            )

            # Evaluate
            metrics = self._evaluate_dataset(model, dataset, dataset_name)
            results[dataset_name] = metrics

            # Print results
            print(f"  Exact Match: {metrics['exact_match']:.2%}")
            print(f"  Token Accuracy: {metrics['token_accuracy']:.2%}")

        return results

    def _evaluate_dataset(
        self,
        model: SCIModel,
        dataset: SCANDataset,
        dataset_name: str,
    ) -> Dict[str, float]:
        """
        Evaluate on a single dataset.

        Args:
            model: Model to evaluate
            dataset: SCAN dataset
            dataset_name: Name for logging

        Returns:
            Dictionary with metrics
        """
        # Get tokenizer from model for decoding
        tokenizer = model.tokenizer
        
        # Create data loader (no pair labels needed for evaluation)
        loader = DataLoader(
            dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            collate_fn=lambda batch: self._collate_eval_batch(batch, tokenizer),
            num_workers=0,
        )

        all_predictions = []
        all_references = []
        all_correct_tokens = []
        all_total_tokens = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                references = batch['references']

                # Generate predictions
                predictions = self._generate_batch(
                    model,
                    input_ids,
                    attention_mask,
                )

                # Decode predictions and references
                pred_texts = [
                    model.tokenizer.decode(pred, skip_special_tokens=True)
                    for pred in predictions
                ]

                all_predictions.extend(pred_texts)
                all_references.extend(references)

                # Compute token-level accuracy
                for pred, ref in zip(pred_texts, references):
                    pred_tokens = pred.split()
                    ref_tokens = ref.split()

                    # Count correct tokens (up to min length)
                    min_len = min(len(pred_tokens), len(ref_tokens))
                    correct = sum(
                        p == r for p, r in zip(pred_tokens[:min_len], ref_tokens[:min_len])
                    )

                    all_correct_tokens.append(correct)
                    all_total_tokens.append(len(ref_tokens))

        # Compute metrics
        exact_matches = sum(
            pred.strip() == ref.strip()
            for pred, ref in zip(all_predictions, all_references)
        )
        exact_match_acc = exact_matches / len(all_predictions)

        token_acc = sum(all_correct_tokens) / sum(all_total_tokens) if sum(all_total_tokens) > 0 else 0.0

        metrics = {
            'exact_match': exact_match_acc,
            'token_accuracy': token_acc,
            'num_examples': len(all_predictions),
            'num_exact_matches': exact_matches,
        }

        # Optionally compute structural invariance
        if self.config.evaluation.get('compute_structural_invariance', False):
            struct_inv = self._compute_structural_invariance(model, dataset)
            metrics['structural_invariance'] = struct_inv

        return metrics

    def _collate_eval_batch(self, batch: List[Dict], tokenizer=None) -> Dict:
        """
        Collate batch for evaluation (no pair labels).

        Args:
            batch: List of examples
            tokenizer: Tokenizer for decoding (passed separately)

        Returns:
            Batched tensors
        """
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])

        # Extract reference outputs (ground truth)
        # Store the raw actions text if available
        references = []
        for ex in batch:
            if 'actions' in ex:
                # Use the raw actions string (preferred)
                references.append(ex['actions'])
            elif tokenizer is not None:
                # Decode the labels to get reference
                labels = ex['labels'].clone()
                labels[labels == -100] = tokenizer.pad_token_id  # Replace -100 with pad token for decoding
                ref_text = tokenizer.decode(labels, skip_special_tokens=True)
                references.append(ref_text)
            else:
                references.append("")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'references': references,
        }

    def _generate_batch(
        self,
        model: SCIModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Generate predictions for a batch.

        Args:
            model: Model to use
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            List of generated token ID sequences
        """
        # Extract instruction part (everything before "Output:")
        # For SCAN format: "Instruction: {cmd}\nOutput: "
        batch_size = input_ids.shape[0]
        instruction_inputs = []

        for i in range(batch_size):
            # Find "Output:" token position
            tokens = input_ids[i]
            mask = attention_mask[i]

            # Decode to find "Output:"
            text = model.tokenizer.decode(tokens[mask == 1], skip_special_tokens=False)

            # Find instruction part
            if "Output:" in text:
                instruction_text = text.split("Output:")[0] + "Output:"
            else:
                instruction_text = text

            # Re-tokenize instruction only
            instruction_ids = model.tokenizer(
                instruction_text,
                return_tensors='pt',
                add_special_tokens=True,
            )['input_ids'].to(self.device)

            instruction_inputs.append(instruction_ids)

        # Pad to same length
        max_len = max(inp.shape[1] for inp in instruction_inputs)
        padded_inputs = []
        padded_masks = []

        for inp in instruction_inputs:
            padding_len = max_len - inp.shape[1]
            if padding_len > 0:
                padded = torch.cat([
                    inp,
                    torch.full((1, padding_len), model.tokenizer.pad_token_id, device=self.device)
                ], dim=1)
                mask = torch.cat([
                    torch.ones(1, inp.shape[1], device=self.device),
                    torch.zeros(1, padding_len, device=self.device)
                ], dim=1)
            else:
                padded = inp
                mask = torch.ones_like(inp)

            padded_inputs.append(padded)
            padded_masks.append(mask)

        instruction_ids = torch.cat(padded_inputs, dim=0)
        instruction_mask = torch.cat(padded_masks, dim=0)

        # Generate
        max_length = self.config.evaluation.get('max_generation_length', 128)
        num_beams = self.config.evaluation.get('num_beams', 1)

        generated_ids = model.generate(
            input_ids=instruction_ids,
            attention_mask=instruction_mask,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=self.config.evaluation.get('do_sample', False),
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )

        return generated_ids

    def _compute_structural_invariance(
        self,
        model: SCIModel,
        dataset: SCANDataset,
    ) -> float:
        """
        Compute structural invariance metric.

        Measures how similar structural representations are for examples
        with the same structure but different content.

        Args:
            model: Model with structural encoder
            dataset: Dataset to evaluate on

        Returns:
            Structural invariance score (average cosine similarity)
        """
        if model.structural_encoder is None:
            return 0.0

        # Sample pairs with same structure
        pair_generator = dataset.pair_generator

        # Get all examples
        all_commands = dataset.commands[:100]  # Limit for efficiency

        # Extract structural representations
        representations = []

        with torch.no_grad():
            for command in tqdm(all_commands, desc="Computing structural invariance"):
                # Tokenize
                encoded = model.tokenizer(
                    f"Instruction: {command}\nOutput:",
                    return_tensors='pt',
                    truncation=True,
                    max_length=128,
                ).to(self.device)

                # Get structural representation
                struct_repr, _ = model.get_structural_representation(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                )

                # Pool slots
                struct_pooled = struct_repr.mean(dim=1)  # [1, d_model]
                representations.append(struct_pooled.cpu())

        # Compute similarities for same-structure pairs
        representations = torch.cat(representations, dim=0)  # [N, d_model]

        # Normalize
        representations = torch.nn.functional.normalize(representations, dim=-1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)

        # Get pair labels for sampled examples
        sampled_indices = list(range(len(all_commands)))
        pair_labels = pair_generator.get_batch_pair_labels(sampled_indices)

        # Average similarity for positive pairs
        positive_mask = (pair_labels == 1)
        if positive_mask.sum() > 0:
            positive_similarities = similarity_matrix[positive_mask]
            structural_invariance = positive_similarities.mean().item()
        else:
            structural_invariance = 0.0

        return structural_invariance


if __name__ == "__main__":
    # Quick test
    print("Testing SCIEvaluator...")

    from sci.config.config_loader import load_config

    # Load config
    config = load_config("configs/sci_full.yaml")

    # Override for testing
    config.evaluation.batch_size = 4

    # Create evaluator
    evaluator = SCIEvaluator(config)

    print(f"✓ Evaluator initialized")
    print(f"  Device: {evaluator.device}")

    # Note: Full test requires trained model
    print("\nNote: Full evaluation test requires a trained model")
    print("✓ Evaluator ready for use")
