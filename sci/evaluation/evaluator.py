"""
SCI Evaluator: Evaluation on SCAN benchmark with exact match metrics.

Key Features:
1. Exact match accuracy (standard for SCAN)
2. Token-level accuracy
3. Structural invariance metrics
4. Support for multiple evaluation datasets
5. Generation with greedy decoding or beam search

CRITICAL: Uses same prompt format as training (SCANDataCollator) to ensure
proper comparison between training and evaluation.
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
    
    CRITICAL: Uses same data collator as training to ensure prompt format parity.
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

            # Create dataset with proper max_length from config
            max_length = getattr(self.config.data, 'max_length', 512)
            dataset = SCANDataset(
                tokenizer=model.tokenizer,
                split_name=dataset_config['split'],
                subset=dataset_config.get('subset', 'test'),
                max_length=max_length,
                cache_dir=getattr(self.config.data, 'pairs_cache_dir', '.cache/scan'),
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
        tokenizer = model.tokenizer
        
        # CRITICAL FIX: Use SCANDataCollator for proper format parity with training
        # This ensures the instruction format matches what the model was trained on
        use_chat_template = getattr(self.config.data, 'use_chat_template', False)
        max_length = getattr(self.config.data, 'max_length', 512)
        
        # Create collator matching training format (but without pair generator for eval)
        collator = SCANDataCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            pair_generator=None,  # No pair labels needed for evaluation
            use_chat_template=use_chat_template,
        )
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

        all_predictions = []
        all_references = []
        all_correct_tokens = []
        all_total_tokens = []

        # Get generation settings - use beam_size if num_beams not set
        num_beams = getattr(self.config.evaluation, 'num_beams', None)
        if num_beams is None:
            num_beams = getattr(self.config.evaluation, 'beam_size', 1)
        
        max_gen_length = getattr(self.config.evaluation, 'max_generation_length', 512)
        do_sample = getattr(self.config.evaluation, 'do_sample', False)
        repetition_penalty = getattr(self.config.evaluation, 'repetition_penalty', 1.0)
        length_penalty = getattr(self.config.evaluation, 'length_penalty', 1.0)

        # Track batch index for reference lookup
        example_idx = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
                # Extract raw commands for reference lookup
                commands = batch['commands']
                batch_size = len(commands)
                
                # The collator creates full sequences (instruction + response)
                # We need to extract just the instruction part for generation
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                instruction_mask = batch['instruction_mask']
                
                # For each example, extract instruction-only tokens and generate
                for i in range(batch_size):
                    # Get instruction tokens (where instruction_mask == 1)
                    inst_mask = instruction_mask[i]
                    inst_len = inst_mask.sum().item()
                    
                    # Extract instruction tokens
                    inst_ids = input_ids[i, :inst_len].unsqueeze(0)
                    inst_attn = attention_mask[i, :inst_len].unsqueeze(0)
                    
                    # Generate output
                    generated_ids = model.generate(
                        input_ids=inst_ids,
                        attention_mask=inst_attn,
                        max_length=inst_len + max_gen_length,  # instruction + output
                        num_beams=num_beams,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
                    # Extract generated part (after instruction)
                    gen_tokens = generated_ids[0, inst_len:]
                    
                    # Decode prediction (skip special tokens)
                    pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    
                    # Get reference from dataset's outputs list
                    ref_text = dataset.outputs[example_idx] if example_idx < len(dataset.outputs) else ""
                    
                    all_predictions.append(pred_text)
                    all_references.append(ref_text)

                    # Compute token-level accuracy
                    pred_tokens = pred_text.split()
                    ref_tokens = ref_text.split()

                    min_len = min(len(pred_tokens), len(ref_tokens))
                    correct = sum(
                        p == r for p, r in zip(pred_tokens[:min_len], ref_tokens[:min_len])
                    )

                    all_correct_tokens.append(correct)
                    all_total_tokens.append(len(ref_tokens) if ref_tokens else 1)
                    
                    example_idx += 1

        # Compute metrics
        exact_matches = sum(
            pred.strip() == ref.strip()
            for pred, ref in zip(all_predictions, all_references)
        )
        exact_match_acc = exact_matches / len(all_predictions) if all_predictions else 0.0

        token_acc = sum(all_correct_tokens) / sum(all_total_tokens) if sum(all_total_tokens) > 0 else 0.0

        metrics = {
            'exact_match': exact_match_acc,
            'token_accuracy': token_acc,
            'num_examples': len(all_predictions),
            'num_exact_matches': exact_matches,
        }

        # Optionally compute structural invariance
        if getattr(self.config.evaluation, 'compute_structural_invariance', False):
            struct_inv = self._compute_structural_invariance(model, dataset)
            metrics['structural_invariance'] = struct_inv

        return metrics

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
        tokenizer = model.tokenizer

        # Get sample of examples
        num_samples = min(100, len(dataset.commands))
        all_commands = dataset.commands[:num_samples]

        # Use same format as training for consistency
        use_chat_template = getattr(self.config.data, 'use_chat_template', False)

        # Extract structural representations
        representations = []

        with torch.no_grad():
            for command in tqdm(all_commands, desc="Computing structural invariance"):
                # Format command same way as training
                if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
                    messages = [{"role": "user", "content": f"Translate to action sequence: {command}"}]
                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    prompt = command
                
                # Tokenize
                encoded = tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=getattr(self.config.data, 'max_length', 512),
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
