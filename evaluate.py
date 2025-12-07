#!/usr/bin/env python
"""
SCI Evaluation Script - Section 4.1-4.4

Implements:
- SCAN benchmark evaluation with exact match metric
- Proper EOS handling during generation
- Consistent padding/masking between training and evaluation
- Detailed error analysis and visualization
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from sci.config.config_loader import load_config
from sci.models.sci_model import SCIModel
from sci.data.datasets.scan_dataset import SCANDataset
from sci.data.scan_data_collator import SCANDataCollator
from sci.evaluation.scan_evaluator import SCANEvaluator
from sci.training.checkpoint_manager import TrainingResumer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate SCI model on SCAN')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/sci_full.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint to evaluate')
    parser.add_argument('--split', type=str, default='length',
                       choices=['simple', 'length', 'template'],
                       help='SCAN split to evaluate')
    parser.add_argument('--subset', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset subset to evaluate')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: use config value)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for debugging)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save all predictions to file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def analyze_errors(errors, output_dir):
    """
    Analyze and save error patterns.

    Args:
        errors: List of error dicts from evaluator
        output_dir: Directory to save analysis
    """
    if not errors:
        print("No errors to analyze!")
        return

    # Group errors by type
    error_types = {
        'length_mismatch': [],
        'token_errors': [],
        'structure_errors': [],
    }

    for error in errors:
        pred_tokens = error['prediction'].split()
        target_tokens = error['target'].split()

        if len(pred_tokens) != len(target_tokens):
            error_types['length_mismatch'].append(error)
        elif any(p != t for p, t in zip(pred_tokens, target_tokens)):
            error_types['token_errors'].append(error)
        else:
            error_types['structure_errors'].append(error)

    # Save error analysis
    analysis_path = output_dir / 'error_analysis.json'
    analysis = {
        'total_errors': len(errors),
        'length_mismatch': len(error_types['length_mismatch']),
        'token_errors': len(error_types['token_errors']),
        'structure_errors': len(error_types['structure_errors']),
        'sample_errors': {
            'length_mismatch': error_types['length_mismatch'][:5],
            'token_errors': error_types['token_errors'][:5],
            'structure_errors': error_types['structure_errors'][:5],
        }
    }

    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nError Analysis:")
    print(f"  Total errors: {len(errors)}")
    print(f"  Length mismatches: {len(error_types['length_mismatch'])}")
    print(f"  Token errors: {len(error_types['token_errors'])}")
    print(f"  Structure errors: {len(error_types['structure_errors'])}")
    print(f"  Analysis saved to: {analysis_path}")


def evaluate_by_length(model, dataset, tokenizer, evaluator, device, batch_size):
    """
    Evaluate model performance by sequence length.

    This is particularly important for length generalization split.
    """
    # Group samples by length
    length_groups = {}
    for sample in dataset.data:
        action_length = len(sample['actions'].split())
        if action_length not in length_groups:
            length_groups[action_length] = []
        length_groups[action_length].append(sample)

    print(f"\nEvaluating by length:")
    print(f"  Found {len(length_groups)} length groups")

    results_by_length = {}

    for length in sorted(length_groups.keys()):
        samples = length_groups[length]

        # Create temporary dataset
        temp_dataset = SCANDataset(split='length', subset='test')
        temp_dataset.data = samples

        # Create dataloader
        collator = SCANDataCollator(tokenizer, max_length=512)
        loader = DataLoader(
            temp_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

        # Evaluate
        results = evaluator.evaluate(model, loader, device=device)

        results_by_length[length] = {
            'num_samples': len(samples),
            'exact_match': results['exact_match'],
            'token_accuracy': results['token_accuracy'],
        }

        print(f"  Length {length:2d}: "
              f"{len(samples):4d} samples, "
              f"Exact Match: {results['exact_match']:.4f}, "
              f"Token Acc: {results['token_accuracy']:.4f}")

    return results_by_length


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override split if specified
    if args.split:
        config['data']['scan_split'] = args.split

    # Setup output directory
    output_dir = Path(args.output_dir) / f"{config['experiment']['name']}_{args.split}_{args.subset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        cache_dir='.cache/models'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    print(f"Loading SCAN dataset (split={args.split}, subset={args.subset})...")
    dataset = SCANDataset(
        split=args.split,
        subset=args.subset,
        cache_dir='.cache/datasets'
    )

    # Limit samples if specified (for debugging)
    if args.max_samples:
        dataset.data = dataset.data[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    print(f"Dataset size: {len(dataset)} samples")

    # Create data collator and dataloader
    collator = SCANDataCollator(tokenizer, max_length=512)
    batch_size = args.batch_size or config['training']['batch_size']

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Create model
    print("Creating SCI model...")
    model = SCIModel(config)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    resumer = TrainingResumer(args.checkpoint)
    resumer.load_checkpoint(model, optimizer=None, scheduler=None)
    print(f"Loaded checkpoint from epoch {resumer.checkpoint['epoch']}")

    # Create evaluator with eval config
    eval_config = config.get('evaluation', {})
    evaluator = SCANEvaluator(tokenizer, eval_config=eval_config)

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Starting evaluation...")
    print(f"{'='*60}")

    model.eval()
    results = evaluator.evaluate(model, dataloader, device=device)

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Exact Match:     {results['exact_match']:.4f} ({results['exact_match']*100:.2f}%)")
    print(f"Token Accuracy:  {results['token_accuracy']:.4f} ({results['token_accuracy']*100:.2f}%)")
    print(f"Length Accuracy: {results['length_accuracy']:.4f} ({results['length_accuracy']*100:.2f}%)")
    print(f"Total Samples:   {results['total_samples']}")
    print(f"Num Errors:      {results['num_errors']}")

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Analyze errors
    if results['sample_errors']:
        analyze_errors(results['sample_errors'], output_dir)

    # Save all predictions if requested
    if args.save_predictions:
        predictions_path = output_dir / 'predictions.json'
        with open(predictions_path, 'w') as f:
            json.dump(results['sample_errors'], f, indent=2)
        print(f"Predictions saved to: {predictions_path}")

    # Evaluate by length (for length split)
    if args.split == 'length' and not args.max_samples:
        print(f"\n{'='*60}")
        print("Evaluating by sequence length...")
        print(f"{'='*60}")

        results_by_length = evaluate_by_length(
            model, dataset, tokenizer, evaluator, device, batch_size
        )

        # Save length analysis
        length_analysis_path = output_dir / 'results_by_length.json'
        with open(length_analysis_path, 'w') as f:
            json.dump(results_by_length, f, indent=2)
        print(f"\nLength analysis saved to: {length_analysis_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Subset: {args.subset}")
    print(f"Output directory: {output_dir}")
    print(f"\nKey Metric (Exact Match): {results['exact_match']:.4f}")


if __name__ == '__main__':
    main()
