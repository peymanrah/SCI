"""
SCAN Dataset with Structural Pairs.

Loads SCAN benchmark and integrates pre-generated structural pairs for SCL training.

SCAN (Simplified version of the CommAI Navigation task):
- Input: Natural language commands (e.g., "jump twice after walk left")
- Output: Action sequences (e.g., "JUMP JUMP WALK LTURN WALK")

Splits:
- simple: Standard train/test split
- length: Train on short sequences, test on longer (OOD generalization)
- template: Train on specific templates, test on novel templates
- addprim_jump/turn_left/around_right: Primitive addition splits

This dataset:
1. Downloads SCAN from Hugging Face datasets
2. Pre-generates structural pairs using SCANPairGenerator
3. Provides pair labels during __getitem__
4. Supports efficient batching with pair-aware collation
"""

import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional
from dataclasses import dataclass

from sci.data.structure_extractors.scan_extractor import SCANStructureExtractor
from sci.data.pair_generators.scan_pair_generator import SCANPairGenerator


class SCANDataset(Dataset):
    """
    SCAN dataset with pre-computed structural pairs.

    Args:
        tokenizer: HuggingFace tokenizer
        split_name: SCAN split ('simple', 'length', 'template', etc.)
        subset: 'train' or 'test'
        max_length: Maximum sequence length
        cache_dir: Directory for caching pairs
        force_regenerate_pairs: Force regeneration of pair cache
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split_name: str = "length",
        subset: str = "train",
        max_length: int = 128,
        cache_dir: str = ".cache/scan",
        force_regenerate_pairs: bool = False,
    ):
        self.tokenizer = tokenizer
        self.split_name = split_name
        self.subset = subset
        self.max_length = max_length
        self.cache_dir = cache_dir

        # Load SCAN dataset
        print(f"Loading SCAN dataset (split={split_name}, subset={subset})...")
        try:
            # Try loading from Hugging Face
            dataset = load_dataset("scan", split_name)
            self.data = dataset[subset]
        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            print("Using dummy data for testing...")
            self._create_dummy_data()

        # Extract commands and outputs
        self.commands = []
        self.outputs = []

        for example in self.data:
            # SCAN format: {"commands": str, "actions": str}
            if isinstance(example, dict):
                self.commands.append(example.get("commands", ""))
                self.outputs.append(example.get("actions", ""))
            else:
                # Handle different formats
                self.commands.append(str(example[0]) if len(example) > 0 else "")
                self.outputs.append(str(example[1]) if len(example) > 1 else "")

        print(f"✓ Loaded {len(self.commands)} examples")

        # Initialize structure extractor
        self.structure_extractor = SCANStructureExtractor()

        # Initialize pair generator
        self.pair_generator = SCANPairGenerator(
            extractor=self.structure_extractor,
            cache_dir=cache_dir,
        )

        # Generate/load pairs
        cache_name = f"scan_{split_name}_{subset}_pairs"
        print(f"Generating/loading structural pairs...")
        self.pair_matrix = self.pair_generator.generate_pairs(
            commands=self.commands,
            cache_name=cache_name,
            force_regenerate=force_regenerate_pairs,
        )

        print(f"✓ SCAN dataset ready: {len(self)} examples")

    def _create_dummy_data(self):
        """Create dummy SCAN data for testing."""
        dummy_examples = [
            {"commands": "walk twice", "actions": "WALK WALK"},
            {"commands": "run twice", "actions": "RUN RUN"},
            {"commands": "jump left", "actions": "LTURN JUMP"},
            {"commands": "look right", "actions": "RTURN LOOK"},
            {"commands": "walk and run", "actions": "WALK RUN"},
        ]
        self.data = dummy_examples * 20  # Replicate for testing

    def __len__(self) -> int:
        return len(self.commands)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Returns:
            Dictionary with:
                - input_ids: Tokenized input
                - attention_mask: Attention mask
                - labels: Tokenized output (with instruction masked as -100)
                - idx: Dataset index (for pair lookup)
        """
        command = self.commands[idx]
        output = self.outputs[idx]

        # Format as instruction-response pair
        # Format: "Instruction: {command}\nOutput: {output}"
        text = f"Instruction: {command}\nOutput: {output}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels with instruction masked
        labels = input_ids.clone()

        # Find where output starts (after "Output: ")
        # Tokenize instruction part only to find the split point
        instruction_text = f"Instruction: {command}\nOutput: "
        instruction_encoding = self.tokenizer(
            instruction_text,
            truncation=False,
            add_special_tokens=False,
        )
        instruction_len = len(instruction_encoding["input_ids"])

        # Mask instruction tokens with -100 (not included in loss)
        labels[:instruction_len] = -100

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "idx": idx,  # Keep index for pair lookup
        }

    def get_structure_stats(self) -> Dict:
        """Get statistics about structural distribution."""
        return self.pair_generator.get_structure_statistics()


@dataclass
class SCANCollator:
    """
    Custom collator for SCAN dataset that includes pair labels.

    This collator:
    1. Batches examples normally
    2. Looks up pair labels for the batch indices
    3. Returns batch with pair_labels tensor
    """

    dataset: SCANDataset

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and add pair labels.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Dictionary with batched tensors including pair_labels
        """
        # Extract indices
        indices = [example["idx"] for example in batch]

        # Stack tensors
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        labels = torch.stack([example["labels"] for example in batch])

        # Get pair labels for this batch
        pair_labels = self.dataset.pair_generator.get_batch_pair_labels(indices)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pair_labels": pair_labels,
            "indices": torch.tensor(indices),
        }


if __name__ == "__main__":
    # Test SCAN dataset
    print("Testing SCANDataset...\n")

    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print("=" * 60)
    print("Creating Dataset")
    print("=" * 60)

    dataset = SCANDataset(
        tokenizer=tokenizer,
        split_name="length",
        subset="train",
        max_length=128,
        cache_dir=".test_cache/scan",
        force_regenerate_pairs=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test __getitem__
    print("\n" + "=" * 60)
    print("Testing __getitem__")
    print("=" * 60)

    example = dataset[0]
    print(f"Keys: {example.keys()}")
    print(f"Input IDs shape: {example['input_ids'].shape}")
    print(f"Attention mask shape: {example['attention_mask'].shape}")
    print(f"Labels shape: {example['labels'].shape}")
    print(f"Index: {example['idx']}")

    # Decode to verify
    print(f"\nDecoded input:")
    print(tokenizer.decode(example['input_ids'], skip_special_tokens=False))

    print(f"\nInstruction mask (labels == -100):")
    instruction_mask = (example['labels'] == -100)
    print(f"  Instruction length: {instruction_mask.sum().item()} tokens")

    # Test collator
    print("\n" + "=" * 60)
    print("Testing Collator")
    print("=" * 60)

    from torch.utils.data import DataLoader

    collator = SCANCollator(dataset)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Pair labels shape: {batch['pair_labels'].shape}")
    print(f"\nPair labels matrix:")
    print(batch['pair_labels'])

    # Count positive pairs
    num_positive = (batch['pair_labels'].sum() - batch['pair_labels'].diagonal().sum()).item()
    print(f"\nPositive pairs in batch: {num_positive}")

    # Test structure stats
    print("\n" + "=" * 60)
    print("Structure Statistics")
    print("=" * 60)

    stats = dataset.get_structure_stats()
    print(f"Unique structures: {stats['num_unique_structures']}")
    print(f"Total examples: {stats['num_examples']}")
    print(f"Avg examples per structure: {stats['avg_examples_per_structure']:.1f}")
    print(f"Positive pair ratio: {stats['positive_ratio']:.2%}")

    print(f"\nTop 5 most common structures:")
    for template, count in stats['most_common'][:5]:
        print(f"  {template}: {count}")

    # Cleanup
    import shutil
    if os.path.exists(".test_cache"):
        shutil.rmtree(".test_cache")
        print("\n✓ Test cache cleaned up")

    print("\n✓ All SCAN dataset tests passed!")
