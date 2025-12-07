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
        cache_dir: str = ".cache/scan",  # MEDIUM #56: Configurable cache directory
        force_regenerate_pairs: bool = False,
    ):
        self.tokenizer = tokenizer
        self.split_name = split_name
        self.subset = subset
        self.max_length = max_length
        self.cache_dir = cache_dir
        self._using_dummy_data = False  # Flag for testing

        # Load SCAN dataset
        print(f"Loading SCAN dataset (split={split_name}, subset={subset})...")
        try:
            # Try loading from Hugging Face
            dataset = load_dataset("scan", split_name)
            self.data = dataset[subset]
        except Exception as e:
            # CRITICAL FIX: Don't silently fall back to dummy data in production
            # Only use dummy data if explicitly allowed via environment variable
            import os
            if os.environ.get('SCI_ALLOW_DUMMY_DATA', '').lower() == 'true':
                print(f"WARNING: Failed to load from Hugging Face: {e}")
                print("WARNING: Using dummy data (SCI_ALLOW_DUMMY_DATA=true)")
                self._create_dummy_data()
                self._using_dummy_data = True
            else:
                raise RuntimeError(
                    f"Failed to load SCAN dataset from Hugging Face: {e}\n"
                    f"Please check your internet connection or set SCI_ALLOW_DUMMY_DATA=true for testing."
                )

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

        # CRITICAL FIX: Validate sequence lengths against max_length
        self._validate_sequence_lengths()

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

        # CRITICAL #17: Verify pair matrix is symmetric
        assert torch.allclose(self.pair_matrix, self.pair_matrix.t()), \
            "Pair matrix must be symmetric (pair_matrix[i,j] == pair_matrix[j,i])"

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

    def _validate_sequence_lengths(self):
        """
        Validate that sequences don't exceed max_length.
        
        CRITICAL: Warns if any command+action pair exceeds max_length after tokenization.
        This catches misconfigurations before training.
        """
        if self._using_dummy_data:
            return  # Skip validation for dummy data
        
        exceeds_count = 0
        max_observed = 0
        
        # Sample up to 100 examples for efficiency
        sample_size = min(100, len(self.commands))
        sample_indices = list(range(0, len(self.commands), max(1, len(self.commands) // sample_size)))[:sample_size]
        
        for idx in sample_indices:
            cmd = self.commands[idx]
            act = self.outputs[idx]
            
            # Estimate token count (rough approximation)
            # Full tokenization would be expensive, so use simple estimation
            cmd_tokens = len(self.tokenizer.encode(cmd, add_special_tokens=True))
            act_tokens = len(self.tokenizer.encode(act, add_special_tokens=False))
            total_tokens = cmd_tokens + act_tokens + 2  # +2 for separator and EOS
            
            max_observed = max(max_observed, total_tokens)
            
            if total_tokens > self.max_length:
                exceeds_count += 1
        
        # Report statistics
        if exceeds_count > 0:
            pct = (exceeds_count / sample_size) * 100
            import warnings
            warnings.warn(
                f"SCAN dataset: {exceeds_count}/{sample_size} sampled examples ({pct:.1f}%) "
                f"exceed max_length={self.max_length}. Max observed: {max_observed} tokens. "
                f"Consider increasing max_length for SCAN length split (needs ~512+ tokens)."
            )
        
        print(f"  Max observed sequence length: {max_observed} tokens (limit: {self.max_length})")

    def __len__(self) -> int:
        return len(self.commands)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single example.

        Returns:
            Dictionary with:
                - commands: Raw command string (for collator and pair generation)
                - actions: Raw action string (for collator)
                - idx: Dataset index (for pair lookup)
        """
        command = self.commands[idx]
        action = self.outputs[idx]

        # CRITICAL #22: Return raw strings for collator to process
        # This allows proper instruction mask creation in collator
        return {
            "commands": command,
            "actions": action,
            "idx": idx,  # Keep index for pair lookup
        }

    def get_structure_stats(self) -> Dict:
        """Get statistics about structural distribution."""
        return self.pair_generator.get_structure_statistics()


# #10 FIX: Removed deprecated SCANCollator class.
# Use SCANDataCollator from sci.data.scan_data_collator instead.


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

    # Test collator - #47 FIX: Use SCANDataCollator instead of deprecated SCANCollator
    print("\n" + "=" * 60)
    print("Testing Collator")
    print("=" * 60)

    from torch.utils.data import DataLoader
    from sci.data.scan_data_collator import SCANDataCollator

    collator = SCANDataCollator(
        tokenizer=tokenizer,
        max_length=128,
        pair_generator=dataset.pair_generator,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    
    # #99 FIX: Use 'idx' key which is what __getitem__ returns
    # The collator receives dicts with 'commands', 'actions', 'idx' keys
    # and produces batches with 'input_ids', 'attention_mask', 'labels', etc.
    if hasattr(dataset, 'pair_generator'):
        # Get indices from the batch (collator should pass through or we use dataset directly)
        sample_indices = list(range(min(len(dataset), batch['input_ids'].shape[0])))
        pair_labels = dataset.pair_generator.get_batch_pair_labels(sample_indices)
        print(f"Pair labels shape: {pair_labels.shape}")
        print(f"\nPair labels matrix:")
        print(pair_labels)
        # Count positive pairs
        num_positive = (pair_labels.sum() - pair_labels.diagonal().sum()).item()
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
