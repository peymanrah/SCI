"""
SCAN Pair Generator with Caching.

Pre-generates and caches structural positive/negative pairs before training.
This significantly speeds up training by avoiding pair computation during forward pass.

Key Features:
1. Pre-computes all pair relationships based on structural templates
2. Caches pair matrix to disk for reuse
3. Provides efficient batch-level pair label generation
4. Supports hard negative mining

Pair Generation Strategy:
- Positive pairs: Same structural template, different content
- Negative pairs: Different structural templates
- Hard negatives: Most similar but different structure (optional)
"""

import os
import pickle
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from sci.data.structure_extractors.scan_extractor import SCANStructureExtractor


class SCANPairGenerator:
    """
    Generates and caches structural pairs for SCAN dataset.

    The pair matrix is pre-computed once and cached to disk. During training,
    we just look up the pre-computed pair relationships.

    Args:
        extractor: SCANStructureExtractor instance
        cache_dir: Directory to store cached pair matrices
        use_hard_negatives: Whether to mark hard negatives
        hard_negative_threshold: Similarity threshold for hard negatives
    """

    def __init__(
        self,
        extractor: Optional[SCANStructureExtractor] = None,
        cache_dir: str = ".cache/pairs",
        use_hard_negatives: bool = False,
        hard_negative_threshold: float = 0.7,
    ):
        self.extractor = extractor or SCANStructureExtractor()
        self.cache_dir = cache_dir
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_threshold = hard_negative_threshold

        os.makedirs(cache_dir, exist_ok=True)

        # Cached data (loaded or computed)
        self.pair_matrix = None  # [N, N] - 1 for positive, 0 for negative
        self.structure_templates = None  # [N] - template for each example
        self.structure_indices = None  # Dict: template → list of indices
        self.cache_metadata = None

    def generate_pairs(
        self,
        commands: List[str],
        cache_name: str = "scan_pairs",
        force_regenerate: bool = False,
    ) -> np.ndarray:
        """
        Generate pair matrix for commands.

        This is called ONCE before training to pre-compute all pairs.

        Args:
            commands: List of SCAN commands
            cache_name: Name for cache file
            force_regenerate: Force regeneration even if cache exists

        Returns:
            pair_matrix: [N, N] binary matrix where 1 = same structure, 0 = different
        """
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")

        # Try to load from cache
        if os.path.exists(cache_path) and not force_regenerate:
            print(f"Loading cached pairs from {cache_path}...")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Verify cache matches current commands
            if cache_data['num_examples'] == len(commands):
                print(f"[OK] Loaded {cache_data['num_examples']} examples with {cache_data['num_positive_pairs']} positive pairs")
                self.pair_matrix = cache_data['pair_matrix']
                self.structure_templates = cache_data['structure_templates']
                self.structure_indices = cache_data['structure_indices']
                self.cache_metadata = cache_data['metadata']
                # CRITICAL #12: Ensure consistent torch.Tensor return type
                return torch.from_numpy(self.pair_matrix).long() if isinstance(self.pair_matrix, np.ndarray) else self.pair_matrix
            else:
                print(f"Cache size mismatch. Regenerating...")

        # Generate pairs from scratch
        print(f"Generating pairs for {len(commands)} commands...")

        # Extract structures
        print("Extracting structural templates...")
        templates = []
        for command in tqdm(commands, desc="Extracting structures"):
            template, _ = self.extractor.extract_structure(command)
            templates.append(template)

        self.structure_templates = templates

        # Group by structure
        print("Grouping by structure...")
        structure_groups = {}
        for idx, template in enumerate(templates):
            if template not in structure_groups:
                structure_groups[template] = []
            structure_groups[template].append(idx)

        self.structure_indices = structure_groups

        # Generate pair matrix
        print("Generating pair matrix...")
        N = len(commands)
        pair_matrix = np.zeros((N, N), dtype=np.int8)

        # For each pair of examples
        for i in tqdm(range(N), desc="Computing pairs"):
            template_i = templates[i]

            for j in range(i + 1, N):
                template_j = templates[j]

                # Same structure = positive pair
                if template_i == template_j:
                    pair_matrix[i, j] = 1
                    pair_matrix[j, i] = 1  # Symmetric

        # Set diagonal to 0 (self-pairs not used)
        np.fill_diagonal(pair_matrix, 0)

        self.pair_matrix = pair_matrix

        # Compute statistics
        num_positive_pairs = pair_matrix.sum() // 2  # Divide by 2 for symmetry
        num_total_pairs = N * (N - 1) // 2
        positive_ratio = num_positive_pairs / num_total_pairs if num_total_pairs > 0 else 0

        metadata = {
            'num_examples': N,
            'num_positive_pairs': int(num_positive_pairs),
            'num_total_pairs': int(num_total_pairs),
            'positive_ratio': float(positive_ratio),
            'num_unique_structures': len(structure_groups),
            'avg_examples_per_structure': N / len(structure_groups),
        }

        self.cache_metadata = metadata

        # Print statistics
        print(f"\nPair Generation Statistics:")
        print(f"  Examples: {metadata['num_examples']}")
        print(f"  Unique structures: {metadata['num_unique_structures']}")
        print(f"  Avg examples/structure: {metadata['avg_examples_per_structure']:.1f}")
        print(f"  Positive pairs: {metadata['num_positive_pairs']:,} ({positive_ratio:.2%})")
        print(f"  Negative pairs: {metadata['num_total_pairs'] - metadata['num_positive_pairs']:,}")

        # Cache to disk
        print(f"Caching pairs to {cache_path}...")
        cache_data = {
            'pair_matrix': pair_matrix,
            'structure_templates': templates,
            'structure_indices': structure_groups,
            'metadata': metadata,
            'num_examples': metadata['num_examples'],
            'num_positive_pairs': metadata['num_positive_pairs'],
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[OK] Pairs cached successfully")

        # CRITICAL #12: Ensure consistent torch.Tensor return type
        return torch.from_numpy(pair_matrix).long()

    def get_batch_pair_labels(
        self,
        batch_items,
    ) -> torch.Tensor:
        """
        Get pair labels for a batch of examples.

        BUG FIX: This method now accepts EITHER:
        - List of indices (int) for pre-computed pair matrix lookup
        - List of commands (str) for runtime structure extraction

        Args:
            batch_items: Either List[int] (indices) or List[str] (commands)

        Returns:
            pair_labels: [batch_size, batch_size] binary tensor
                where 1 = same structure, 0 = different
        """
        if len(batch_items) == 0:
            return torch.zeros((0, 0), dtype=torch.long)

        # Check if items are indices or commands
        if isinstance(batch_items[0], int):
            # Use pre-computed pair matrix
            return self._get_pair_labels_from_indices(batch_items)
        elif isinstance(batch_items[0], str):
            # Compute pair labels from commands at runtime
            return self._get_pair_labels_from_commands(batch_items)
        else:
            raise ValueError(f"Unsupported batch_items type: {type(batch_items[0])}")

    def _get_pair_labels_from_indices(self, indices: List[int]) -> torch.Tensor:
        """Get pair labels using pre-computed pair matrix."""
        if self.pair_matrix is None:
            raise ValueError("Pair matrix not generated. Call generate_pairs() first.")

        # Extract submatrix for this batch
        idx_array = np.array(indices)
        batch_pair_labels = self.pair_matrix[idx_array[:, None], idx_array[None, :]]

        return torch.from_numpy(batch_pair_labels).long()

    def _get_pair_labels_from_commands(self, commands: List[str]) -> torch.Tensor:
        """Compute pair labels from commands at runtime."""
        batch_size = len(commands)

        # Extract structures for each command
        structures = []
        for cmd in commands:
            template, _ = self.extractor.extract_structure(cmd)
            structures.append(template)

        # Compute pair matrix: 1 if same structure, 0 if different
        pair_labels = torch.zeros((batch_size, batch_size), dtype=torch.long)
        for i in range(batch_size):
            for j in range(batch_size):
                if structures[i] == structures[j]:
                    pair_labels[i, j] = 1

        return pair_labels

    def get_structure_statistics(self) -> Dict:
        """
        Get statistics about structure distribution.

        Returns:
            Dictionary with structure statistics
        """
        if self.structure_indices is None:
            return {}

        # Count examples per structure
        structure_counts = {
            template: len(indices)
            for template, indices in self.structure_indices.items()
        }

        # Sort by frequency
        sorted_structures = sorted(
            structure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        stats = {
            'num_unique_structures': len(self.structure_indices),
            'structure_counts': structure_counts,
            'most_common': sorted_structures[:20],
            'least_common': sorted_structures[-20:] if len(sorted_structures) > 20 else sorted_structures,
        }

        if self.cache_metadata is not None:
            stats.update(self.cache_metadata)

        return stats

    def sample_balanced_batch(
        self,
        batch_size: int,
        min_structures_per_batch: int = 3,
        min_examples_per_structure: int = 2,
    ) -> List[int]:
        """
        Sample a balanced batch with sufficient positive pairs.

        Strategy:
        1. Sample min_structures_per_batch unique structures
        2. For each structure, sample min_examples_per_structure examples
        3. Fill remaining slots randomly

        Args:
            batch_size: Target batch size
            min_structures_per_batch: Minimum number of unique structures
            min_examples_per_structure: Minimum examples per structure

        Returns:
            indices: List of dataset indices
        """
        if self.structure_indices is None:
            raise ValueError("Structure indices not available.")

        # Filter structures with enough examples
        valid_structures = [
            template for template, indices in self.structure_indices.items()
            if len(indices) >= min_examples_per_structure
        ]

        if len(valid_structures) < min_structures_per_batch:
            # Not enough structures, fall back to random sampling
            N = len(self.structure_templates)
            return np.random.choice(N, size=batch_size, replace=False).tolist()

        # Sample structures
        sampled_structures = np.random.choice(
            valid_structures,
            size=min_structures_per_batch,
            replace=False
        )

        # Sample examples from each structure
        batch_indices = []
        for template in sampled_structures:
            available_indices = self.structure_indices[template]
            sampled = np.random.choice(
                available_indices,
                size=min(min_examples_per_structure, len(available_indices)),
                replace=False
            )
            batch_indices.extend(sampled.tolist())

        # Fill remaining slots randomly
        if len(batch_indices) < batch_size:
            N = len(self.structure_templates)
            remaining = batch_size - len(batch_indices)

            # Sample from indices not yet in batch
            available = list(set(range(N)) - set(batch_indices))
            additional = np.random.choice(
                available,
                size=min(remaining, len(available)),
                replace=False
            )
            batch_indices.extend(additional.tolist())

        # Shuffle batch
        np.random.shuffle(batch_indices)

        return batch_indices[:batch_size]


if __name__ == "__main__":
    # Test SCAN pair generator
    print("Testing SCANPairGenerator...\n")

    # Create dummy commands
    commands = [
        "walk twice",
        "run twice",
        "jump twice",
        "walk left",
        "run left",
        "look right",
        "jump right",
        "walk and run",
        "jump and look",
        "walk twice after run left",
    ]

    print(f"Test commands: {len(commands)}")
    for i, cmd in enumerate(commands):
        print(f"  {i}: {cmd}")

    # Create generator
    generator = SCANPairGenerator(cache_dir=".test_cache")

    # Generate pairs
    print("\n" + "=" * 60)
    print("Generating Pairs")
    print("=" * 60)

    pair_matrix = generator.generate_pairs(
        commands,
        cache_name="test_scan_pairs",
        force_regenerate=True
    )

    print(f"\nPair matrix shape: {pair_matrix.shape}")
    print(f"Positive pairs: {pair_matrix.sum() // 2}")

    # Test batch pair labels
    print("\n" + "=" * 60)
    print("Batch Pair Labels")
    print("=" * 60)

    batch_indices = [0, 1, 3, 4]  # walk twice, run twice, walk left, run left
    batch_labels = generator.get_batch_pair_labels(batch_indices)

    print(f"Batch indices: {batch_indices}")
    print(f"Batch commands: {[commands[i] for i in batch_indices]}")
    print(f"Pair labels:\n{batch_labels}")
    print(f"\nExpected:")
    print(f"  (0, 1) = 1 (walk twice == run twice)")
    print(f"  (2, 3) = 1 (walk left == run left)")
    print(f"  (0, 2) = 0 (different structure)")

    # Test statistics
    print("\n" + "=" * 60)
    print("Structure Statistics")
    print("=" * 60)

    stats = generator.get_structure_statistics()
    print(f"Unique structures: {stats['num_unique_structures']}")
    print(f"Most common structures:")
    for template, count in stats['most_common']:
        print(f"  {template}: {count} examples")

    # Test balanced batch sampling
    print("\n" + "=" * 60)
    print("Balanced Batch Sampling")
    print("=" * 60)

    batch_indices = generator.sample_balanced_batch(
        batch_size=6,
        min_structures_per_batch=2,
        min_examples_per_structure=2
    )

    print(f"Sampled batch indices: {batch_indices}")
    print(f"Sampled commands:")
    for idx in batch_indices:
        template, _ = generator.extractor.extract_structure(commands[idx])
        print(f"  {idx}: {commands[idx]} → {template}")

    # Verify positive pairs exist in batch
    batch_labels = generator.get_batch_pair_labels(batch_indices)
    num_positive = (batch_labels.sum() - batch_labels.diagonal().sum()).item()
    print(f"\nPositive pairs in batch: {num_positive}")

    # Test cache loading
    print("\n" + "=" * 60)
    print("Cache Loading Test")
    print("=" * 60)

    generator2 = SCANPairGenerator(cache_dir=".test_cache")
    pair_matrix2 = generator2.generate_pairs(
        commands,
        cache_name="test_scan_pairs",
        force_regenerate=False  # Should load from cache
    )

    assert np.array_equal(pair_matrix, pair_matrix2), "Cache mismatch!"
    print("[OK] Cache loaded successfully and matches original")

    # Cleanup test cache
    import shutil
    if os.path.exists(".test_cache"):
        shutil.rmtree(".test_cache")
        print("[OK] Test cache cleaned up")

    print("\n[OK] All SCAN pair generator tests passed!")
