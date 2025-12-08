"""
Generate training pairs for SCAN dataset.

This script generates structural pairs for SCL (Structural Contrastive Learning):
- Same-structure pairs: Different content, same structure
- Different-structure pairs: For contrastive learning

Pairs are cached for efficient training.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci.data.structure_extractors.scan_extractor import SCANStructureExtractor
from sci.data.pair_generators.scan_pair_generator import SCANPairGenerator
from sci.data.scan_loader import load_scan


def generate_pairs_for_split(split_name: str = "length", subset: str = "train"):
    """Generate pairs for a specific SCAN split.
    
    #106 FIX: Add validation for split_name and subset parameters.
    """
    # #106 FIX: Validate split_name
    valid_splits = ['simple', 'length', 'template', 'addprim_jump', 'addprim_turn_left']
    if split_name not in valid_splits:
        raise ValueError(f"Invalid split_name '{split_name}'. Must be one of: {valid_splits}")
    
    # #106 FIX: Validate subset
    valid_subsets = ['train', 'test']
    if subset not in valid_subsets:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of: {valid_subsets}")
    
    print(f"\n{'='*70}")
    print(f"Generating pairs for SCAN {split_name}/{subset}")
    print(f"{'='*70}")

    # Load dataset
    cache_dir = project_root / ".cache" / "datasets" / "scan"
    print(f"\nLoading SCAN dataset from {cache_dir}...")

    data = load_scan(split_name, subset)
    print(f"[OK] Loaded {len(data)} examples")

    # Extract command strings from dicts
    commands = [example['commands'] for example in data]

    # Initialize structure extractor
    print("\nInitializing structure extractor...")
    extractor = SCANStructureExtractor()

    # Initialize pair generator
    print("Initializing pair generator...")
    pairs_cache_dir = project_root / ".cache" / "scan" / split_name / subset
    pairs_cache_dir.mkdir(parents=True, exist_ok=True)

    pair_generator = SCANPairGenerator(
        extractor=extractor,
        cache_dir=str(pairs_cache_dir),
    )

    # Generate pairs
    print(f"\nGenerating pairs (this may take a while)...")
    print(f"Cache directory: {pairs_cache_dir}")

    pairs = pair_generator.generate_pairs(commands)

    print(f"\n[OK] Generated {len(pairs)} pairs")
    print(f"[OK] Pairs cached at: {pairs_cache_dir}")

    return pairs


def validate_pairs(pairs, sample_size: int = 10):
    """Validate generated pairs."""
    print(f"\n{'='*70}")
    print("Validating Generated Pairs")
    print(f"{'='*70}")

    if len(pairs) == 0:
        print("[ERROR] No pairs to validate!")
        return False

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Validating {min(sample_size, len(pairs))} samples...")

    extractor = SCANStructureExtractor()

    valid_count = 0
    for i, (idx1, idx2, label) in enumerate(pairs[:sample_size]):
        # Note: pairs contain indices, not actual examples
        # In real usage, these indices refer to dataset examples

        if label == 1:  # Same structure
            valid_count += 1
            if i < 3:  # Show first 3
                print(f"\n  Pair {i+1}: Same-structure (label=1)")
                print(f"    Index 1: {idx1}")
                print(f"    Index 2: {idx2}")

        elif label == 0:  # Different structure
            valid_count += 1
            if i < 3:
                print(f"\n  Pair {i+1}: Different-structure (label=0)")
                print(f"    Index 1: {idx1}")
                print(f"    Index 2: {idx2}")

    print(f"\n[OK] Validated {valid_count}/{min(sample_size, len(pairs))} pairs")

    # Check pair distribution
    positive_pairs = sum(1 for _, _, label in pairs if label == 1)
    negative_pairs = sum(1 for _, _, label in pairs if label == 0)

    print(f"\nPair distribution:")
    print(f"  Positive (same structure): {positive_pairs} ({positive_pairs/len(pairs)*100:.1f}%)")
    print(f"  Negative (diff structure): {negative_pairs} ({negative_pairs/len(pairs)*100:.1f}%)")

    if positive_pairs == 0:
        print("\n[ERROR] WARNING: No positive pairs found!")
        return False

    if negative_pairs == 0:
        print("\n[ERROR] WARNING: No negative pairs found!")
        return False

    print(f"\n[OK] Pair validation passed")
    return True


def test_structure_extraction():
    """Test structure extraction on sample SCAN commands."""
    print(f"\n{'='*70}")
    print("Testing Structure Extraction")
    print(f"{'='*70}")

    extractor = SCANStructureExtractor()

    test_commands = [
        "jump twice",
        "walk twice",
        "jump thrice",
        "walk and jump",
        "turn left twice and walk",
    ]

    print("\nExtracting structures from sample commands:")

    for cmd in test_commands:
        template, content = extractor.extract_structure(cmd)
        print(f"\n  Command:   '{cmd}'")
        print(f"  Template:  '{template}'")
        print(f"  Content:   {content}")

    # Test same structure detection
    print(f"\n{'-'*70}")
    print("Testing same-structure detection:")

    pairs = [
        ("jump twice", "walk twice", True),  # Same structure
        ("jump thrice", "walk thrice", True),  # Same structure
        ("jump twice", "jump thrice", False),  # Different structure
        ("walk and jump", "look and run", True),  # Same structure
    ]

    correct = 0
    for cmd1, cmd2, expected_same in pairs:
        template1, content1 = extractor.extract_structure(cmd1)
        template2, content2 = extractor.extract_structure(cmd2)
        is_same = (template1 == template2)

        status = "[OK]" if (is_same == expected_same) else "[ERROR]"
        correct += (is_same == expected_same)

        print(f"\n  {status} '{cmd1}' vs '{cmd2}'")
        print(f"     Templates: '{template1}' vs '{template2}'")
        print(f"     Same: {is_same} (expected: {expected_same})")

    print(f"\n{'='*70}")
    print(f"Structure extraction test: {correct}/{len(pairs)} correct")
    print(f"{'='*70}")

    return correct == len(pairs)


def main():
    """Main pair generation script."""
    print(f"\n{'='*70}")
    print("SCI Training Pair Generation Script")
    print(f"{'='*70}")
    print(f"\nProject root: {project_root}")

    # Test structure extraction first
    print("\n[STEP 1/3] Testing structure extraction...")
    if not test_structure_extraction():
        print("\n[ERROR] Structure extraction test failed. Please check the implementation.")
        return 1

    # Generate pairs for training split
    print("\n[STEP 2/3] Generating training pairs...")
    try:
        train_pairs = generate_pairs_for_split(split_name="length", subset="train")
    except Exception as e:
        print(f"\n[ERROR] Failed to generate training pairs: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Validate pairs
    print("\n[STEP 3/3] Validating pairs...")
    print(f"Pair matrix shape: {train_pairs.shape}")
    print(f"Pair matrix is a {train_pairs.shape[0]}x{train_pairs.shape[1]} binary matrix")
    print("[OK] Pair matrix cached successfully. Validation skipped (matrix format).")

    # Generate test pairs
    print(f"\n{'-'*70}")
    print("Generating test pairs...")
    try:
        test_pairs = generate_pairs_for_split(split_name="length", subset="test")
        print(f"Test pair matrix shape: {test_pairs.shape}")
    except Exception as e:
        print(f"\nWarning: Failed to generate test pairs: {e}")
        test_pairs = None

    print(f"\n{'='*70}")
    print("[OK][OK][OK] PAIR GENERATION COMPLETE [OK][OK][OK]")
    print(f"{'='*70}")
    print(f"\nTraining pair matrix: {train_pairs.shape}")
    if test_pairs is not None:
        print(f"Test pair matrix: {test_pairs.shape}")
    else:
        print(f"Test pair matrix: Not generated")
    print(f"\nPairs are cached in: {project_root / '.cache' / 'scan'}")
    print("\nYou can now run training with:")
    print("  python train.py --config configs/sci_full.yaml")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
