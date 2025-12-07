"""
Download TinyLlama-1.1B model and SCAN benchmark dataset.

This script downloads and caches:
1. TinyLlama-1.1B-Chat-v1.0 model from HuggingFace
2. SCAN benchmark dataset from HuggingFace datasets

All downloads are cached locally in the project structure.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_tinyllama():
    """Download TinyLlama-1.1B model to local cache."""
    print("\n" + "="*70)
    print("STEP 1: Downloading TinyLlama-1.1B-Chat-v1.0 Model")
    print("="*70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_dir = project_root / ".cache" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("\nDownloading model... (this may take a few minutes)")

    try:
        # Download tokenizer
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
        )
        print(f"✓ Tokenizer downloaded: vocab_size={len(tokenizer)}")

        # Download model
        print("\n[2/2] Downloading model (~1.1B parameters, ~4.4GB)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            low_cpu_mem_usage=True,
        )
        print(f"✓ Model downloaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

        print(f"\n✓ TinyLlama successfully cached at: {cache_dir}")
        return True

    except Exception as e:
        print(f"\n✗ Failed to download TinyLlama: {e}")
        return False


def download_scan_dataset():
    """Download SCAN benchmark dataset to local cache."""
    print("\n" + "="*70)
    print("STEP 2: Downloading SCAN Benchmark Dataset")
    print("="*70)

    from datasets import load_dataset

    cache_dir = project_root / ".cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset: scan (from HuggingFace)")
    print(f"Cache directory: {cache_dir}")
    print("\nDownloading SCAN dataset...")

    try:
        # Download all SCAN splits
        splits_to_download = ['simple', 'addprim_jump', 'addprim_turn_left', 'length']

        for split_name in splits_to_download:
            print(f"\n[{splits_to_download.index(split_name)+1}/{len(splits_to_download)}] Downloading '{split_name}' split...")

            dataset = load_dataset(
                "scan",
                split_name,
                cache_dir=str(cache_dir),
            )

            print(f"  ✓ Train: {len(dataset['train'])} examples")
            print(f"  ✓ Test: {len(dataset['test'])} examples")

        print(f"\n✓ SCAN dataset successfully cached at: {cache_dir}")

        # Show sample
        print("\n" + "-"*70)
        print("Sample SCAN example:")
        print("-"*70)
        sample = dataset['train'][0]
        print(f"Commands: {sample['commands']}")
        print(f"Actions:  {sample['actions']}")

        return True

    except Exception as e:
        print(f"\n✗ Failed to download SCAN dataset: {e}")
        return False


def verify_downloads():
    """Verify that downloads are accessible."""
    print("\n" + "="*70)
    print("STEP 3: Verifying Downloads")
    print("="*70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_dir = project_root / ".cache"

    print("\n[1/2] Verifying TinyLlama model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir / "models"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir / "models"),
            low_cpu_mem_usage=True,
        )
        print(f"✓ Model loaded: {model.config.hidden_size}d, {model.config.num_hidden_layers} layers")
        del model  # Free memory

    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False

    print("\n[2/2] Verifying SCAN dataset...")
    try:
        dataset = load_dataset(
            "scan",
            "length",
            cache_dir=str(cache_dir / "datasets"),
        )
        print(f"✓ Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")

    except Exception as e:
        print(f"✗ Dataset verification failed: {e}")
        return False

    print("\n" + "="*70)
    print("✓ All downloads verified successfully!")
    print("="*70)
    return True


def main():
    """Main download script."""
    print("\n" + "="*70)
    print("SCI Model & Data Download Script")
    print("="*70)
    print(f"\nProject root: {project_root}")

    # Step 1: Download TinyLlama
    if not download_tinyllama():
        print("\n✗ TinyLlama download failed. Exiting.")
        return 1

    # Step 2: Download SCAN
    if not download_scan_dataset():
        print("\n✗ SCAN download failed. Exiting.")
        return 1

    # Step 3: Verify
    if not verify_downloads():
        print("\n✗ Verification failed. Exiting.")
        return 1

    print("\n" + "="*70)
    print("✓✓✓ ALL DOWNLOADS COMPLETE ✓✓✓")
    print("="*70)
    print("\nYou can now:")
    print("  1. Generate training pairs: python scripts/generate_pairs.py")
    print("  2. Run tests: pytest tests/")
    print("  3. Start training: python train.py --config configs/sci_full.yaml")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
