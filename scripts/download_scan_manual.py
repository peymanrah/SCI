"""
Manual SCAN Dataset Download and Setup

Since SCAN is no longer available via HuggingFace's new datasets library,
this script downloads SCAN data directly from the source and creates a
compatible local dataset structure.
"""

import os
import json
import urllib.request
from pathlib import Path

# SCAN data URLs (from original Lake & Baroni 2018 repository)
SCAN_BASE_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/"

SCAN_FILES = {
    "simple": {
        "train": "simple_split/tasks_train_simple.txt",
        "test": "simple_split/tasks_test_simple.txt",
    },
    "length": {
        "train": "length_split/tasks_train_length.txt",
        "test": "length_split/tasks_test_length.txt",
    },
    "addprim_jump": {
        "train": "add_prim_split/tasks_train_addprim_jump.txt",
        "test": "add_prim_split/tasks_test_addprim_jump.txt",
    },
    "addprim_turn_left": {
        "train": "add_prim_split/tasks_train_addprim_turn_left.txt",
        "test": "add_prim_split/tasks_test_addprim_turn_left.txt",
    },
}


def parse_scan_file(content: str):
    """Parse SCAN txt file format: 'IN: command OUT: actions'"""
    examples = []
    for line in content.strip().split('\n'):
        if not line.strip():
            continue

        parts = line.split('OUT:')
        if len(parts) != 2:
            continue

        command = parts[0].replace('IN:', '').strip()
        actions = parts[1].strip()

        examples.append({
            'commands': command,
            'actions': actions,
        })

    return examples


def download_scan_split(split_name: str, subset: str, output_dir: Path):
    """Download a specific SCAN split and subset."""
    if split_name not in SCAN_FILES:
        raise ValueError(f"Unknown split: {split_name}")

    if subset not in SCAN_FILES[split_name]:
        raise ValueError(f"Unknown subset: {subset}")

    url = SCAN_BASE_URL + SCAN_FILES[split_name][subset]

    print(f"Downloading {split_name}/{subset} from {url}...")

    try:
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')

        examples = parse_scan_file(content)

        # Save as JSON
        output_file = output_dir / split_name / f"{subset}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2)

        print(f"  OK: {len(examples)} examples saved to {output_file}")
        return examples

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    """Download all SCAN splits."""
    print("="*70)
    print("SCAN Dataset Manual Download")
    print("="*70)

    project_root = Path(__file__).parent.parent
    output_dir = project_root / ".cache" / "datasets" / "scan"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    total_examples = 0

    for split_name in SCAN_FILES.keys():
        print(f"\n[Split: {split_name}]")

        for subset in ['train', 'test']:
            examples = download_scan_split(split_name, subset, output_dir)
            if examples:
                total_examples += len(examples)

    print("\n" + "="*70)
    print(f"Download Complete: {total_examples} total examples")
    print("="*70)

    # Create a simple dataset loader class
    loader_code = '''"""
Simple SCAN Dataset Loader

Usage:
    from sci.data.scan_loader import load_scan

    dataset = load_scan('length', 'train')
    print(f"Loaded {len(dataset)} examples")
    print(dataset[0])  # {'commands': '...', 'actions': '...'}
"""

import json
from pathlib import Path
from typing import List, Dict

def load_scan(split_name: str = 'length', subset: str = 'train') -> List[Dict[str, str]]:
    """Load SCAN dataset from local cache."""
    project_root = Path(__file__).parent.parent.parent
    scan_dir = project_root / ".cache" / "datasets" / "scan"

    data_file = scan_dir / split_name / f"{subset}.json"

    if not data_file.exists():
        raise FileNotFoundError(
            f"SCAN dataset not found: {data_file}\\n"
            f"Please run: python scripts/download_scan_manual.py"
        )

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    # Test loader
    dataset = load_scan('length', 'train')
    print(f"Loaded {len(dataset)} examples")
    print(f"Sample: {dataset[0]}")
'''

    loader_file = project_root / "sci" / "data" / "scan_loader.py"
    loader_file.parent.mkdir(parents=True, exist_ok=True)

    with open(loader_file, 'w', encoding='utf-8') as f:
        f.write(loader_code)

    print(f"\nCreated loader: {loader_file}")
    print("\nUsage:")
    print("  from sci.data.scan_loader import load_scan")
    print("  dataset = load_scan('length', 'train')")


if __name__ == "__main__":
    main()
