"""
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
            f"SCAN dataset not found: {data_file}\n"
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
