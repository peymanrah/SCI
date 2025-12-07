#!/usr/bin/env python
"""Fix all ablation config files with critical fixes."""

import yaml
import os

def fix_ablation_config(config_path):
    """Apply critical fixes to an ablation config file."""
    print(f"Fixing {config_path}...")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Fix 1: Add sci_learning_rate if training section exists
    if 'training' in config:
        if 'learning_rate' in config['training']:
            config['training']['sci_learning_rate'] = 5e-5

    # Fix 2: Update orthogonality weight if it exists
    if 'loss' in config and 'ortho_weight' in config['loss']:
        config['loss']['ortho_weight'] = 0.1

    # Fix 3: Add/update evaluation settings
    if 'evaluation' not in config:
        config['evaluation'] = {}

    config['evaluation']['max_generation_length'] = 300
    config['evaluation']['num_beams'] = 1
    config['evaluation']['do_sample'] = False
    config['evaluation']['repetition_penalty'] = 1.0
    config['evaluation']['length_penalty'] = 1.0

    # Save back
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Fixed {config_path}")

def main():
    """Fix all ablation configs."""
    ablation_dir = "configs/ablations"

    if not os.path.exists(ablation_dir):
        print(f"Error: {ablation_dir} not found")
        return

    print("=" * 70)
    print("Fixing Ablation Configs")
    print("=" * 70)

    for filename in os.listdir(ablation_dir):
        if filename.endswith('.yaml'):
            config_path = os.path.join(ablation_dir, filename)
            fix_ablation_config(config_path)

    print("\n" + "=" * 70)
    print("✓ All ablation configs fixed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
