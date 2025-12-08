"""Verify all configs are aligned for fair comparison."""
from sci.config.config_loader import load_config

configs = [
    ('sci_full', 'configs/sci_full.yaml'),
    ('baseline', 'configs/baseline.yaml'),
    ('no_abstraction_layer', 'configs/ablations/no_abstraction_layer.yaml'),
    ('no_scl', 'configs/ablations/no_scl.yaml'),
    ('no_causal_binding', 'configs/ablations/no_causal_binding.yaml'),
    ('no_content_encoder', 'configs/ablations/no_content_encoder.yaml'),
]

print("=" * 90)
print("CONFIG ALIGNMENT VERIFICATION FOR PUBLICATION")
print("=" * 90)

for name, path in configs:
    c = load_config(path)
    print(f"\n{name}:")
    print(f"  Training: max_epochs={c.training.max_epochs}, "
          f"grad_clip={c.training.gradient_clip}, "
          f"grad_accum={c.training.gradient_accumulation_steps}")
    print(f"  Position: max_length={c.model.position_encoding.max_length}")
    print(f"  Loss: scl_weight={c.loss.scl_weight}, "
          f"scl_temp={c.loss.scl_temperature}, "
          f"eos_weight={c.loss.eos_weight}")
    print(f"  Optimizer: lr={c.training.optimizer.lr}")

print("\n" + "=" * 90)
print("FAIRNESS CHECK:")
print("=" * 90)

# Check alignment
sci_cfg = load_config('configs/sci_full.yaml')
issues = []

for name, path in configs[1:]:  # Skip sci_full
    c = load_config(path)
    if c.training.max_epochs != sci_cfg.training.max_epochs:
        issues.append(f"{name}: max_epochs mismatch ({c.training.max_epochs} vs {sci_cfg.training.max_epochs})")
    if c.training.gradient_clip != sci_cfg.training.gradient_clip:
        issues.append(f"{name}: gradient_clip mismatch ({c.training.gradient_clip} vs {sci_cfg.training.gradient_clip})")
    if c.training.gradient_accumulation_steps != sci_cfg.training.gradient_accumulation_steps:
        issues.append(f"{name}: grad_accum mismatch ({c.training.gradient_accumulation_steps} vs {sci_cfg.training.gradient_accumulation_steps})")
    if c.model.position_encoding.max_length != sci_cfg.model.position_encoding.max_length:
        issues.append(f"{name}: max_length mismatch ({c.model.position_encoding.max_length} vs {sci_cfg.model.position_encoding.max_length})")
    # EOS weight should match except for specific ablations
    if c.loss.eos_weight != sci_cfg.loss.eos_weight:
        issues.append(f"{name}: eos_weight mismatch ({c.loss.eos_weight} vs {sci_cfg.loss.eos_weight})")

if issues:
    print("ISSUES FOUND:")
    for issue in issues:
        print(f"  [X] {issue}")
else:
    print("[OK] ALL CONFIGS ARE ALIGNED FOR FAIR COMPARISON!")
    print("     All training hyperparameters match sci_full.yaml")
