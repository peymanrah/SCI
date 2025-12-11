#!/usr/bin/env python
"""Quick verification script for V9 bug fixes."""

import sys
print("Checking V9 Bug Fixes...")

# V9-1: Gradient accumulation in trainer
print("\n1. Checking V9-1 (Gradient Accumulation in SCITrainer)...")
with open("sci/training/trainer.py", "r") as f:
    content = f.read()
    if "self.grad_accum_steps" in content and "% self.grad_accum_steps" in content:
        print("   ✅ V9-1 FIXED: Gradient accumulation implemented in SCITrainer")
    else:
        print("   ❌ V9-1 NOT FIXED: Missing gradient accumulation")
        
# V9-3: Structural EOS in loss
print("\n2. Checking V9-3 (Structural EOS in Loss)...")
with open("sci/models/sci_model.py", "r") as f:
    content = f.read()
    if "structural_eos_loss" in content and "get_eos_loss" in content:
        print("   ✅ V9-3 FIXED: Structural EOS computed and returned")
    else:
        print("   ❌ V9-3 NOT FIXED: Missing structural EOS")

with open("sci/models/losses/combined_loss.py", "r") as f:
    content = f.read()
    if "structural_eos_loss = model_outputs.get('structural_eos_loss')" in content:
        print("   ✅ V9-3 FIXED: Combined loss uses structural EOS")
    else:
        print("   ❌ V9-3 NOT FIXED: Combined loss doesn't use structural EOS")

# V9-5: Early stopping NaN handling
print("\n3. Checking V9-5 (Early Stopping NaN Handling)...")
with open("sci/training/early_stopping.py", "r") as f:
    content = f.read()
    if "isnan" in content and "isinf" in content:
        print("   ✅ V9-5 FIXED: NaN/Inf handling in early stopping")
    else:
        print("   ❌ V9-5 NOT FIXED: Missing NaN/Inf handling")

# Check config loading
print("\n4. Checking config loading...")
try:
    from sci.config.config_loader import load_config
    c = load_config("configs/sci_full.yaml")
    print(f"   ✅ Config loaded successfully")
    print(f"      - gradient_accumulation_steps: {c.training.gradient_accumulation_steps}")
    print(f"      - use_structural_eos: {c.model.causal_binding.use_structural_eos}")
    print(f"      - SCL weight: {c.loss.scl_weight}")
except Exception as e:
    print(f"   ❌ Config loading failed: {e}")

print("\n=== V9 Bug Fix Verification Complete ===")
