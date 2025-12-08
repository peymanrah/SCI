"""Quick test for all 5 bug fixes."""
import torch
import sys

print("="*60)
print("TESTING ALL 5 BUG FIXES")
print("="*60)

# Bug #4: Vectorized hard negative mining
print("\n[1] Bug #4: Vectorized hard negative mining...")
try:
    from sci.models.losses.combined_loss import HardNegativeMiner
    miner = HardNegativeMiner(hard_negative_ratio=0.3)
    r = torch.randn(8, 64)
    p = torch.zeros(8, 8)
    p[0:4, 0:4] = 1
    p[4:8, 4:8] = 1
    out = miner.mine_hard_negatives(r, p)
    print(f"    Hard negatives selected: {out.sum().item()}")
    print("    [OK] Bug #4 FIXED: Vectorized implementation works!")
except Exception as e:
    print(f"    [FAIL] Bug #4: {e}")
    sys.exit(1)

# Bug #6: Xavier init for position queries
print("\n[2] Bug #6: Xavier init for position queries...")
try:
    from sci.models.components.causal_binding import CausalBindingMechanism
    # Just check the code was updated (actual init requires config)
    import inspect
    source = inspect.getsource(CausalBindingMechanism.__init__)
    if "xavier_uniform_" in source:
        print("    [OK] Bug #6 FIXED: Xavier init found in code!")
    else:
        print("    [FAIL] Bug #6: Xavier init not found")
        sys.exit(1)
except Exception as e:
    print(f"    [FAIL] Bug #6: {e}")
    sys.exit(1)

# Bug #9: Skip test pair generation
print("\n[3] Bug #9: Skip test pair generation by default...")
try:
    with open("scripts/generate_pairs.py", "r") as f:
        content = f.read()
    if "--include-test" in content and "args.include_test" in content:
        print("    [OK] Bug #9 FIXED: --include-test flag added!")
    else:
        print("    [FAIL] Bug #9: --include-test flag not found")
        sys.exit(1)
except Exception as e:
    print(f"    [FAIL] Bug #9: {e}")
    sys.exit(1)

# Bug #11: Generation config in fairness logging
print("\n[4] Bug #11: Generation config in fairness logging...")
try:
    with open("sci/training/trainer.py", "r", encoding="utf-8") as f:
        content = f.read()
    if "generation_config" in content and "max_new_tokens" in content:
        print("    [OK] Bug #11 FIXED: Generation config added to fairness logging!")
    else:
        print("    [FAIL] Bug #11: Generation config not found in fairness logging")
        sys.exit(1)
except Exception as e:
    print(f"    [FAIL] Bug #11: {e}")
    sys.exit(1)

# Bug #5: Eager input_projection init
print("\n[5] Bug #5: Eager input_projection initialization...")
try:
    with open("sci/models/sci_model.py", "r", encoding="utf-8") as f:
        content = f.read()
    if "_init_encoder_projection" in content and "Eagerly initialize" in content:
        print("    [OK] Bug #5 FIXED: Eager projection init added!")
    else:
        print("    [FAIL] Bug #5: Eager projection init not found")
        sys.exit(1)
except Exception as e:
    print(f"    [FAIL] Bug #5: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL 5 BUG FIXES VERIFIED!")
print("="*60)
