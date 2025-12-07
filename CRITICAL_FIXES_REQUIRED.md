# CRITICAL FIXES REQUIRED BEFORE TRAINING

**Status:** Implementation is 85% complete with CRITICAL FIXES NEEDED

---

## IMMEDIATE ACTION ITEMS (Priority 1)

### 1. Install Dependencies ⚠️ BLOCKING
```bash
pip install -r requirements.txt
```

**Current Issue:** pytest not installed, cannot run tests

**Impact:** Cannot verify implementation correctness

---

### 2. Fix Max Generation Length ❌ CRITICAL

**File:** `configs/sci_full.yaml` and `configs/baseline.yaml`

**Current (WRONG):**
```yaml
# Line 132
max_generation_length: 128
```

**Fix (CORRECT):**
```yaml
# Line 132
max_generation_length: 300  # Must be >= 288 for SCAN length split
```

**Why:** SCAN length split has outputs up to 288 tokens. Current limit of 128 will truncate OOD test outputs, making exact match impossible.

**Impact:** Without this fix, OOD evaluation will FAIL completely

---

### 3. Enable EOS Loss ❌ CRITICAL

**File:** `configs/sci_full.yaml`

**Current (WRONG):**
```yaml
# Lines 114-116
use_eos_loss: false
eos_weight: 0.1
```

**Fix (CORRECT):**
```yaml
# Lines 114-116
use_eos_loss: true   # ENABLE
eos_weight: 2.0      # INCREASE (was 0.1)
```

**Why:** From SCI_HYPERPARAMETER_GUIDE.md:
> "EOS PREDICTION LOSS weight: 2.0
> REASONING:
> - CRITICAL for exact match
> - Higher weight emphasizes stopping at correct position
> - Model must learn exact length, not approximate"

**Impact:** Without this, model may not learn to stop at correct position, hurting exact match accuracy

---

### 4. Implement Separate Learning Rates ❌ CRITICAL

**File:** `configs/sci_full.yaml`

**Add after line 83:**
```yaml
training:
  # ... existing ...
  learning_rate: 2e-5         # Base model LR
  sci_learning_rate: 5e-5     # SCI modules LR (2.5x higher)
```

**File:** `sci/training/trainer.py`

**Replace lines 81-85:**
```python
# OLD (WRONG):
self.optimizer = AdamW(
    self.model.parameters(),
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
)

# NEW (CORRECT):
optimizer_groups = self._get_optimizer_groups()
self.optimizer = AdamW(
    optimizer_groups,
    weight_decay=config.training.weight_decay,
)
```

**Add new method to SCITrainer:**
```python
def _get_optimizer_groups(self):
    """Create parameter groups with different learning rates."""
    base_params = []
    sci_params = []
    no_decay_params = []

    base_lr = self.config.training.learning_rate  # 2e-5
    sci_lr = getattr(self.config.training, 'sci_learning_rate', base_lr * 2.5)  # 5e-5

    for name, param in self.model.named_parameters():
        if not param.requires_grad:
            continue

        # No decay for biases and layer norms
        if any(nd in name for nd in ['bias', 'LayerNorm', 'layer_norm']):
            no_decay_params.append(param)
        # SCI modules get higher LR
        elif any(sci in name.lower() for sci in [
            'structural_encoder', 'content_encoder',
            'causal_binding', 'abstraction'
        ]):
            sci_params.append(param)
        # Base model parameters
        else:
            base_params.append(param)

    print(f"Optimizer groups:")
    print(f"  Base params: {len(base_params)} @ LR={base_lr}")
    print(f"  SCI params: {len(sci_params)} @ LR={sci_lr}")
    print(f"  No decay params: {len(no_decay_params)} @ LR={base_lr}, WD=0")

    return [
        {
            'params': base_params,
            'lr': base_lr,
            'weight_decay': self.config.training.weight_decay
        },
        {
            'params': sci_params,
            'lr': sci_lr,
            'weight_decay': self.config.training.weight_decay
        },
        {
            'params': no_decay_params,
            'lr': base_lr,
            'weight_decay': 0.0
        },
    ]
```

**Why:** From SCI_HYPERPARAMETER_GUIDE.md:
> "SCI MODULE LR = 5e-5 (2.5x higher):
> - SCI modules are randomly initialized (no pretrained weights)
> - Need faster learning to catch up with pretrained base
> - Higher LR helps AbstractionLayer learn structuralness quickly"

**Impact:** Without this, SCI modules will learn too slowly and may not reach optimal performance

---

## IMPORTANT FIXES (Priority 2)

### 5. Increase Orthogonality Weight ⚠️ RECOMMENDED

**File:** `configs/sci_full.yaml`

**Current:**
```yaml
# Line 112
ortho_weight: 0.01
```

**Fix:**
```yaml
# Line 112
ortho_weight: 0.1  # 10x increase
```

**Why:** Standard specifies 0.1 for proper structure-content orthogonality

**Impact:** May allow some structure-content entanglement with current value

---

### 6. Add Explicit Generation Penalties ⚠️ RECOMMENDED

**File:** `configs/sci_full.yaml` and `configs/baseline.yaml`

**Add to evaluation section (after line 134):**
```yaml
evaluation:
  # ... existing ...
  max_generation_length: 300
  num_beams: 1
  do_sample: false
  repetition_penalty: 1.0  # ADD: Explicit (no penalty)
  length_penalty: 1.0      # ADD: Explicit (no penalty)
```

**Why:** From SCI_ENGINEERING_STANDARDS.md:
> "MUST BE IDENTICAL for baseline and SCI
> repetition_penalty: 1.0  # No penalty (let model learn naturally)
> length_penalty: 1.0  # No penalty"

**Impact:** Defaults may vary between runs, affecting reproducibility

---

## VERIFICATION ITEMS

### 7. Run All Tests ⚠️ REQUIRED

```bash
# First install dependencies
pip install -r requirements.txt

# Then run tests
python tests/run_tests.py --verbose

# Expected: ALL TESTS PASS ✓
```

**Critical Tests That MUST Pass:**
1. `test_data_leakage.py` - Verifies no response leakage to encoders
2. `test_abstraction_layer.py` - Verifies scores in [0, 1]
3. `test_pair_generation.py` - Verifies structural pairs correct
4. `test_losses.py` - Verifies SCL loss computation

**If any test fails:** DO NOT PROCEED WITH TRAINING

---

### 8. Verify TinyLlama Has RoPE ✓ INFORMATIONAL

TinyLlama-1.1B uses RoPE (Rotary Position Embeddings) natively.

**Verification:**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(model.config.position_embedding_type)
# Should output: 'rope' or similar
```

**Status:** ✓ TinyLlama uses RoPE, no action needed

---

## QUICK FIX SCRIPT

Save this as `apply_critical_fixes.py`:

```python
#!/usr/bin/env python
"""Apply critical fixes to SCI configs."""

import yaml

def fix_config(config_path):
    """Fix a config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Fix 1: Max generation length
    if 'evaluation' in config:
        config['evaluation']['max_generation_length'] = 300
        config['evaluation']['repetition_penalty'] = 1.0
        config['evaluation']['length_penalty'] = 1.0

    # Fix 2: EOS loss (only for sci_full)
    if 'sci_full' in config_path and 'loss' in config:
        config['loss']['use_eos_loss'] = True
        config['loss']['eos_weight'] = 2.0
        config['loss']['ortho_weight'] = 0.1

    # Fix 3: Separate LR (only for sci_full)
    if 'sci_full' in config_path and 'training' in config:
        config['training']['sci_learning_rate'] = 5e-5

    # Save
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Fixed {config_path}")

if __name__ == "__main__":
    fix_config("configs/sci_full.yaml")
    fix_config("configs/baseline.yaml")
    print("\n✓ All critical config fixes applied!")
    print("\nNext steps:")
    print("1. Apply trainer.py changes manually (see CRITICAL_FIXES_REQUIRED.md)")
    print("2. pip install -r requirements.txt")
    print("3. python tests/run_tests.py")
```

Run with:
```bash
python apply_critical_fixes.py
```

---

## SUMMARY

| Fix | Priority | File | Status |
|-----|----------|------|--------|
| 1. Install dependencies | P0 | requirements.txt | ❌ TODO |
| 2. Max generation length | P0 | configs/*.yaml | ❌ TODO |
| 3. Enable EOS loss | P0 | configs/sci_full.yaml | ❌ TODO |
| 4. Separate learning rates | P0 | trainer.py + config | ❌ TODO |
| 5. Orthogonality weight | P1 | configs/sci_full.yaml | ⚠️ RECOMMENDED |
| 6. Generation penalties | P1 | configs/*.yaml | ⚠️ RECOMMENDED |
| 7. Run tests | P0 | - | ❌ BLOCKED |
| 8. Verify RoPE | INFO | - | ✓ CONFIRMED |

**Estimated Time to Fix:** 2-3 hours

**After Fixes:** Ready for full training run

---

**IMPORTANT:** Do NOT start training until ALL P0 (Priority 0) items are completed and all tests pass.

