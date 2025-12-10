# SCI Codebase Comprehensive Bug Report (V9)
## Line-by-Line Review Following Recent Commits

**Date:** December 10, 2025
**Reviewer:** AI Code Review System
**Focus:** Verification of recent changes and identification of new bugs

---

## EXECUTIVE SUMMARY

After a comprehensive line-by-line review of the codebase following the recent commits, I identified:
- **7 previously reported bugs that are NOW FIXED** ✅
- **5 NEW BUGS discovered** that need attention
- **3 POTENTIAL ISSUES** that should be monitored

The codebase is in **PRODUCTION-READY** state for single-GPU training, but the new bugs should be addressed for optimal performance.

---

## ✅ VERIFIED FIXES FROM PREVIOUS REPORTS

### V8 Critical #1: Validation Collator Pair-Label Mismatch - **FIXED**
**File:** `sci/training/trainer.py` (lines 145-168)
**Fix:** Created separate `val_collator` with `pair_generator=None` to prevent validation from using training pair cache.
```python
self.val_collator = SCANDataCollator(
    tokenizer=self.model.tokenizer,
    max_length=config.data.max_length,
    pair_generator=None,  # No pair generation for validation
    use_chat_template=getattr(config.data, 'use_chat_template', False),
)
```

### V8 Critical #2: Unnecessary Pair Generation for Test Sets - **FIXED**
**File:** `sci/data/datasets/scan_dataset.py` (lines 106-130)
**Fix:** Pair generation is now skipped for non-training subsets.
```python
if subset == 'train':
    # Initialize pair generator
    self.pair_generator = SCANPairGenerator(...)
else:
    # For non-training subsets, skip pair generation
    print(f"Skipping pair generation for '{subset}' subset (not needed for SCL loss)")
    self.pair_generator = None
    self.pair_matrix = None
```

### V8 Performance #3: Non-Vectorized SCL Loss - **FIXED**
**File:** `sci/models/losses/scl_loss.py` (lines 111-132)
**Fix:** Replaced Python loops with vectorized operations using `logsumexp` and masked operations.
```python
# V8 PERFORMANCE #3 FIX: Vectorized NT-Xent loss computation
sim_matrix_masked = similarity_matrix.clone()
sim_matrix_masked.masked_fill_(self_mask, float('-inf'))
log_sum_exp_all = torch.logsumexp(sim_matrix_masked, dim=1)
loss = (loss_matrix * positive_mask.float()).sum() / num_positives.float()
```

### V8 Fix #5: Lazy Projection Initialization - **FIXED**
**File:** `sci/models/sci_model.py` (lines 127-150)
**Fix:** Added eager initialization of encoder projections via `_init_encoder_projection()` method.

### V8 Fix #6: Position Query Initialization - **FIXED**
**File:** `sci/models/components/causal_binding.py` (lines 142-170)
**Fix:** Implemented RoPE-based broadcast for length generalization instead of learned position embeddings.
```python
self.broadcast_pos_encoding = RotaryPositionalEncoding(
    d_model=self.d_model,
    max_length=4096,
    base=10000,
)
```

### V8 Minor #8: Gradient Norm Logging - **FIXED**
**File:** `sci/training/trainer.py` (lines 503-510)
**Fix:** Gradient norm is now logged to wandb.
```python
'train/grad_norm': grad_norm_value,  # V8 #8: Log gradient norm
```

### V8 Issue #9: Test-time Pair Generation in Scripts - **FIXED**
**File:** `scripts/generate_pairs.py` (lines 30-40)
**Fix:** Added validation for split_name and subset parameters with explicit error messages.

---

## 🔴 NEW BUGS DISCOVERED (MUST FIX)

### Bug V9-1: Missing Gradient Accumulation in Trainer
**File:** `sci/training/trainer.py`
**Location:** `train_epoch()` method (lines 423-485)
**Severity:** HIGH

**Issue:** The trainer does NOT implement gradient accumulation despite config having `gradient_accumulation_steps: 2`. The optimizer step is called after every batch instead of accumulating.

**Current Code:**
```python
for batch_idx, batch in enumerate(progress_bar):
    # ... forward pass ...
    loss.backward()
    self.optimizer.step()  # Steps every batch!
    self.optimizer.zero_grad()
```

**Expected:** Accumulate gradients for `gradient_accumulation_steps` before optimizer step.

**Impact:** Effective batch size is half of what's configured (32 instead of 64), affecting contrastive learning quality.

**Fix Required:**
```python
if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()
```

---

### Bug V9-2: train.py Has Gradient Accumulation But Trainer Doesn't
**File:** `train.py` vs `sci/training/trainer.py`
**Severity:** HIGH - INCONSISTENCY

**Issue:** `train.py` correctly implements gradient accumulation (line 168):
```python
grad_accum_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
```
But `SCITrainer` in `trainer.py` does NOT.

**Impact:** Using `SCITrainer` class (recommended) vs `train.py` script produces different training behavior.

---

### Bug V9-3: Structural EOS Not Used in Loss
**File:** `sci/models/sci_model.py` + `sci/models/losses/combined_loss.py`
**Severity:** MEDIUM

**Issue:** The CBM has a `use_structural_eos` flag and EOS predictor head, but:
1. `current_bound_repr` is set in hook but never returned in model outputs
2. `SCICombinedLoss` doesn't use structural EOS predictions

**Impact:** The structural EOS feature advertised in config is not actually affecting training.

**Current Code in CBM hook (line 289):**
```python
# Store bound_repr for structural EOS loss computation
self.current_bound_repr = bound_repr
```
But `forward()` doesn't return this and `combined_loss.py` doesn't use it.

---

### Bug V9-4: Config Mismatch Between sci_full.yaml and base_config.yaml
**Files:** `configs/sci_full.yaml`, `configs/base_config.yaml`
**Severity:** LOW - Documentation

**Issue:** `sci_full.yaml` says it "inherits from base_config.yaml" but doesn't use `_base_` field. The comment is misleading - it's actually a standalone config.

---

### Bug V9-5: Early Stopping Not Using Correct Mode in train.py
**File:** `train.py` (line 436)
**Severity:** LOW

**Issue:** Early stopping is configured with `mode='max'` which is correct for exact_match, but there's no validation that the metric passed is within expected range [0, 1].

---

## 🟡 POTENTIAL ISSUES (MONITOR)

### Issue V9-A: Memory Leak Risk with Hook-stored Tensors
**File:** `sci/models/sci_model.py`
**Location:** `_register_cbm_hooks()` method

**Issue:** `self.current_structural_slots`, `self.current_content_repr`, and `self.current_bound_repr` are stored as instance variables during forward pass. In `generate()`, they're cleared in a try/finally block, but regular `forward()` doesn't clear them.

**Risk:** Memory accumulation during training if garbage collection is delayed.

**Recommendation:** Clear these variables at the start of each forward pass.

---

### Issue V9-B: No Validation of Chat Template Application
**File:** `sci/data/scan_data_collator.py`
**Location:** `_format_instruction()` method

**Issue:** If `use_chat_template=True` but the template produces unexpected output (e.g., includes response tokens), there's no validation.

**Recommendation:** Add assertion that chat-formatted instruction doesn't contain action tokens.

---

### Issue V9-C: Hard-coded Separator Not Documented
**File:** `sci/data/scan_data_collator.py`
**Location:** Constructor

**Issue:** The separator " -> " is hard-coded as default but the config has `use_chat_template: true` which bypasses the separator. This could confuse users.

---

## 📊 VERIFICATION SUMMARY

| Component | Status | Notes |
|-----------|--------|-------|
| Config Loading | ✅ PASS | All configs load correctly |
| SE/CE/CBM Initialization | ✅ PASS | Proper dimensions, projections |
| Instruction Masking | ✅ PASS | Data leakage prevented |
| Pair Generation | ✅ PASS | Only for training subset |
| Validation Collator | ✅ PASS | Separate from training |
| SCL Loss (Vectorized) | ✅ PASS | Efficient implementation |
| Gradient Accumulation | ❌ FAIL | Missing in SCITrainer |
| Structural EOS | ❌ PARTIAL | Hook stores but not used |
| RoPE Broadcast | ✅ PASS | Length generalization enabled |

---

## RECOMMENDED FIX PRIORITY

1. **CRITICAL:** Fix gradient accumulation in `SCITrainer` (Bug V9-1)
2. **HIGH:** Either remove structural EOS code or wire it to loss (Bug V9-3)
3. **MEDIUM:** Add tensor cleanup at start of `forward()` (Issue V9-A)
4. **LOW:** Update config comments (Bug V9-4)

---

## PRODUCTION READINESS

**Status:** CONDITIONALLY READY

The codebase is functional for training but the gradient accumulation bug means effective batch size is 50% of configured. For proper SCL training with the recommended effective batch size of 64, this must be fixed.

To proceed with training as-is:
- Use `batch_size: 64` in config to compensate
- OR use `train.py` script instead of `SCITrainer` class

---

## FILES REVIEWED (KEY FILES)

- `sci/models/sci_model.py` (714 lines) ✅
- `sci/models/losses/scl_loss.py` (281 lines) ✅
- `sci/models/losses/combined_loss.py` (480 lines) ✅
- `sci/training/trainer.py` (763 lines) ✅
- `sci/data/datasets/scan_dataset.py` (335 lines) ✅
- `sci/data/scan_data_collator.py` (241 lines) ✅
- `sci/data/pair_generators/scan_pair_generator.py` (469 lines) ✅
- `sci/models/components/causal_binding.py` (673 lines) ✅
- `sci/models/components/structural_encoder.py` (393 lines) ✅
- `sci/models/components/content_encoder.py` (346 lines) ✅
- `sci/evaluation/scan_evaluator.py` (222 lines) ✅
- `train.py` (554 lines) ✅
- `configs/sci_full.yaml` (147 lines) ✅
- `configs/base_config.yaml` ✅
- `configs/ablations/*.yaml` ✅
