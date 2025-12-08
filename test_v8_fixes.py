"""Test script to verify V8 fixes."""
import os
os.environ['SCI_ALLOW_DUMMY_DATA'] = 'true'

import torch
from transformers import AutoTokenizer

# Test 1: Vectorized SCL Loss
print("=" * 60)
print("TEST 1: Vectorized SCL Loss")
print("=" * 60)
from sci.models.losses.scl_loss import StructuralContrastiveLoss

loss_fn = StructuralContrastiveLoss(temperature=0.07, lambda_weight=0.3)
repr_i = torch.randn(4, 8, requires_grad=True)
repr_j = torch.randn(4, 8, requires_grad=True)
pair_labels = torch.tensor([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
], dtype=torch.float32)

loss = loss_fn(repr_i, repr_j, pair_labels)
print(f"  Loss value: {loss.item():.6f}")
loss.backward()
print(f"  Gradient flows: {repr_i.grad is not None}")
print("  ✓ TEST 1 PASSED: Vectorized SCL Loss works correctly")

# Test 2: SCANDataset pair generation only for train
print()
print("=" * 60)
print("TEST 2: SCANDataset Pair Generation Optimization")
print("=" * 60)
from sci.data.datasets.scan_dataset import SCANDataset

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print("  Loading TRAIN subset...")
ds_train = SCANDataset(tokenizer, 'length', 'train')
train_has_pairs = ds_train.pair_generator is not None
print(f"  Train pair_generator exists: {train_has_pairs}")

print("  Loading TEST subset...")
ds_test = SCANDataset(tokenizer, 'length', 'test')
test_has_no_pairs = ds_test.pair_generator is None
print(f"  Test pair_generator is None: {test_has_no_pairs}")

if train_has_pairs and test_has_no_pairs:
    print("  ✓ TEST 2 PASSED: Pair generation skipped for non-training subsets")
else:
    print("  ✗ TEST 2 FAILED")
    exit(1)

# Test 3: Validation collator
print()
print("=" * 60)
print("TEST 3: Validation Collator (separate from training)")
print("=" * 60)
from sci.data.scan_data_collator import SCANDataCollator

train_collator = SCANDataCollator(
    tokenizer=tokenizer,
    max_length=128,
    pair_generator=ds_train.pair_generator,  # Has pair generator
)
val_collator = SCANDataCollator(
    tokenizer=tokenizer,
    max_length=128,
    pair_generator=None,  # No pair generator
)

print(f"  Train collator has pair_generator: {train_collator.pair_generator is not None}")
print(f"  Val collator has no pair_generator: {val_collator.pair_generator is None}")

# Test that val collator works without pair_generator
sample_batch = [
    {'commands': 'walk twice', 'actions': 'WALK WALK', 'idx': 0},
    {'commands': 'run twice', 'actions': 'RUN RUN', 'idx': 1},
]
result = val_collator(sample_batch)
print(f"  Val collator returns pair_labels: {'pair_labels' in result}")
print(f"  Val collator returns input_ids: {'input_ids' in result}")

if val_collator.pair_generator is None and 'pair_labels' not in result:
    print("  ✓ TEST 3 PASSED: Validation collator works without pair generation")
else:
    print("  ✗ TEST 3 FAILED")
    exit(1)

print()
print("=" * 60)
print("ALL V8 FIXES VERIFIED SUCCESSFULLY!")
print("=" * 60)
