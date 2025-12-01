# SCI QUICK REFERENCE CARD
## For AI Agent Implementation

---

## CRITICAL CHECKS (Run Before Every Major Step)

```python
# CHECK 1: AbstractionLayer exists and is correct
assert hasattr(model.structural_encoder, 'abstraction')
assert hasattr(model.structural_encoder.abstraction, 'structural_detector')
assert hasattr(model.structural_encoder.abstraction, 'residual_gate')

# CHECK 2: SCL loss is being computed
assert losses['scl_loss'] > 0, "SCL loss is zero - check pair generation!"

# CHECK 3: No data leakage
assert (labels[:, :instruction_len] == -100).all(), "Instruction not masked!"

# CHECK 4: Pair labels have positives
assert pair_labels.sum() > 0, "No positive pairs in batch!"
```

---

## FILE CREATION ORDER

```
1. requirements.txt
2. src/models/components/abstraction_layer.py  ← KEY FILE
3. src/models/structural_encoder.py
4. src/models/content_encoder.py
5. src/models/causal_binding.py
6. src/models/sci_model.py
7. src/losses/structural_contrastive.py  ← KEY FILE
8. src/data/structure_extractors/scan_extractor.py
9. src/data/pair_generators/scan_pair_generator.py  ← KEY FILE
10. src/data/datasets/scan_dataset.py
11. src/training/trainer.py
12. src/evaluation/evaluator.py
13. configs/*.yaml
14. tests/*.py
15. scripts/*.py
```

---

## ABSTRACTIONLAYER (THE KEY INNOVATION)

```python
class AbstractionLayer(nn.Module):
    def __init__(self, d_model, hidden_mult=2):
        super().__init__()
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.Sigmoid()  # MUST be Sigmoid for [0,1] output
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.1))
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        scores = self.structural_detector(x)  # [0,1] structuralness
        abstracted = x * scores + self.residual_gate * x * (1 - scores)
        return self.layer_norm(abstracted)
```

---

## SCAN STRUCTURE EXTRACTION

```python
# "walk twice" → template: "ACTION_0 twice", content: ["walk"]
# "run twice"  → template: "ACTION_0 twice", content: ["run"]
# SAME template = POSITIVE pair

ACTIONS = {'walk', 'run', 'jump', 'look'}  # Content words
# Everything else is structural
```

---

## PAIR LABELS

```python
# For batch with indices [0, 1, 2, 3]
# If 0,1,2 have "X twice" structure and 3 has "X and Y":
pair_labels = [
    [0, 1, 1, 0],  # 0 matches with 1,2 not 3
    [1, 0, 1, 0],  # 1 matches with 0,2 not 3
    [1, 1, 0, 0],  # 2 matches with 0,1 not 3
    [0, 0, 0, 0],  # 3 matches with nobody
]
```

---

## DATA LEAKAGE PREVENTION

```python
# GOOD: SE only sees instruction
instruction_mask = (labels == -100).long()
instruction_embeddings = base_embeddings * instruction_mask.unsqueeze(-1)
struct_graph, _ = structural_encoder(instruction_embeddings, instruction_mask)

# BAD: SE sees everything (DATA LEAKAGE!)
struct_graph, _ = structural_encoder(base_embeddings, attention_mask)
```

---

## LOSS COMPUTATION

```python
# Combined loss
total_loss = lm_loss + scl_weight * scl_loss + ortho_weight * ortho_loss

# SCL warmup (prevent instability)
if epoch < warmup_epochs:
    scl_weight = base_scl_weight * (epoch + 1) / warmup_epochs
```

---

## BENCHMARKS

| Benchmark | Training | Evaluation | Expected SCI |
|-----------|----------|------------|--------------|
| SCAN length | ✅ | In-dist + OOD | >85% OOD |
| SCAN template | ❌ | OOD only | >90% |
| COGS gen | Optional | Zero-shot | >70% |
| GSM8K | ❌ | NOT suitable | N/A |
| BBH | ❌ | NOT suitable | N/A |

---

## COMMON MISTAKES TO AVOID

| Mistake | How to Detect | Fix |
|---------|---------------|-----|
| No AbstractionLayer | Check SE.abstraction type | Use AbstractionLayer class |
| SCL always 0 | Log scl_loss each step | Fix pair generation |
| Response in SE | Run test_data_leakage | Apply instruction_mask |
| Merged SE+CE | Check class structure | Keep separate modules |
| Simple CBM | Check for 3 attention layers | Add binding+intervention+broadcast |

---

## VERIFICATION COMMANDS

```bash
# Before training
pytest tests/test_abstraction_layer.py -v
pytest tests/test_data_leakage.py -v
pytest tests/test_pair_generation.py -v
python scripts/verify_implementation.py

# During training
# Watch for: SCL loss > 0 and decreasing

# After training
pytest tests/test_structural_invariance.py -v
```

---

## CONFIG QUICK REFERENCE

```yaml
# Full SCI
sci:
  structural_encoder: {enabled: true, n_slots: 8}
  content_encoder: {enabled: true}
  causal_binding: {enabled: true, injection_layers: [6,11,16]}
  contrastive_learning: {enabled: true, scl_weight: 0.1}

# Baseline (no SCI)
sci:
  structural_encoder: {enabled: false}
  content_encoder: {enabled: false}
  causal_binding: {enabled: false}
  contrastive_learning: {enabled: false}
```

---

## SUCCESS CRITERIA

```
✅ SCAN length OOD: >85% (baseline ~20%)
✅ SCAN template: >90% (baseline ~50%)
✅ Same-structure similarity: >0.85
✅ Content substitution similarity: >0.90
✅ SCL loss: starts high, decreases to low stable value
✅ Full SCI > All ablations > Baseline
```
