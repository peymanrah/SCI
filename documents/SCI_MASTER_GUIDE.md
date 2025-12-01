# SCI (Structural Causal Invariance) - COMPLETE MASTER IMPLEMENTATION GUIDE

## VERSION: 1.0 | FOR AI AGENT IMPLEMENTATION

---

# TABLE OF CONTENTS

1. [CRITICAL WARNINGS AND ANTI-CHEATING MEASURES](#section-1)
2. [COMPLETE PROJECT STRUCTURE](#section-2)
3. [BENCHMARK STRATEGY AND RATIONALE](#section-3)
4. [AUTOMATED PAIR GENERATION WITHOUT HUMANS](#section-4)
5. [DATA LEAKAGE PREVENTION](#section-5)
6. [IMPLEMENTATION VERIFICATION TESTS](#section-6)
7. [COMPLETE FILE IMPLEMENTATIONS](#section-7)
8. [TRAINING REGIME](#section-8)
9. [EVALUATION REGIME](#section-9)
10. [CONFIGURATION FILES](#section-10)
11. [EXECUTION CHECKLIST](#section-11)

---

<a name="section-1"></a>
# SECTION 1: CRITICAL WARNINGS AND ANTI-CHEATING MEASURES

## 1.1 WHAT THE AI AGENT MUST NOT DO

**READ THIS CAREFULLY. VIOLATIONS WILL CAUSE SCI TO FAIL.**

### ARCHITECTURAL CHEATS TO DETECT AND PREVENT:

```
❌ CHEAT 1: Skipping AbstractionLayer
   - The AbstractionLayer is THE key innovation
   - Without it, SE is just cross-attention (not novel)
   - DETECTION: Check that AbstractionLayer class exists with:
     - structural_detector: nn.Sequential with Sigmoid output
     - residual_gate: nn.Parameter
   - If AbstractionLayer is missing or simplified, FAIL

❌ CHEAT 2: Merging SE and CE into one encoder
   - SE and CE MUST be separate modules
   - They have DIFFERENT purposes (structure vs content)
   - DETECTION: Verify separate class files and separate forward passes
   - If merged, FAIL

❌ CHEAT 3: Implementing CBM as concatenation
   - CBM is NOT just concat(structure, content)
   - It requires: binding_attention, CausalInterventionLayer, broadcast_attention
   - DETECTION: Check for all three sub-components
   - If CBM is just concatenation or simple addition, FAIL

❌ CHEAT 4: Skipping SCL loss
   - SCL is what trains the structural invariance
   - Without SCL, structure won't be invariant to content
   - DETECTION: Verify SCL loss is computed and added to total_loss
   - If scl_loss is always 0 or not computed, FAIL

❌ CHEAT 5: Not masking response tokens
   - SE and CE must ONLY see instruction tokens
   - Response tokens must be masked with -100 in labels
   - DETECTION: Verify instruction_mask is created and applied
   - If SE/CE see response tokens, DATA LEAKAGE = FAIL

❌ CHEAT 6: Using pre-computed or hardcoded pairs
   - Pairs must be generated dynamically based on structure
   - No hardcoded pair_labels
   - DETECTION: Verify pair_generator extracts patterns at runtime
   - If pairs are hardcoded, FAIL

❌ CHEAT 7: Training on test data
   - Training and test splits must be separate
   - No examples from test split in training
   - DETECTION: Verify dataset loading uses correct splits
   - If test data leaks into training, FAIL

❌ CHEAT 8: Simplified edge prediction
   - Edge predictor must compute pairwise relationships
   - Not just uniform or identity edges
   - DETECTION: Verify edge_predictor has learned parameters
   - If edges are uniform/identity, partial FAIL
```

## 1.2 MANDATORY VERIFICATION CHECKPOINTS

The AI agent MUST run verification at these points:

```
CHECKPOINT 1: After implementing each component
  - Run unit tests for that component
  - Verify shapes match expected
  - Verify gradients flow

CHECKPOINT 2: After implementing full model
  - Run scripts/verify_implementation.py
  - ALL checks must pass
  - If any fail, DO NOT proceed to training

CHECKPOINT 3: After first training epoch
  - Verify losses are decreasing
  - Verify SCL loss is non-zero
  - Verify structural representations vary across examples

CHECKPOINT 4: After training completes
  - Run full evaluation suite
  - Compare against baseline
  - Verify improvement on OOD splits
```

## 1.3 IMPLEMENTATION QUALITY ASSERTIONS

Add these assertions to the code to catch implementation errors:

```python
# In StructuralEncoder.__init__
assert hasattr(self, 'abstraction'), "CRITICAL: AbstractionLayer missing!"
assert isinstance(self.abstraction, AbstractionLayer), "CRITICAL: Wrong abstraction type!"

# In ContentEncoder.__init__
assert hasattr(self, 'orthogonal_proj'), "CRITICAL: OrthogonalProjector missing!"

# In CausalBindingMechanism.__init__
assert hasattr(self, 'binding_attention'), "CRITICAL: Binding attention missing!"
assert hasattr(self, 'intervention'), "CRITICAL: CausalInterventionLayer missing!"
assert hasattr(self, 'broadcast_attention'), "CRITICAL: Broadcast attention missing!"

# In SCIModel.forward
assert structural_graph is not None or not self.config['sci']['structural_encoder']['enabled'], \
    "CRITICAL: SE enabled but structural_graph is None!"
assert content is not None or not self.config['sci']['content_encoder']['enabled'], \
    "CRITICAL: CE enabled but content is None!"

# In training loop
assert pair_labels is not None, "CRITICAL: pair_labels missing!"
assert pair_labels.sum() > 0, "CRITICAL: No positive pairs in batch!"
assert (pair_labels.sum(dim=1) > 0).any(), "WARNING: Some examples have no positive pairs"
```

---

<a name="section-2"></a>
# SECTION 2: COMPLETE PROJECT STRUCTURE

## 2.1 DIRECTORY TREE

```
SCI/
│
├── README.md                                    # [CREATE FIRST]
├── requirements.txt                             # [CREATE FIRST]
├── setup.py                                     # [CREATE FIRST]
├── pyproject.toml                               # [CREATE FIRST]
│
├── configs/                                     # [CREATE SECOND]
│   ├── base.yaml                               # Base configuration (inherited by all)
│   ├── sci_tinyllama_scan.yaml                 # SCI + TinyLlama on SCAN
│   ├── sci_tinyllama_cogs.yaml                 # SCI + TinyLlama on COGS  
│   ├── baseline_tinyllama_scan.yaml            # Baseline (no SCI) on SCAN
│   ├── baseline_tinyllama_cogs.yaml            # Baseline on COGS
│   ├── ablation_no_se.yaml                     # Ablation: no Structural Encoder
│   ├── ablation_no_ce.yaml                     # Ablation: no Content Encoder
│   ├── ablation_no_cbm.yaml                    # Ablation: no Causal Binding
│   ├── ablation_no_scl.yaml                    # Ablation: no SCL loss
│   └── ablation_no_abstraction.yaml            # Ablation: SE without AbstractionLayer
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/                                  # [CREATE THIRD - ORDER MATTERS]
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── abstraction_layer.py            # [FILE 1] AbstractionLayer
│   │   │   ├── structural_gnn.py               # [FILE 2] GNN components
│   │   │   ├── edge_predictor.py               # [FILE 3] Edge prediction
│   │   │   ├── content_refiner.py              # [FILE 4] Content refiner
│   │   │   ├── orthogonal_projector.py         # [FILE 5] Orthogonal projection
│   │   │   └── causal_intervention.py          # [FILE 6] Causal intervention
│   │   │
│   │   ├── structural_encoder.py               # [FILE 7] Full SE
│   │   ├── content_encoder.py                  # [FILE 8] Full CE
│   │   ├── causal_binding.py                   # [FILE 9] Full CBM
│   │   └── sci_model.py                        # [FILE 10] Complete SCI model
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── structural_contrastive.py           # [FILE 11] SCL loss
│   │   ├── orthogonality_loss.py               # [FILE 12] Orthogonality loss
│   │   └── combined_loss.py                    # [FILE 13] Combined loss
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── structure_extractors/
│   │   │   ├── __init__.py
│   │   │   ├── base_extractor.py               # [FILE 14] Base class
│   │   │   ├── scan_extractor.py               # [FILE 15] SCAN structure extraction
│   │   │   ├── cogs_extractor.py               # [FILE 16] COGS structure extraction
│   │   │   └── llm_extractor.py                # [FILE 17] LLM-based extraction (fallback)
│   │   │
│   │   ├── pair_generators/
│   │   │   ├── __init__.py
│   │   │   ├── base_generator.py               # [FILE 18] Base pair generator
│   │   │   ├── scan_pair_generator.py          # [FILE 19] SCAN pairs
│   │   │   ├── cogs_pair_generator.py          # [FILE 20] COGS pairs
│   │   │   └── synthetic_generator.py          # [FILE 21] Synthetic pair creation
│   │   │
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── base_dataset.py                 # [FILE 22] Base dataset class
│   │   │   ├── scan_dataset.py                 # [FILE 23] SCAN dataset
│   │   │   ├── cogs_dataset.py                 # [FILE 24] COGS dataset
│   │   │   └── collators.py                    # [FILE 25] Custom collators
│   │   │
│   │   └── download.py                         # [FILE 26] Download datasets
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                          # [FILE 27] Main trainer
│   │   ├── scheduler.py                        # [FILE 28] LR scheduling
│   │   └── callbacks.py                        # [FILE 29] Training callbacks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py                        # [FILE 30] Main evaluator
│   │   ├── metrics.py                          # [FILE 31] Metrics computation
│   │   ├── invariance_tester.py                # [FILE 32] Test structural invariance
│   │   └── analysis.py                         # [FILE 33] Result analysis
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                           # [FILE 34] Config loading
│       ├── logging_utils.py                    # [FILE 35] Logging
│       ├── checkpoint.py                       # [FILE 36] Checkpointing
│       └── seed.py                             # [FILE 37] Random seed
│
├── tests/                                       # [CREATE FOURTH]
│   ├── __init__.py
│   ├── test_abstraction_layer.py               # [TEST 1]
│   ├── test_structural_encoder.py              # [TEST 2]
│   ├── test_content_encoder.py                 # [TEST 3]
│   ├── test_causal_binding.py                  # [TEST 4]
│   ├── test_scl_loss.py                        # [TEST 5]
│   ├── test_pair_generation.py                 # [TEST 6]
│   ├── test_data_leakage.py                    # [TEST 7] CRITICAL
│   ├── test_structural_invariance.py           # [TEST 8] CRITICAL
│   ├── test_full_model.py                      # [TEST 9]
│   └── test_training_loop.py                   # [TEST 10]
│
├── scripts/
│   ├── download_data.sh                        # Download datasets
│   ├── verify_implementation.py                # Verify correct implementation
│   ├── train.py                                # Main training script
│   ├── evaluate.py                             # Evaluation script
│   ├── run_ablations.py                        # Run all ablations
│   ├── run_baseline.py                         # Run baseline experiments
│   ├── compare_results.py                      # Compare SCI vs baseline
│   └── generate_figures.py                     # Generate paper figures
│
├── notebooks/                                   # [OPTIONAL]
│   ├── 01_data_exploration.ipynb
│   ├── 02_structural_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
└── outputs/
    ├── checkpoints/
    │   ├── sci/
    │   ├── baseline/
    │   └── ablations/
    ├── logs/
    ├── results/
    └── figures/
```

## 2.2 FILE CREATION ORDER

**THE ORDER MATTERS. Create files in this exact sequence:**

```
PHASE 1: Setup (5 files)
  1. requirements.txt
  2. setup.py
  3. pyproject.toml
  4. README.md
  5. src/__init__.py (and all other __init__.py files)

PHASE 2: Components (6 files) - Build blocks first
  6. src/models/components/abstraction_layer.py
  7. src/models/components/structural_gnn.py
  8. src/models/components/edge_predictor.py
  9. src/models/components/content_refiner.py
  10. src/models/components/orthogonal_projector.py
  11. src/models/components/causal_intervention.py

PHASE 3: Encoders (4 files) - Use components
  12. src/models/structural_encoder.py
  13. src/models/content_encoder.py
  14. src/models/causal_binding.py
  15. src/models/sci_model.py

PHASE 4: Losses (3 files)
  16. src/losses/structural_contrastive.py
  17. src/losses/orthogonality_loss.py
  18. src/losses/combined_loss.py

PHASE 5: Data Pipeline (12 files)
  19. src/data/structure_extractors/base_extractor.py
  20. src/data/structure_extractors/scan_extractor.py
  21. src/data/structure_extractors/cogs_extractor.py
  22. src/data/structure_extractors/llm_extractor.py
  23. src/data/pair_generators/base_generator.py
  24. src/data/pair_generators/scan_pair_generator.py
  25. src/data/pair_generators/cogs_pair_generator.py
  26. src/data/pair_generators/synthetic_generator.py
  27. src/data/datasets/base_dataset.py
  28. src/data/datasets/scan_dataset.py
  29. src/data/datasets/cogs_dataset.py
  30. src/data/datasets/collators.py

PHASE 6: Training (3 files)
  31. src/training/trainer.py
  32. src/training/scheduler.py
  33. src/training/callbacks.py

PHASE 7: Evaluation (4 files)
  34. src/evaluation/evaluator.py
  35. src/evaluation/metrics.py
  36. src/evaluation/invariance_tester.py
  37. src/evaluation/analysis.py

PHASE 8: Utilities (4 files)
  38. src/utils/config.py
  39. src/utils/logging_utils.py
  40. src/utils/checkpoint.py
  41. src/utils/seed.py

PHASE 9: Tests (10 files) - MUST RUN BEFORE TRAINING
  42-51. All test files

PHASE 10: Configs (10 files)
  52-61. All config files

PHASE 11: Scripts (7 files)
  62-68. All script files
```

---

<a name="section-3"></a>
# SECTION 3: BENCHMARK STRATEGY AND RATIONALE

## 3.1 WHICH BENCHMARKS FOR WHAT

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SCI BENCHMARK STRATEGY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRAINING BENCHMARKS (where we train SCI):                             │
│  ─────────────────────────────────────────                             │
│                                                                         │
│  1. SCAN (PRIMARY)                                                      │
│     - Why: Perfect for SCI - has exact compositional grammar           │
│     - Structure is 100% deterministic                                   │
│     - Pairs can be generated perfectly without LLM                      │
│     - Known compositional generalization challenge                      │
│                                                                         │
│  2. COGS (SECONDARY)                                                    │
│     - Why: Semantic compositionality with logical forms                 │
│     - Structure comes from semantic roles                               │
│     - More natural language-like                                        │
│     - Good generalization test                                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  EVALUATION BENCHMARKS:                                                 │
│  ──────────────────────                                                 │
│                                                                         │
│  IN-DISTRIBUTION:                                                       │
│    - SCAN length (train split) → expect ~98%                           │
│    - COGS in-distribution → expect ~95%                                 │
│                                                                         │
│  COMPOSITIONAL GENERALIZATION (KEY METRIC):                            │
│    - SCAN length (test split, longer sequences) → expect >85%          │
│    - SCAN template (unseen templates) → expect >90%                    │
│    - SCAN addprim_jump (new primitive compositions) → expect >80%      │
│    - COGS gen (generalization split) → expect >70%                     │
│                                                                         │
│  ZERO-SHOT TRANSFER (NOT PRIMARY GOAL):                                │
│    - GSM8K: NOT suitable - math reasoning, not compositional           │
│    - BBH: NOT suitable - too diverse, no clear structure               │
│    - These require different training, not SCI's focus                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.2 WHY NOT GSM8K/BBH FOR SCI

```
SCAN/COGS vs GSM8K/BBH:

SCAN: "walk twice" → "WALK WALK"
  - Structure: "X twice" pattern
  - Content: "walk", "run", "jump"
  - Perfect structural equivalence: "walk twice" ≡ "run twice"
  - SCI can learn this perfectly

GSM8K: "If John has 5 apples and gives 2 to Mary..."
  - Structure: Multi-step arithmetic reasoning
  - Content: Numbers, entities
  - NO clear structural equivalence
  - "5 apples" and "7 oranges" are NOT structurally equivalent
  - SCI's structural invariance doesn't apply

BBH: Diverse reasoning tasks
  - No consistent structure across tasks
  - Each task has different "structure"
  - Would need task-specific structure extractors
  - Not suitable for proving SCI's core claim

CONCLUSION:
- Train on SCAN/COGS to PROVE compositional generalization
- Do NOT claim SCI improves GSM8K/BBH zero-shot
- If evaluated on GSM8K/BBH, expect NO improvement (that's fine)
```

## 3.3 TRAINING REGIME OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING REGIME                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  EXPERIMENT 1: SCAN Primary (Proves core SCI claim)                    │
│  ──────────────────────────────────────────────────                    │
│    Train: SCAN length split (train)                                    │
│    Eval:  SCAN length split (test) - OOD length generalization        │
│           SCAN template split - OOD template generalization            │
│           SCAN addprim_jump - OOD primitive generalization             │
│                                                                         │
│  EXPERIMENT 2: COGS Transfer                                           │
│  ─────────────────────────────                                         │
│    Train: COGS (train)                                                 │
│    Eval:  COGS gen split - semantic generalization                     │
│                                                                         │
│  EXPERIMENT 3: Cross-Dataset Transfer                                  │
│  ────────────────────────────────────                                  │
│    Train: SCAN length                                                  │
│    Eval:  COGS gen (zero-shot) - tests if structural learning         │
│           transfers across datasets                                     │
│                                                                         │
│  ABLATION EXPERIMENTS:                                                 │
│  ────────────────────                                                  │
│    For each experiment, run with:                                      │
│    - Full SCI                                                          │
│    - No SE (structural encoder)                                        │
│    - No CE (content encoder)                                           │
│    - No CBM (causal binding)                                           │
│    - No SCL (contrastive loss)                                         │
│    - No AbstractionLayer (SE without abstraction)                      │
│    - Baseline (no SCI components)                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

<a name="section-4"></a>
# SECTION 4: AUTOMATED PAIR GENERATION WITHOUT HUMANS

## 4.1 THE CORE PROBLEM

```
We need pairs of examples where:
  - POSITIVE: Same structure, different content
  - NEGATIVE: Different structure

We CANNOT use humans to label millions of pairs.
We MUST generate them automatically using:
  1. Rule-based extraction (for SCAN - has known grammar)
  2. Parser-based extraction (for COGS - has logical forms)
  3. LLM-based extraction (fallback for unknown datasets)
```

## 4.2 SCAN PAIR GENERATION (RULE-BASED)

```python
# src/data/structure_extractors/scan_extractor.py
"""
SCAN Structure Extractor - Rule-Based

SCAN has a known context-free grammar. We exploit this for perfect extraction.

GRAMMAR (simplified):
  command → action | action modifier | command 'and' command | command 'after' command
  action → 'walk' | 'run' | 'jump' | 'look'
  modifier → 'left' | 'right' | 'twice' | 'thrice' | 'around' | 'opposite'

STRUCTURE = the pattern of modifiers and connectives
CONTENT = the action words

EXAMPLES:
  "walk twice" → structure: "ACTION twice", content: ["walk"]
  "run twice"  → structure: "ACTION twice", content: ["run"]
  These are POSITIVE pairs (same structure)

  "walk twice" → structure: "ACTION twice"
  "walk and run" → structure: "ACTION and ACTION"
  These are NEGATIVE pairs (different structure)
"""

from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import re


@dataclass
class SCANStructure:
    """Represents extracted structure from SCAN command."""
    template: str           # e.g., "ACTION_0 twice"
    pattern_type: str       # e.g., "modifier_twice"
    content_slots: List[str]  # e.g., ["walk"]
    depth: int              # Nesting depth
    raw_command: str        # Original command


class SCANStructureExtractor:
    """
    Extract structural patterns from SCAN commands.
    
    This is DETERMINISTIC - same input always gives same structure.
    No LLM needed.
    """
    
    # Action words (content) - these get replaced with slots
    ACTIONS: Set[str] = {'walk', 'run', 'jump', 'look'}
    
    # Structural keywords - these define the structure
    MODIFIERS: Set[str] = {'twice', 'thrice', 'around', 'opposite'}
    DIRECTIONS: Set[str] = {'left', 'right'}
    CONNECTIVES: Set[str] = {'and', 'after'}
    
    def __init__(self):
        # Compile patterns for efficiency
        self.action_pattern = re.compile(r'\b(' + '|'.join(self.ACTIONS) + r')\b')
        
    def extract(self, command: str) -> SCANStructure:
        """
        Extract structure from SCAN command.
        
        Algorithm:
        1. Tokenize command
        2. Replace action words with numbered slots
        3. Keep structural words as-is
        4. Compute pattern type and depth
        """
        command = command.lower().strip()
        tokens = command.split()
        
        # Extract content and create template
        template_tokens = []
        content_slots = []
        slot_counter = 0
        
        for token in tokens:
            if token in self.ACTIONS:
                slot_name = f"ACTION_{slot_counter}"
                template_tokens.append(slot_name)
                content_slots.append(token)
                slot_counter += 1
            else:
                template_tokens.append(token)
                
        template = ' '.join(template_tokens)
        
        # Determine pattern type
        pattern_type = self._get_pattern_type(tokens)
        
        # Compute depth (number of nested structures)
        depth = self._compute_depth(tokens)
        
        return SCANStructure(
            template=template,
            pattern_type=pattern_type,
            content_slots=content_slots,
            depth=depth,
            raw_command=command
        )
    
    def _get_pattern_type(self, tokens: List[str]) -> str:
        """Determine the main pattern type."""
        types = []
        
        # Check for modifiers
        for mod in self.MODIFIERS:
            if mod in tokens:
                types.append(f"modifier_{mod}")
                
        # Check for connectives
        for conn in self.CONNECTIVES:
            if conn in tokens:
                types.append(f"connective_{conn}")
                
        # Check for direction changes
        if 'turn' in tokens:
            types.append("turn")
            
        if not types:
            types.append("simple")
            
        return '_'.join(sorted(types))
    
    def _compute_depth(self, tokens: List[str]) -> int:
        """Compute structural depth."""
        depth = 1
        for conn in self.CONNECTIVES:
            depth += tokens.count(conn)
        return depth
    
    def are_structurally_equivalent(
        self, 
        struct1: SCANStructure, 
        struct2: SCANStructure
    ) -> bool:
        """
        Check if two structures are equivalent.
        
        Two structures are equivalent if they have the SAME template
        (ignoring content slots).
        """
        return struct1.template == struct2.template
    
    def generate_structural_variant(
        self,
        structure: SCANStructure,
        new_content: List[str]
    ) -> str:
        """
        Generate a new command with same structure but different content.
        
        This is used for SYNTHETIC pair generation.
        """
        if len(new_content) != len(structure.content_slots):
            raise ValueError(
                f"Content length mismatch: expected {len(structure.content_slots)}, "
                f"got {len(new_content)}"
            )
            
        result = structure.template
        for i, content in enumerate(new_content):
            result = result.replace(f"ACTION_{i}", content)
            
        return result
    
    def get_all_structure_groups(
        self, 
        commands: List[str]
    ) -> Dict[str, List[int]]:
        """
        Group all commands by their structure template.
        
        Returns: dict mapping template -> list of command indices
        """
        groups: Dict[str, List[int]] = {}
        
        for idx, command in enumerate(commands):
            structure = self.extract(command)
            template = structure.template
            
            if template not in groups:
                groups[template] = []
            groups[template].append(idx)
            
        return groups
```

## 4.3 SCAN PAIR GENERATOR

```python
# src/data/pair_generators/scan_pair_generator.py
"""
SCAN Pair Generator - Creates training pairs for SCL

This generates pair_labels for each batch WITHOUT human annotation.

Algorithm:
1. Pre-process entire dataset to extract structures
2. Group examples by structure template
3. For each batch:
   a. Get indices of examples in batch
   b. Create pair_labels matrix: 1 if same structure, 0 otherwise
   c. Optionally augment batch to ensure minimum positive pairs
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random

from .base_generator import BasePairGenerator
from ..structure_extractors.scan_extractor import SCANStructureExtractor, SCANStructure


class SCANPairGenerator(BasePairGenerator):
    """
    Generate structural pairs for SCAN dataset.
    
    USAGE:
        generator = SCANPairGenerator()
        generator.process_dataset(train_commands)  # One-time preprocessing
        
        # During training:
        pair_labels = generator.generate_batch_pairs(batch_indices)
    """
    
    def __init__(
        self,
        min_positives_per_example: int = 2,
        augment_batches: bool = True,
        synthetic_ratio: float = 0.0  # Ratio of synthetic pairs to add
    ):
        super().__init__()
        self.extractor = SCANStructureExtractor()
        self.min_positives = min_positives_per_example
        self.augment_batches = augment_batches
        self.synthetic_ratio = synthetic_ratio
        
        # Populated by process_dataset
        self.structures: List[SCANStructure] = []
        self.template_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.index_to_template: Dict[int, str] = {}
        
    def process_dataset(self, commands: List[str]) -> None:
        """
        Pre-process dataset to extract all structures.
        
        This should be called ONCE before training.
        """
        print(f"Processing {len(commands)} commands for pair generation...")
        
        self.structures = []
        self.template_to_indices = defaultdict(list)
        self.index_to_template = {}
        
        for idx, command in enumerate(commands):
            structure = self.extractor.extract(command)
            self.structures.append(structure)
            
            template = structure.template
            self.template_to_indices[template].append(idx)
            self.index_to_template[idx] = template
            
        # Report statistics
        n_templates = len(self.template_to_indices)
        sizes = [len(indices) for indices in self.template_to_indices.values()]
        
        print(f"  Found {n_templates} unique structural templates")
        print(f"  Group sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        
        # Warn if many singletons
        singletons = sum(1 for s in sizes if s == 1)
        if singletons > 0.5 * n_templates:
            print(f"  WARNING: {singletons} templates have only 1 example (no positive pairs)")
            
    def generate_batch_pairs(
        self, 
        batch_indices: List[int]
    ) -> torch.Tensor:
        """
        Generate pair_labels matrix for a batch.
        
        Args:
            batch_indices: Indices of examples in this batch
            
        Returns:
            pair_labels: [B, B] tensor where 1 = same structure, 0 = different
        """
        B = len(batch_indices)
        pair_labels = torch.zeros(B, B)
        
        for i, idx_i in enumerate(batch_indices):
            template_i = self.index_to_template[idx_i]
            
            for j, idx_j in enumerate(batch_indices):
                if i == j:
                    continue  # Skip self-pairs
                    
                template_j = self.index_to_template[idx_j]
                
                if template_i == template_j:
                    pair_labels[i, j] = 1.0
                    
        return pair_labels
    
    def augment_batch_for_positives(
        self,
        base_indices: List[int],
        target_batch_size: int
    ) -> List[int]:
        """
        Augment batch to ensure minimum positive pairs.
        
        If a batch has examples with no positive pairs in the batch,
        add examples from the same structural group.
        """
        augmented = list(base_indices)
        indices_set = set(augmented)
        
        for idx in base_indices:
            template = self.index_to_template[idx]
            
            # Count existing positives in batch
            same_template_in_batch = sum(
                1 for other in augmented 
                if other != idx and self.index_to_template[other] == template
            )
            
            # Add more if needed
            needed = self.min_positives - same_template_in_batch
            if needed > 0 and len(augmented) < target_batch_size:
                # Find candidates not in batch
                candidates = [
                    other for other in self.template_to_indices[template]
                    if other not in indices_set
                ]
                
                if candidates:
                    to_add = random.sample(
                        candidates, 
                        min(needed, len(candidates), target_batch_size - len(augmented))
                    )
                    augmented.extend(to_add)
                    indices_set.update(to_add)
                    
        return augmented
    
    def get_positive_ratio(self, batch_indices: List[int]) -> float:
        """Calculate ratio of positive pairs in batch."""
        pair_labels = self.generate_batch_pairs(batch_indices)
        B = len(batch_indices)
        total_pairs = B * (B - 1)  # Exclude diagonal
        if total_pairs == 0:
            return 0.0
        return pair_labels.sum().item() / total_pairs
    
    def generate_synthetic_positives(
        self,
        idx: int,
        n_synthetic: int = 5
    ) -> List[str]:
        """
        Generate synthetic positive examples for a given index.
        
        Creates new commands with same structure but different content.
        """
        structure = self.structures[idx]
        n_slots = len(structure.content_slots)
        
        if n_slots == 0:
            return []
            
        available_actions = list(SCANStructureExtractor.ACTIONS - set(structure.content_slots))
        
        synthetic = []
        for _ in range(n_synthetic):
            if not available_actions:
                break
                
            # Random new content
            new_content = [
                random.choice(available_actions) 
                for _ in range(n_slots)
            ]
            
            new_command = self.extractor.generate_structural_variant(structure, new_content)
            synthetic.append(new_command)
            
        return synthetic
```

## 4.4 COGS PAIR GENERATION (PARSER-BASED)

```python
# src/data/structure_extractors/cogs_extractor.py
"""
COGS Structure Extractor - Parser-Based

COGS provides logical forms that we can use to extract structure.

Example:
  Input: "A cat saw a dog"
  Logical Form: "* cat ( x _ 1 ) ; see . agent ( x _ 2 , x _ 1 ) AND see . theme ( x _ 2 , x _ 3 ) ; * dog ( x _ 3 )"
  
  Structure: The predicate pattern (see.agent, see.theme) with argument structure
  Content: The specific nouns (cat, dog)
  
The logical form IS the structure. We extract it by:
1. Removing specific entity mentions (*, cat, dog)
2. Keeping predicates and their argument patterns
"""

import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class COGSStructure:
    """Represents extracted structure from COGS example."""
    predicate_pattern: str    # Pattern of predicates
    argument_pattern: str     # Pattern of arguments
    content_entities: List[str]  # Specific entity words
    raw_sentence: str
    raw_logical_form: str


class COGSStructureExtractor:
    """
    Extract structural patterns from COGS examples.
    
    Uses the logical form to determine structure.
    """
    
    # Entity markers in COGS logical forms
    ENTITY_PATTERN = re.compile(r'\*\s*(\w+)')
    
    # Predicate patterns
    PREDICATE_PATTERN = re.compile(r'(\w+)\s*\.\s*(\w+)\s*\(([^)]+)\)')
    
    def __init__(self):
        self.seen_structures: Set[str] = set()
        
    def extract(
        self, 
        sentence: str, 
        logical_form: Optional[str] = None
    ) -> COGSStructure:
        """
        Extract structure from COGS example.
        """
        if logical_form:
            return self._extract_from_logical_form(sentence, logical_form)
        else:
            return self._extract_heuristic(sentence)
            
    def _extract_from_logical_form(
        self, 
        sentence: str, 
        logical_form: str
    ) -> COGSStructure:
        """Extract structure directly from logical form."""
        # Extract entities (content)
        entities = self.ENTITY_PATTERN.findall(logical_form)
        
        # Create template by replacing entities with placeholders
        template = logical_form
        for i, entity in enumerate(set(entities)):
            template = template.replace(f'* {entity}', f'* ENTITY_{i}')
            
        # Extract predicates
        predicates = self.PREDICATE_PATTERN.findall(logical_form)
        predicate_pattern = '_'.join([f"{p[0]}.{p[1]}" for p in predicates])
        
        # Extract argument pattern (how many args, their relationships)
        args = [p[2] for p in predicates]
        arg_pattern = '_'.join([str(len(a.split(','))) for a in args])
        
        return COGSStructure(
            predicate_pattern=predicate_pattern,
            argument_pattern=arg_pattern,
            content_entities=list(set(entities)),
            raw_sentence=sentence,
            raw_logical_form=logical_form
        )
        
    def _extract_heuristic(self, sentence: str) -> COGSStructure:
        """Fallback: heuristic extraction without logical form."""
        # Very simple: use POS-like heuristics
        words = sentence.lower().split()
        
        # Assume nouns after articles are entities
        entities = []
        for i, word in enumerate(words):
            if i > 0 and words[i-1] in {'a', 'an', 'the'}:
                entities.append(word)
                
        # Create simple template
        template = sentence.lower()
        for i, entity in enumerate(set(entities)):
            template = template.replace(entity, f'ENTITY_{i}')
            
        return COGSStructure(
            predicate_pattern=template,
            argument_pattern='heuristic',
            content_entities=entities,
            raw_sentence=sentence,
            raw_logical_form=''
        )
        
    def are_structurally_equivalent(
        self, 
        struct1: COGSStructure, 
        struct2: COGSStructure
    ) -> bool:
        """Check if two COGS structures are equivalent."""
        # Same predicate pattern = same structure
        return (struct1.predicate_pattern == struct2.predicate_pattern and
                struct1.argument_pattern == struct2.argument_pattern)
```

## 4.5 LLM-BASED EXTRACTION (FALLBACK)

```python
# src/data/structure_extractors/llm_extractor.py
"""
LLM-Based Structure Extractor - Fallback for Unknown Datasets

When we don't have known grammar or logical forms, use an LLM to
reason about structural equivalence.

USAGE: Only when rule-based and parser-based methods are not available.

WARNING: This is slower and less reliable. Use only as fallback.
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass 
class LLMStructure:
    """Structure extracted by LLM reasoning."""
    structural_template: str
    content_words: List[str]
    confidence: float
    reasoning: str


class LLMStructureExtractor:
    """
    Use an LLM to extract and compare structures.
    
    The LLM is prompted to:
    1. Identify structural patterns
    2. Identify content words
    3. Judge structural equivalence
    """
    
    EXTRACTION_PROMPT = """You are a linguistic structure analyzer.

Given an input sentence, extract:
1. STRUCTURAL_TEMPLATE: The abstract syntactic/semantic pattern with content words replaced by placeholders
2. CONTENT_WORDS: The specific words that are "content" (nouns, action words)
3. REASONING: Brief explanation of the structural pattern

Example:
Input: "walk twice"
Output:
{
  "structural_template": "ACTION twice",
  "content_words": ["walk"],
  "reasoning": "Pattern is 'do X twice' where X is an action verb"
}

Example:
Input: "jump left after running right"
Output:
{
  "structural_template": "ACTION DIRECTION after ACTION DIRECTION",
  "content_words": ["jump", "left", "running", "right"],
  "reasoning": "Pattern is 'do X1 in direction D1 after doing X2 in direction D2'"
}

Now analyze this input:
Input: "{input}"
Output (JSON only):"""

    EQUIVALENCE_PROMPT = """Are these two sentences structurally equivalent?
Structurally equivalent means they have the same abstract pattern,
even if the specific words are different.

Sentence 1: "{sent1}"
Sentence 2: "{sent2}"

Answer with just "YES" or "NO" followed by brief reasoning.
Answer:"""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize with a local LLM for extraction.
        
        NOTE: For actual implementation, you may want to use a larger model
        or an API (OpenAI, Anthropic) for better accuracy.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def extract(self, sentence: str) -> LLMStructure:
        """Extract structure using LLM."""
        prompt = self.EXTRACTION_PROMPT.format(input=sentence)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Output (JSON only):")[-1].strip()
        
        try:
            parsed = json.loads(response)
            return LLMStructure(
                structural_template=parsed.get("structural_template", sentence),
                content_words=parsed.get("content_words", []),
                confidence=0.8,  # Heuristic confidence
                reasoning=parsed.get("reasoning", "")
            )
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return LLMStructure(
                structural_template=sentence,
                content_words=[],
                confidence=0.2,
                reasoning="Failed to parse LLM response"
            )
            
    def are_structurally_equivalent(
        self, 
        sent1: str, 
        sent2: str
    ) -> Tuple[bool, str]:
        """Use LLM to judge structural equivalence."""
        prompt = self.EQUIVALENCE_PROMPT.format(sent1=sent1, sent2=sent2)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Answer:")[-1].strip()
        
        is_equivalent = response.upper().startswith("YES")
        reasoning = response
        
        return is_equivalent, reasoning
```

---

<a name="section-5"></a>
# SECTION 5: DATA LEAKAGE PREVENTION

## 5.1 WHAT IS DATA LEAKAGE IN SCI?

```
DATA LEAKAGE = The model sees information it shouldn't during training

For SCI, there are TWO types of leakage to prevent:

TYPE 1: RESPONSE LEAKAGE INTO STRUCTURE
────────────────────────────────────────
  - SE and CE should ONLY see the INSTRUCTION
  - They should NOT see the RESPONSE (target output)
  - If they see the response, they can "cheat" by encoding the answer
  
  Example (SCAN):
    Input: "IN: walk twice OUT: WALK WALK"
    Instruction: "IN: walk twice OUT:"
    Response: "WALK WALK"
    
    SE should ONLY process "IN: walk twice OUT:"
    SE should NOT see "WALK WALK"

TYPE 2: TEST DATA IN TRAINING
─────────────────────────────
  - Training should use ONLY train split
  - Evaluation should use ONLY test/gen split
  - No overlap allowed
```

## 5.2 HOW TO PREVENT RESPONSE LEAKAGE

```python
# In src/models/sci_model.py

class SCIModel(nn.Module):
    
    def get_instruction_mask(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Create mask that is 1 for instruction tokens, 0 for response tokens.
        
        CRITICAL: This is how we prevent data leakage.
        
        The key insight is that in the training data:
        - labels[i] == -100 means token i is NOT a target (instruction token)
        - labels[i] != -100 means token i IS a target (response token)
        
        So instruction_mask = (labels == -100)
        """
        # labels == -100 indicates tokens that are NOT part of the target
        # These are the instruction tokens (what we want SE/CE to see)
        instruction_mask = (labels == -100).long()
        
        return instruction_mask
    
    def forward(self, input_ids, attention_mask, labels, ...):
        # Step 1: Get instruction mask
        instruction_mask = self.get_instruction_mask(input_ids, labels)
        
        # Step 2: Get base embeddings
        base_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Step 3: MASK OUT response tokens for SE and CE
        # This is the CRITICAL step for preventing leakage
        instruction_only_embeddings = base_embeddings * instruction_mask.unsqueeze(-1)
        
        # Step 4: SE only processes instruction
        if self.structural_encoder is not None:
            structural_graph, edge_weights = self.structural_encoder(
                instruction_only_embeddings,  # NOT base_embeddings
                instruction_mask              # Only attend to instruction
            )
            
        # Step 5: CE only processes instruction
        if self.content_encoder is not None:
            instruction_only_ids = input_ids * instruction_mask
            content = self.content_encoder(
                instruction_only_ids,  # NOT input_ids
                structural_graph
            )
```

## 5.3 LABEL CREATION FOR LEAKAGE PREVENTION

```python
# In src/data/datasets/scan_dataset.py

class SCANDataset(Dataset):
    
    def __getitem__(self, idx):
        example = self.raw_data[idx]
        
        command = example['commands']  # "walk twice"
        actions = example['actions']    # "WALK WALK"
        
        # Create input-output format
        # IMPORTANT: The "OUT:" token marks the boundary
        instruction = f"IN: {command} OUT:"
        response = f" {actions}"
        full_text = instruction + response
        
        # Tokenize full sequence
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Find where instruction ends
        instruction_tokens = self.tokenizer(
            instruction,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        instruction_length = instruction_tokens['input_ids'].shape[1]
        
        # Create labels with -100 for instruction tokens
        # -100 tells CrossEntropyLoss to IGNORE these tokens
        labels = tokenized['input_ids'].clone()
        labels[0, :instruction_length] = -100  # Mask instruction
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            # ... other fields
        }
```

## 5.4 VERIFICATION TEST FOR DATA LEAKAGE

```python
# tests/test_data_leakage.py
"""
CRITICAL TEST: Verify no data leakage from response to structure encoding.

This test MUST pass before training.
"""

import torch
import pytest
from transformers import AutoTokenizer


class TestDataLeakage:
    """Tests to verify no data leakage."""
    
    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    def test_labels_mask_instruction(self, tokenizer):
        """Verify that labels correctly mask instruction tokens."""
        instruction = "IN: walk twice OUT:"
        response = " WALK WALK"
        full_text = instruction + response
        
        # Tokenize
        full_tokens = tokenizer(full_text, return_tensors='pt')
        instruction_tokens = tokenizer(instruction, return_tensors='pt')
        
        instruction_length = instruction_tokens['input_ids'].shape[1]
        
        # Create labels
        labels = full_tokens['input_ids'].clone()
        labels[0, :instruction_length] = -100
        
        # Verify
        # All instruction tokens should be -100
        assert (labels[0, :instruction_length] == -100).all(), \
            "Not all instruction tokens are masked!"
            
        # Response tokens should NOT be -100
        assert (labels[0, instruction_length:] != -100).any(), \
            "Response tokens should not all be masked!"
            
    def test_instruction_mask_creation(self):
        """Test that instruction_mask correctly identifies instruction vs response."""
        from src.models.sci_model import SCIModel
        
        # Create minimal config
        config = {
            'model': {'base_model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'},
            'sci': {
                'structural_encoder': {'enabled': False},
                'content_encoder': {'enabled': False},
                'causal_binding': {'enabled': False},
                'contrastive_learning': {'enabled': False}
            },
            'training': {'fp16': False}
        }
        
        model = SCIModel(config)
        
        # Create fake labels
        # First 10 tokens are instruction (-100), next 5 are response
        labels = torch.tensor([[-100]*10 + [1, 2, 3, 4, 5]])
        input_ids = torch.ones_like(labels)
        
        # Get instruction mask
        instruction_mask = model.get_instruction_mask(input_ids, labels)
        
        # Verify
        assert instruction_mask[0, :10].sum() == 10, "Instruction tokens not masked correctly"
        assert instruction_mask[0, 10:].sum() == 0, "Response tokens should not be masked"
        
    def test_se_does_not_see_response(self):
        """
        Verify that structural encoder cannot see response tokens.
        
        Method: Check that changing response tokens does NOT change
        structural representation.
        """
        from src.models.structural_encoder import StructuralEncoder
        
        encoder = StructuralEncoder(d_model=256, n_slots=8, n_layers=2)
        encoder.eval()
        
        # Create fake sequence: [instruction] [response]
        instruction_embeddings = torch.randn(1, 10, 256)
        response_embeddings_v1 = torch.randn(1, 5, 256)
        response_embeddings_v2 = torch.randn(1, 5, 256)  # Different response
        
        # Full sequence with different responses
        full_v1 = torch.cat([instruction_embeddings, response_embeddings_v1], dim=1)
        full_v2 = torch.cat([instruction_embeddings, response_embeddings_v2], dim=1)
        
        # Instruction mask: 1 for instruction, 0 for response
        instruction_mask = torch.tensor([[1]*10 + [0]*5])
        
        with torch.no_grad():
            # Apply instruction mask to embeddings (this is what the model should do)
            masked_v1 = full_v1 * instruction_mask.unsqueeze(-1)
            masked_v2 = full_v2 * instruction_mask.unsqueeze(-1)
            
            # Get structural representations
            struct_v1, _ = encoder(masked_v1, instruction_mask)
            struct_v2, _ = encoder(masked_v2, instruction_mask)
        
        # Representations should be IDENTICAL because response is masked
        diff = (struct_v1 - struct_v2).abs().max().item()
        assert diff < 1e-5, f"SE should not see response! Diff: {diff}"
        
    def test_training_test_split_separation(self):
        """Verify train and test splits are separate."""
        from src.data.datasets.scan_dataset import SCANDataset
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        train_dataset = SCANDataset(tokenizer, split='length', subset='train', max_length=64)
        test_dataset = SCANDataset(tokenizer, split='length', subset='test', max_length=64)
        
        train_commands = set(ex['command'] for ex in train_dataset.raw_data)
        test_commands = set(ex['command'] for ex in test_dataset.raw_data)
        
        overlap = train_commands & test_commands
        
        assert len(overlap) == 0, f"Found {len(overlap)} commands in both train and test!"
```

---

**CONTINUED IN NEXT FILE: SCI_MASTER_GUIDE_PART2.md**

Contains:
- Section 6: Implementation Verification Tests
- Section 7: Complete File Implementations
- Section 8: Training Regime
- Section 9: Evaluation Regime
- Section 10: Configuration Files
- Section 11: Execution Checklist
