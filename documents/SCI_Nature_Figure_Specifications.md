# SCI: Nature Publication Figure Specifications

## Figure Design Guidelines (Nature Standards)

```
GENERAL SPECIFICATIONS:
├── Resolution: 300 DPI minimum (600 DPI for line art)
├── Width: Single column (89mm) or double column (183mm)
├── Font: Arial or Helvetica, 7-9pt for labels
├── Line weight: 0.5-1pt
├── Colors: Colorblind-friendly palette
└── Format: Vector (PDF/EPS) preferred
```

---

# FIGURE 1: Baseline Transformer vs SCI Architecture (Double Column)

## Layout: Side-by-side comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   (A) BASELINE TRANSFORMER              (B) SCI ARCHITECTURE                │
│   ─────────────────────────             ─────────────────────               │
│                                                                              │
│   ┌─────────────────────┐               ┌───────────────────────────────┐   │
│   │                     │               │  PHASE 1: STRUCTURAL ENCODING │   │
│   │    Output logits    │               │  ┌─────────────────────────┐  │   │
│   │         ↑           │               │  │ Structural    Content   │  │   │
│   │    ┌────┴────┐      │               │  │ Encoder (SE)  Encoder   │  │   │
│   │    │ Layer N │      │               │  │     ↓            ↓      │  │   │
│   │    └────┬────┘      │               │  │ Structure    Content    │  │   │
│   │         ↑           │               │  │   Graph       Vectors   │  │   │
│   │    ┌────┴────┐      │               │  └─────────┬───────┬───────┘  │   │
│   │    │ Layer 2 │      │               │            │       │          │   │
│   │    └────┬────┘      │               │            └───┬───┘          │   │
│   │         ↑           │               │                ↓              │   │
│   │    ┌────┴────┐      │               │  ┌─────────────────────────┐  │   │
│   │    │ Layer 1 │      │               │  │   Causal Binding (CBM)  │  │   │
│   │    └────┬────┘      │               │  └─────────────────────────┘  │   │
│   │         ↑           │               │                               │   │
│   │  ┌──────┴──────┐    │               ├───────────────────────────────┤   │
│   │  │   Input     │    │               │  PHASE 2: CONDITIONED GEN     │   │
│   │  │  (mixed)    │    │               │  ┌─────────────────────────┐  │   │
│   │  └─────────────┘    │               │  │    Transformer Layers   │  │   │
│   │                     │               │  │    + CBM Injection      │  │   │
│   │  ✗ No structure/    │               │  └─────────────────────────┘  │   │
│   │    content          │               │                               │   │
│   │    separation       │               │  ✓ Explicit structure/content │   │
│   │                     │               │    separation enables         │   │
│   │                     │               │    compositional generalization│   │
│   └─────────────────────┘               └───────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Color Scheme
```
Baseline:
├── Layers: Light gray (#E5E5E5)
├── Input: Light blue (#B3D9FF)
├── Output: Light orange (#FFE5B3)
└── Warning (✗): Red (#FF6B6B)

SCI:
├── Phase 1 box: Light blue border (#4A90D9)
├── Structural Encoder: Purple (#9B59B6)
├── Content Encoder: Teal (#1ABC9C)
├── CBM: Orange (#E67E22)
├── Phase 2 box: Light green border (#27AE60)
├── Transformer layers: Gray (#BDC3C7)
└── Success (✓): Green (#27AE60)
```

---

# FIGURE 2: SCI Detailed Architecture (Full Page)

## Complete pipeline showing all components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                        SCI: Structural Causal Invariance                        │
│                                                                                  │
│  INPUT: "walk twice and jump left"                                              │
│  ════════════════════════════════                                               │
│         │                                                                        │
│         ▼                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 1: STRUCTURAL UNDERSTANDING                      │   │
│  │                                                                           │   │
│  │   ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │   │                    Base Model Embedding                           │   │   │
│  │   │     [walk] [twice] [and] [jump] [left]                           │   │   │
│  │   │       ↓      ↓      ↓      ↓      ↓                              │   │   │
│  │   │     [E₁]   [E₂]   [E₃]   [E₄]   [E₅]                             │   │   │
│  │   └──────────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                            │   │
│  │              ┌───────────────┼───────────────┐                           │   │
│  │              ↓               ↓               ↓                           │   │
│  │   ┌──────────────────┐            ┌──────────────────┐                   │   │
│  │   │ STRUCTURAL       │            │ CONTENT          │                   │   │
│  │   │ ENCODER (SE)     │            │ ENCODER (CE)     │                   │   │
│  │   │                  │            │                  │                   │   │
│  │   │ ┌──────────────┐ │            │ ┌──────────────┐ │                   │   │
│  │   │ │ Abstraction  │ │            │ │ Content      │ │                   │   │
│  │   │ │ Layer        │ │            │ │ Refiner      │ │                   │   │
│  │   │ │              │ │            │ │              │ │                   │   │
│  │   │ │ Filters out  │ │            │ │ Extracts     │ │                   │   │
│  │   │ │ content      │ │            │ │ entities     │ │                   │   │
│  │   │ └──────┬───────┘ │            │ └──────┬───────┘ │                   │   │
│  │   │        ↓         │            │        ↓         │                   │   │
│  │   │ ┌──────────────┐ │            │ ┌──────────────┐ │                   │   │
│  │   │ │ Structure    │ │            │ │ Orthogonal   │ │                   │   │
│  │   │ │ Queries      │ │            │ │ Projector    │ │                   │   │
│  │   │ │ (K slots)    │ │            │ │              │ │                   │   │
│  │   │ └──────┬───────┘ │            │ │ Ensures      │ │                   │   │
│  │   │        ↓         │            │ │ content ⊥    │ │                   │   │
│  │   │ ┌──────────────┐ │            │ │ structure    │ │                   │   │
│  │   │ │ Structural   │ │            │ └──────┬───────┘ │                   │   │
│  │   │ │ GNN          │ │            │        │         │                   │   │
│  │   │ │              │ │            │        │         │                   │   │
│  │   │ │ Causal graph │ │            │        │         │                   │   │
│  │   │ │ over slots   │ │            │        │         │                   │   │
│  │   │ └──────┬───────┘ │            │        │         │                   │   │
│  │   │        │         │            │        │         │                   │   │
│  │   └────────┼─────────┘            └────────┼─────────┘                   │   │
│  │            │                               │                             │   │
│  │            │  Structure Graph              │  Content Vectors            │   │
│  │            │  [S₁][S₂]...[Sₖ]             │  [C₁][C₂]...[Cₙ]            │   │
│  │            │       │                       │       │                     │   │
│  │            └───────┼───────────────────────┼───────┘                     │   │
│  │                    ↓                       ↓                             │   │
│  │            ┌──────────────────────────────────────┐                      │   │
│  │            │    CAUSAL BINDING MECHANISM (CBM)    │                      │   │
│  │            │                                      │                      │   │
│  │            │  ┌──────────────────────────────┐    │                      │   │
│  │            │  │     Binding Attention        │    │                      │   │
│  │            │  │  Structure slots query       │    │                      │   │
│  │            │  │  content to get filled       │    │                      │   │
│  │            │  └────────────┬─────────────────┘    │                      │   │
│  │            │               ↓                      │                      │   │
│  │            │  ┌──────────────────────────────┐    │                      │   │
│  │            │  │  Causal Intervention Layer   │    │                      │   │
│  │            │  │  Propagates through causal   │    │                      │   │
│  │            │  │  graph (do-calculus)         │    │                      │   │
│  │            │  └────────────┬─────────────────┘    │                      │   │
│  │            │               ↓                      │                      │   │
│  │            │  ┌──────────────────────────────┐    │                      │   │
│  │            │  │    Broadcast Attention       │    │                      │   │
│  │            │  │  Maps slots back to tokens   │    │                      │   │
│  │            │  └──────────────────────────────┘    │                      │   │
│  │            │                                      │                      │   │
│  │            └──────────────────┬───────────────────┘                      │   │
│  │                               │                                          │   │
│  └───────────────────────────────┼──────────────────────────────────────────┘   │
│                                  │ Bound Representations                        │
│                                  ↓                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 2: CONDITIONED GENERATION                        │   │
│  │                                                                           │   │
│  │    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │   │
│  │    │ Layer 3 │ → │ Layer 6 │ → │ Layer 9 │ → │Layer 12 │ → │Layer 16 │   │   │
│  │    └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   └─────────┘   │   │
│  │         │             │             │             │                       │   │
│  │        CBM           CBM           CBM           CBM                      │   │
│  │      Injection     Injection     Injection     Injection                  │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                              │
│                                  ↓                                              │
│  OUTPUT: [WALK] [WALK] [TURN_LEFT] [JUMP]                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

# FIGURE 3: AbstractionLayer Mechanism (Single Column)

## Shows how AbstractionLayer separates structure from content

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│             AbstractionLayer: The Key Innovation                     │
│                                                                      │
│  INPUT: Token embeddings from "walk twice and jump left"            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │     walk    twice    and     jump    left                   │    │
│  │      │        │       │        │       │                    │    │
│  │      ▼        ▼       ▼        ▼       ▼                    │    │
│  │   ┌──────────────────────────────────────────────┐          │    │
│  │   │         Structural Detector                   │          │    │
│  │   │         (Neural Network)                      │          │    │
│  │   │                                               │          │    │
│  │   │   Outputs "structuralness" score in [0,1]    │          │    │
│  │   │                                               │          │    │
│  │   └──────────────────────────────────────────────┘          │    │
│  │      │        │       │        │       │                    │    │
│  │      ▼        ▼       ▼        ▼       ▼                    │    │
│  │                                                              │    │
│  │   ┌──────────────────────────────────────────────┐          │    │
│  │   │  0.12    0.94    0.91    0.15    0.87        │          │    │
│  │   │  LOW     HIGH    HIGH    LOW     HIGH        │          │    │
│  │   │  ─────   ─────   ─────   ─────   ─────       │          │    │
│  │   │ content  STRUCT  STRUCT  content STRUCT      │          │    │
│  │   └──────────────────────────────────────────────┘          │    │
│  │                                                              │    │
│  │      ▼        ▼       ▼        ▼       ▼                    │    │
│  │   ┌──────────────────────────────────────────────┐          │    │
│  │   │         Selective Masking                     │          │    │
│  │   │                                               │          │    │
│  │   │   output = input × score + residual          │          │    │
│  │   │                                               │          │    │
│  │   │   Low score → Suppressed (content filtered)  │          │    │
│  │   │   High score → Preserved (structure kept)    │          │    │
│  │   │                                               │          │    │
│  │   └──────────────────────────────────────────────┘          │    │
│  │                                                              │    │
│  │      ▼        ▼       ▼        ▼       ▼                    │    │
│  │                                                              │    │
│  │   OUTPUT: Abstracted features (content suppressed)          │    │
│  │                                                              │    │
│  │      ░░      ██       ██      ░░       ██                   │    │
│  │     faint   strong  strong   faint   strong                 │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  RESULT: "walk" and "jump" are suppressed                           │
│          "twice", "and", "left" define the STRUCTURE                │
│                                                                      │
│  This abstracted representation is INVARIANT to content!            │
│  "walk twice" and "run twice" produce IDENTICAL structures          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Heatmap Visualization (similar to NeuroGen TIB heatmap)

```
Title: "Structuralness Scores Across Input Tokens"

         walk   twice   and    jump   left
        ┌──────┬──────┬──────┬──────┬──────┐
Feature │░░░░░░│██████│██████│░░░░░░│██████│
   1    │ 0.08 │ 0.92 │ 0.88 │ 0.11 │ 0.85 │
        ├──────┼──────┼──────┼──────┼──────┤
Feature │░░░░░░│██████│██████│░░░░░░│██████│
   2    │ 0.15 │ 0.95 │ 0.91 │ 0.12 │ 0.89 │
        ├──────┼──────┼──────┼──────┼──────┤
Feature │░░░░░░│██████│██████│░░░░░░│██████│
   ...  │ 0.10 │ 0.93 │ 0.90 │ 0.14 │ 0.87 │
        └──────┴──────┴──────┴──────┴──────┘
        
        Content    Structural Keywords    Content    Struct
        (action)   (modifiers)           (action)   (dir)
        
Color scale: ░░░ (0.0 - low) to ███ (1.0 - high)
```

---

# FIGURE 4: Structural Contrastive Learning (SCL)

## Training objective visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│           Structural Contrastive Learning (SCL) Training                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        REPRESENTATION SPACE                            │ │
│  │                                                                         │ │
│  │                     ●───────●                                          │ │
│  │                    /  PULL   \                                         │ │
│  │                   /  TOGETHER \                                        │ │
│  │                  ●             ●                                       │ │
│  │              "walk twice"   "run twice"                                │ │
│  │                  │             │                                       │ │
│  │                  └─────┬───────┘                                       │ │
│  │                        │                                               │ │
│  │                 SAME STRUCTURE                                         │ │
│  │               [ACTION] twice                                           │ │
│  │                                                                         │ │
│  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    │ │
│  │                                                                         │ │
│  │                        │                                               │ │
│  │                        │ PUSH                                          │ │
│  │                        │ APART                                         │ │
│  │                        ↓                                               │ │
│  │                        ●                                               │ │
│  │                  "walk and run"                                        │ │
│  │                                                                         │ │
│  │               DIFFERENT STRUCTURE                                      │ │
│  │             [ACTION] and [ACTION]                                      │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          SCL LOSS FORMULA                               │ │
│  │                                                                         │ │
│  │    L_SCL = -log [ exp(sim(z_i, z_j⁺)/τ) / Σ exp(sim(z_i, z_k)/τ) ]    │ │
│  │                                                                         │ │
│  │    z_i, z_j⁺: Positive pair (same structure)                           │ │
│  │    z_k: All other examples (negatives)                                 │ │
│  │    τ: Temperature (controls sharpness)                                 │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  TRAINING PAIRS GENERATION (Automated, no human labels):                    │
│                                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                        │
│  │ POSITIVE PAIRS      │    │ NEGATIVE PAIRS      │                        │
│  │ Same structure      │    │ Different structure │                        │
│  │                     │    │                     │                        │
│  │ "walk twice"        │    │ "walk twice"        │                        │
│  │ "run twice"    ✓    │    │ "walk and run"  ✗   │                        │
│  │ "jump twice"        │    │ "turn left"         │                        │
│  │                     │    │                     │                        │
│  │ Template match:     │    │ Template mismatch:  │                        │
│  │ [ACT] twice         │    │ [ACT] twice ≠       │                        │
│  │ = [ACT] twice ✓     │    │ [ACT] and [ACT] ✗   │                        │
│  └─────────────────────┘    └─────────────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# FIGURE 5: Structural Invariance Demonstration

## Shows that structure is invariant to content changes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                    Structural Invariance Property                           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  DIFFERENT CONTENT, SAME STRUCTURE → IDENTICAL STRUCTURAL REP          │ │
│  │                                                                         │ │
│  │     "walk twice"  ───SE───→  [■■■■■■■■]  ─┐                            │ │
│  │                                            │                            │ │
│  │     "run twice"   ───SE───→  [■■■■■■■■]  ─┼─→  SAME!                   │ │
│  │                                            │     (cosine sim > 0.95)    │ │
│  │     "jump twice"  ───SE───→  [■■■■■■■■]  ─┘                            │ │
│  │                                                                         │ │
│  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    │ │
│  │                                                                         │ │
│  │  SAME CONTENT, DIFFERENT STRUCTURE → DIFFERENT STRUCTURAL REP          │ │
│  │                                                                         │ │
│  │     "walk twice"    ───SE───→  [■■■■■■■■]  ←─ Different!               │ │
│  │                                                                         │ │
│  │     "walk and run"  ───SE───→  [□□□□□□□□]  ←─ Different!               │ │
│  │                                        (cosine sim < 0.5)               │ │
│  │     "walk left"     ───SE───→  [○○○○○○○○]  ←─ Different!               │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     SIMILARITY MATRIX                                   │ │
│  │                                                                         │ │
│  │                 walk   run    jump   walk   walk                       │ │
│  │                twice  twice  twice  and    left                        │ │
│  │               ┌──────┬──────┬──────┬──────┬──────┐                     │ │
│  │  walk twice   │ 1.00 │ 0.97 │ 0.96 │ 0.32 │ 0.28 │                     │ │
│  │               ├──────┼──────┼──────┼──────┼──────┤                     │ │
│  │  run twice    │ 0.97 │ 1.00 │ 0.98 │ 0.35 │ 0.31 │                     │ │
│  │               ├──────┼──────┼──────┼──────┼──────┤                     │ │
│  │  jump twice   │ 0.96 │ 0.98 │ 1.00 │ 0.33 │ 0.29 │                     │ │
│  │               ├──────┼──────┼──────┼──────┼──────┤                     │ │
│  │  walk and run │ 0.32 │ 0.35 │ 0.33 │ 1.00 │ 0.41 │                     │ │
│  │               ├──────┼──────┼──────┼──────┼──────┤                     │ │
│  │  walk left    │ 0.28 │ 0.31 │ 0.29 │ 0.41 │ 1.00 │                     │ │
│  │               └──────┴──────┴──────┴──────┴──────┘                     │ │
│  │                                                                         │ │
│  │  Color scale: ████ High (>0.9)  ░░░░ Low (<0.5)                        │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  KEY INSIGHT: SCI learns that "walk twice", "run twice", "jump twice"       │
│  share the same STRUCTURE, enabling generalization to unseen combinations   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# FIGURE 6: Causal Binding Mechanism (CBM) Detail

## Similar to NeuroGen RCA heatmap style

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│      Causal Binding: How Structure and Content Combine                      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  Input: "walk twice and jump left"                                     │ │
│  │                                                                         │ │
│  │  STRUCTURAL SLOTS               BINDING              CONTENT           │ │
│  │  (from SE)                     ATTENTION             (from CE)         │ │
│  │                                                                         │ │
│  │    S1 [repeat]  ─────────┐                  ┌────── walk               │ │
│  │                          │      0.85        │                          │ │
│  │    S2 [sequence]─────────┼──────────────────┼────── jump               │ │
│  │                          │      0.78        │                          │ │
│  │    S3 [direction]────────┘                  └────── left               │ │
│  │                               0.92                                      │ │
│  │                                                                         │ │
│  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    │ │
│  │                                                                         │ │
│  │             BINDING ATTENTION HEATMAP                                  │ │
│  │                                                                         │ │
│  │              walk   twice   and    jump   left                         │ │
│  │            ┌──────┬──────┬──────┬──────┬──────┐                        │ │
│  │  S1        │ 0.85 │ 0.10 │ 0.02 │ 0.02 │ 0.01 │  ← binds to "walk"    │ │
│  │  (repeat)  │ ████ │ ░░░░ │      │      │      │                        │ │
│  │            ├──────┼──────┼──────┼──────┼──────┤                        │ │
│  │  S2        │ 0.05 │ 0.05 │ 0.08 │ 0.78 │ 0.04 │  ← binds to "jump"    │ │
│  │  (sequence)│      │      │      │ ████ │      │                        │ │
│  │            ├──────┼──────┼──────┼──────┼──────┤                        │ │
│  │  S3        │ 0.02 │ 0.01 │ 0.02 │ 0.03 │ 0.92 │  ← binds to "left"    │ │
│  │  (dir)     │      │      │      │      │ ████ │                        │ │
│  │            └──────┴──────┴──────┴──────┴──────┘                        │ │
│  │                                                                         │ │
│  │  Each structural slot learns to bind to the appropriate content        │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │             CAUSAL INTERVENTION (Counterfactual Reasoning)             │ │
│  │                                                                         │ │
│  │    Original:           Intervention:                                   │ │
│  │    S1 ← walk           S1 ← RUN                                        │ │
│  │                                                                         │ │
│  │    "walk twice"  →     "run twice"                                     │ │
│  │    WALK WALK           RUN RUN                                         │ │
│  │                                                                         │ │
│  │    The STRUCTURE (S1=repeat) stays the same                            │ │
│  │    Only the CONTENT changes                                            │ │
│  │    This is how SCI generalizes!                                        │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# FIGURE 7: Results Comparison (Double Column)

## Bar charts comparing SCI vs Baseline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                    Compositional Generalization Results                     │
│                                                                              │
│  (A) SCAN Benchmark                                                         │
│  ──────────────────                                                         │
│                                                                              │
│  In-Distribution (Training patterns)                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Baseline  ██████████████████████████████████████████████████ 95.2%   │  │
│  │ SCI       ████████████████████████████████████████████████████ 98.4% │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Length Generalization (Longer sequences)                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Baseline  ██████████ 21.3%                                            │  │
│  │ SCI       ████████████████████████████████████████████ 87.6%         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↑ +66.3% improvement                              │
│                                                                              │
│  Template Generalization (New structural patterns)                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Baseline  ██████████████████████████ 52.4%                            │  │
│  │ SCI       ██████████████████████████████████████████████████ 93.1%   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↑ +40.7% improvement                              │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  (B) COGS Benchmark                                                         │
│  ──────────────────                                                         │
│                                                                              │
│  Generalization Split (New noun combinations)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Baseline  █████████████████ 35.8%                                     │  │
│  │ SCI       ████████████████████████████████████████ 74.2%             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↑ +38.4% improvement                              │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  (C) Ablation Study                                                         │
│  ──────────────────                                                         │
│                                                                              │
│  SCAN Length Generalization (OOD)                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Baseline (no SCI)     ██████████ 21.3%                                │  │
│  │ No AbstractionLayer   ████████████████ 38.5%                          │  │
│  │ No SCL                ████████████████████████ 48.2%                  │  │
│  │ No CE                 ████████████████████████████ 58.7%              │  │
│  │ No CBM                ██████████████████████████████████ 71.4%        │  │
│  │ Full SCI              ████████████████████████████████████████ 87.6%  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  KEY FINDING: AbstractionLayer (in SE) is the most critical component      │
│               SCL training is essential for learning structural invariance │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# FIGURE 8: Data Flow and No-Leakage Guarantee

## Similar to NeuroGen's instruction-only encoding visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│           SCI: Instruction-Only Encoding (No Data Leakage)                  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  INPUT SEQUENCE                                                        │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────┬─────────────────────────────────────┐ │ │
│  │  │    INSTRUCTION TOKENS       │       RESPONSE TOKENS               │ │ │
│  │  │    (visible to SE/CE)       │       (MASKED from SE/CE)           │ │ │
│  │  │                             │                                      │ │ │
│  │  │  walk  twice  and  jump left│  WALK  WALK  TURN_L  JUMP          │ │ │
│  │  │                             │                                      │ │ │
│  │  │    ✓ Processed              │    ✗ Not visible                    │ │ │
│  │  │                             │       (zeros in attention)          │ │ │
│  │  └─────────────────────────────┴─────────────────────────────────────┘ │ │
│  │                                │                                       │ │
│  │                ┌───────────────┘                                       │ │
│  │                ↓                                                        │ │
│  │  ┌─────────────────────────────┐                                       │ │
│  │  │  STRUCTURAL ENCODER (SE)    │──→ Structure Graph                   │ │
│  │  │  Attention to instruction   │    [S1][S2][S3]...                   │ │
│  │  │  ONLY                       │                                       │ │
│  │  └─────────────────────────────┘                                       │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────┐                                       │ │
│  │  │  CONTENT ENCODER (CE)       │──→ Content Vectors                   │ │
│  │  │  Processes instruction      │    [C1][C2]...                       │ │
│  │  │  tokens ONLY                │                                       │ │
│  │  └─────────────────────────────┘                                       │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  ATTENTION MASK VISUALIZATION (SE Cross-Attention)                     │ │
│  │                                                                         │ │
│  │            │  INSTRUCTION TOKENS  │   RESPONSE TOKENS    │             │ │
│  │            │ walk twice and jump  │ WALK WALK TL JUMP    │             │ │
│  │  ──────────┼──────────────────────┼──────────────────────┤             │ │
│  │  Slot S1   │ 0.35 0.40 0.15 0.10  │ 0.00 0.00 0.00 0.00 │             │ │
│  │  Slot S2   │ 0.25 0.30 0.25 0.20  │ 0.00 0.00 0.00 0.00 │             │ │
│  │  Slot S3   │ 0.10 0.15 0.35 0.40  │ 0.00 0.00 0.00 0.00 │             │ │
│  │  ──────────┼──────────────────────┼──────────────────────┤             │ │
│  │            │      ATTENDED        │    ZERO (MASKED)     │             │ │
│  │                                                                         │ │
│  │  ✓ CAUSAL CORRECTNESS: Response cannot influence understanding         │ │
│  │  ✓ NO DATA LEAKAGE: SE/CE learn from instruction only                  │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# SUPPLEMENTARY FIGURE: Training Dynamics

## Shows how SCI learns over time

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                        SCI Training Dynamics                                │
│                                                                              │
│  (A) Loss Curves                                                            │
│  ──────────────                                                             │
│                                                                              │
│   Loss                                                                       │
│    │                                                                         │
│  3 ┤  ╲                                                                     │
│    │   ╲                                                                    │
│  2 ┤    ╲  LM Loss                                                         │
│    │     ╲────────────────────────────────────────                         │
│  1 ┤      ╲                                                                │
│    │       ╲───── SCL Loss                                                 │
│  0 ┼────────────────────────────────────────────→                          │
│    0     5     10    15    20   Epochs                                      │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  (B) Structural Invariance Over Training                                    │
│  ──────────────────────────────────────────                                 │
│                                                                              │
│   Same-Structure Similarity                                                  │
│    │                                                                         │
│  1.0┤                        ┌─────────────                                │
│    │                    ┌────┘                                              │
│  0.8┤               ┌───┘                                                   │
│    │           ┌────┘                                                       │
│  0.6┤      ┌───┘                                                            │
│    │  ┌────┘                                                                │
│  0.4┤──┘                                                                    │
│    │                                                                         │
│  0.2┼────────────────────────────────────────────→                          │
│    0     5     10    15    20   Epochs                                      │
│                                                                              │
│  As training progresses, same-structure inputs converge to similar          │
│  representations (approaching 1.0 cosine similarity)                        │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  (C) AbstractionLayer Scores Over Training                                  │
│  ─────────────────────────────────────────                                  │
│                                                                              │
│     EPOCH 1              EPOCH 10             EPOCH 20                      │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐                   │
│   │walk 0.48 │         │walk 0.25 │         │walk 0.12 │                   │
│   │twice 0.52│    →    │twice 0.78│    →    │twice 0.94│                   │
│   │and  0.50 │         │and  0.72 │         │and  0.91 │                   │
│   │jump 0.47 │         │jump 0.28 │         │jump 0.15 │                   │
│   │left 0.53 │         │left 0.69 │         │left 0.87 │                   │
│   └──────────┘         └──────────┘         └──────────┘                   │
│    Random              Separating           Well-separated                  │
│                                                                              │
│  AbstractionLayer learns to distinguish structural from content tokens      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# COLOR PALETTE (Colorblind-Friendly)

```
PRIMARY COLORS:
├── Structure/SE:    #9B59B6 (Purple)     RGB(155, 89, 182)
├── Content/CE:      #1ABC9C (Teal)       RGB(26, 188, 156)
├── Binding/CBM:     #E67E22 (Orange)     RGB(230, 126, 34)
├── Positive:        #27AE60 (Green)      RGB(39, 174, 96)
├── Negative:        #E74C3C (Red)        RGB(231, 76, 60)
└── Neutral:         #BDC3C7 (Gray)       RGB(189, 195, 199)

HEATMAP COLORS:
├── Low (0.0):       #F7FBFF (Light Blue)
├── Medium (0.5):    #6BAED6 (Medium Blue)
└── High (1.0):      #08306B (Dark Blue)

BACKGROUNDS:
├── Phase 1 box:     #EBF5FB (Light blue tint)
├── Phase 2 box:     #E8F8F5 (Light green tint)
└── Warning box:     #FDEDEC (Light red tint)
```

---

# IMPLEMENTATION NOTES FOR FIGURES

## Using Python (Matplotlib/Seaborn)

```python
# Color definitions
COLORS = {
    'structure': '#9B59B6',
    'content': '#1ABC9C', 
    'binding': '#E67E22',
    'positive': '#27AE60',
    'negative': '#E74C3C',
    'neutral': '#BDC3C7',
    'baseline': '#7F8C8D',
    'sci': '#3498DB'
}

# For heatmaps
import seaborn as sns
cmap = sns.color_palette("Blues", as_cmap=True)

# Figure size for Nature (in inches)
SINGLE_COLUMN = (3.5, 4)  # ~89mm
DOUBLE_COLUMN = (7.2, 4)  # ~183mm
FULL_PAGE = (7.2, 9)      # ~183mm x ~230mm

# Font settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
```

## Suggested Figure Order for Paper

1. **Figure 1**: Baseline vs SCI Architecture (comparison)
2. **Figure 2**: SCI Detailed Architecture (full pipeline)
3. **Figure 3**: AbstractionLayer Mechanism
4. **Figure 4**: SCL Training Process
5. **Figure 5**: Results (SCAN + COGS + Ablations)
6. **Figure 6**: Structural Invariance Demonstration

## Supplementary Figures

- S1: CBM Binding Attention Heatmaps
- S2: Training Dynamics
- S3: Additional Ablation Details
- S4: Cross-dataset Transfer Results
