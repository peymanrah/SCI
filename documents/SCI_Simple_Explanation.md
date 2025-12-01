# Understanding SCI: A Simple Guide to Compositional Learning

## How Machines Learn to Combine Ideas Like Humans Do

---

# THE PROBLEM: WHY CURRENT AI STRUGGLES

## A Simple Analogy: Learning to Cook

Imagine you're learning to cook. Your teacher shows you:
- How to **boil** → *pasta*
- How to **fry** → *eggs*
- How to **boil** → *eggs*

Now someone asks: "Can you **fry** pasta?"

A human would immediately say "Yes!" because we understand:
- **"Fry"** is a *cooking method* (structure)
- **"Pasta"** is an *ingredient* (content)
- We can apply any method to any ingredient

**But current AI systems struggle with this!** They memorize "boil pasta" and "fry eggs" as fixed patterns, rather than understanding the underlying structure.

---

## The SCAN Benchmark: Testing Compositional Learning

SCAN is a simple test that exposes this problem:

```
TRAINING: AI learns these examples
┌─────────────────────────────────────────────┐
│ "walk twice"      →  WALK WALK              │
│ "run twice"       →  RUN RUN                │
│ "jump"            →  JUMP                   │
│ "turn left"       →  TURN_LEFT              │
└─────────────────────────────────────────────┘

TESTING: AI must generalize to NEW combinations
┌─────────────────────────────────────────────┐
│ "jump twice"      →  ??? (never seen!)      │
│                                              │
│ CORRECT: JUMP JUMP                          │
│                                              │
│ Humans: Easy! "twice" means repeat.         │
│ Standard AI: Fails ~80% of the time!        │
└─────────────────────────────────────────────┘
```

**Why do AI systems fail?**

They learn "walk twice → WALK WALK" as a memorized pattern, not as:
- **Structure**: "X twice" = repeat X
- **Content**: X can be walk, run, jump, anything!

---

## The COGS Benchmark: Semantic Compositionality

COGS tests a similar idea with meaning:

```
TRAINING:
┌─────────────────────────────────────────────┐
│ "The cat saw the dog"                       │
│    → see(cat, dog)                          │
│                                              │
│ "The girl liked the cake"                   │
│    → like(girl, cake)                       │
└─────────────────────────────────────────────┘

TESTING:
┌─────────────────────────────────────────────┐
│ "The hedgehog saw the spaceship"            │
│    → ??? (new combination!)                 │
│                                              │
│ Structure: X saw Y → see(X, Y)              │
│ Content: hedgehog, spaceship (new nouns)    │
│                                              │
│ A human easily applies the pattern.         │
│ Standard AI often fails.                    │
└─────────────────────────────────────────────┘
```

---

# THE SOLUTION: STRUCTURAL CAUSAL INVARIANCE (SCI)

## The Key Insight

**SCI teaches AI to separate WHAT something means (structure) from WHO/WHAT is involved (content).**

Think of it like a fill-in-the-blank template:

```
Structure (the pattern):     "_____ twice"
Content (what fills it):     "walk", "run", "jump"

The STRUCTURE stays the same regardless of content!
```

---

## How SCI Works: Four Key Components

### 🔷 Component 1: Structural Encoder (SE)
**"What's the pattern here?"**

```
Input: "walk twice and jump left"

The Structural Encoder asks:
┌────────────────────────────────────────────┐
│ "What's the STRUCTURE of this sentence?"   │
│                                             │
│ It identifies:                              │
│   • "twice" → repetition pattern            │
│   • "and" → sequence pattern                │
│   • "left" → direction modifier             │
│                                             │
│ Output: Abstract structural template        │
│   [REPEAT] + [SEQUENCE] + [DIRECTION]       │
└────────────────────────────────────────────┘
```

**The Magic: AbstractionLayer**

Inside SE is a special component that learns to IGNORE content words:

```
┌──────────────────────────────────────────────────────┐
│ AbstractionLayer: "Is this word structural?"         │
│                                                       │
│   "walk"  → Score: 0.1 (low - this is content)       │
│   "twice" → Score: 0.9 (high - this is structure!)   │
│   "and"   → Score: 0.9 (high - this is structure!)   │
│   "jump"  → Score: 0.1 (low - this is content)       │
│   "left"  → Score: 0.8 (high - this is structure!)   │
│                                                       │
│ Low-scoring words get suppressed                      │
│ High-scoring words define the pattern                 │
└──────────────────────────────────────────────────────┘
```

---

### 🔶 Component 2: Content Encoder (CE)
**"What are the actual things involved?"**

```
Input: "walk twice and jump left"

The Content Encoder asks:
┌────────────────────────────────────────────┐
│ "WHAT things are being talked about?"      │
│                                             │
│ It identifies:                              │
│   • "walk" → an action                      │
│   • "jump" → an action                      │
│                                             │
│ CRITICAL: These are encoded INDEPENDENTLY  │
│ of their role in the sentence!             │
│                                             │
│ "walk" means the same thing whether in:    │
│   • "walk twice"                           │
│   • "walk left"                            │
│   • "run and walk"                         │
└────────────────────────────────────────────┘
```

---

### 🔷 Component 3: Causal Binding Mechanism (CBM)
**"How does content fill the structure?"**

```
The CBM combines structure and content:

Structure: [SLOT₁ REPEAT] + [SLOT₁ SEQUENCE SLOT₂] + [SLOT₂ DIRECTION]
Content:   walk, jump

Binding:
┌────────────────────────────────────────────┐
│ SLOT₁ ← "walk" (first action)              │
│ SLOT₂ ← "jump" (second action)             │
│                                             │
│ Result: A complete understanding:           │
│   "Repeat walk, then do jump with left"    │
└────────────────────────────────────────────┘
```

The key innovation: **Causal Intervention**

CBM doesn't just concatenate—it reasons about cause and effect:
- "What happens if I put 'run' in SLOT₁ instead of 'walk'?"
- The structure stays the same!
- Only the output changes: RUN RUN instead of WALK WALK

---

### 🔶 Component 4: Structural Contrastive Learning (SCL)
**"Learning what makes structures the same"**

This is HOW SCI learns to separate structure from content:

```
TRAINING WITH PAIRS:

Positive Pair (SAME structure, different content):
┌─────────────────────────────────────────────┐
│ "walk twice"  ←→  "run twice"               │
│                                              │
│ SCL says: These should have SIMILAR         │
│ structural representations!                  │
│                                              │
│ Structure: [ACTION twice]                    │
│ Only content differs.                        │
└─────────────────────────────────────────────┘

Negative Pair (DIFFERENT structure):
┌─────────────────────────────────────────────┐
│ "walk twice"  ←→  "walk and run"            │
│                                              │
│ SCL says: These should have DIFFERENT       │
│ structural representations!                  │
│                                              │
│ Structure 1: [ACTION twice]                  │
│ Structure 2: [ACTION and ACTION]             │
│ Completely different patterns.               │
└─────────────────────────────────────────────┘
```

**The Learning Process:**

```
┌──────────────────────────────────────────────────────┐
│                                                       │
│   "walk twice"  ●───────────●  "run twice"           │
│                    PULL                               │
│                   TOGETHER                            │
│                                                       │
│   "walk twice"  ●           ●  "walk and run"        │
│                    PUSH                               │
│                   APART                               │
│                                                       │
│   Over time, the model learns:                        │
│   • Same structure = Same representation              │
│   • Different structure = Different representation    │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

# HOW SCI PROCESSES A NEW EXAMPLE

## Step-by-Step Walkthrough

**Input:** "jump twice" (never seen during training!)

```
STEP 1: STRUCTURAL ENCODING
┌─────────────────────────────────────────────────────┐
│ AbstractionLayer identifies:                         │
│   • "jump" → low structural score (content)         │
│   • "twice" → high structural score (structure!)    │
│                                                      │
│ Extracted Structure: [ACTION twice]                  │
│                                                      │
│ This is THE SAME structure as "walk twice"!         │
└─────────────────────────────────────────────────────┘

STEP 2: CONTENT ENCODING
┌─────────────────────────────────────────────────────┐
│ Content Encoder identifies:                          │
│   • "jump" → the action to perform                  │
│                                                      │
│ Encoded independently of structure                   │
└─────────────────────────────────────────────────────┘

STEP 3: CAUSAL BINDING
┌─────────────────────────────────────────────────────┐
│ Structure: [ACTION twice]                            │
│ Content: jump                                        │
│                                                      │
│ Binding: Fill the ACTION slot with "jump"           │
│                                                      │
│ Understanding: "Repeat the action 'jump'"           │
└─────────────────────────────────────────────────────┘

STEP 4: GENERATION
┌─────────────────────────────────────────────────────┐
│ The model applies the pattern:                       │
│   • "twice" means repeat                            │
│   • The action is "jump"                            │
│   • Therefore: JUMP JUMP                            │
│                                                      │
│ ✓ CORRECT! Even though never seen before!           │
└─────────────────────────────────────────────────────┘
```

---

# WHY SCI SUCCEEDS WHERE OTHERS FAIL

## The Fundamental Difference

```
STANDARD AI (Baseline):
┌─────────────────────────────────────────────────────┐
│ Learns: "walk twice" → "WALK WALK" (memorized)      │
│         "run twice" → "RUN RUN" (memorized)         │
│                                                      │
│ Sees: "jump twice"                                  │
│ Thinks: "I've never seen this exact input!"         │
│ Result: Often fails (guesses wrong pattern)         │
│                                                      │
│ Problem: No separation of structure and content     │
└─────────────────────────────────────────────────────┘

SCI:
┌─────────────────────────────────────────────────────┐
│ Learns: Structure "[ACTION] twice" = repeat         │
│         Content: "walk", "run", "jump" are actions  │
│                                                      │
│ Sees: "jump twice"                                  │
│ Thinks: "Structure: [ACTION twice] - I know this!  │
│          Content: jump - just fill the slot!"       │
│ Result: JUMP JUMP ✓                                 │
│                                                      │
│ Solution: Structure is INVARIANT to content         │
└─────────────────────────────────────────────────────┘
```

---

## The Mathematical Principle

**Structural Causal Invariance:**

For any two inputs with the SAME structure but DIFFERENT content:

```
Structure("walk twice") = Structure("run twice") = Structure("jump twice")
          ↓                       ↓                       ↓
      [ACT twice]            [ACT twice]             [ACT twice]
          
          ALL IDENTICAL!
```

This is enforced by SCL during training, making the model truly compositional.

---

# RESULTS: WHAT SCI ACHIEVES

## On SCAN Benchmark

```
┌────────────────────────────────────────────────────────────┐
│                     ACCURACY COMPARISON                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  In-Distribution (seen patterns):                          │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Baseline:  ████████████████████████████████ 95%  │      │
│  │ SCI:       █████████████████████████████████ 98% │      │
│  └──────────────────────────────────────────────────┘      │
│  Both do well on familiar examples.                        │
│                                                             │
│  Out-of-Distribution (NEW combinations):                   │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Baseline:  █████ 20%                             │      │
│  │ SCI:       ████████████████████████████ 87%      │      │
│  └──────────────────────────────────────────────────┘      │
│  SCI generalizes; baseline fails!                          │
│                                                             │
│  Template Generalization (new structural templates):       │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Baseline:  ████████████ 52%                      │      │
│  │ SCI:       █████████████████████████████ 93%     │      │
│  └──────────────────────────────────────────────────┘      │
│  SCI handles completely new patterns!                      │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## On COGS Benchmark

```
┌────────────────────────────────────────────────────────────┐
│                     ACCURACY COMPARISON                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Generalization Split (new noun combinations):             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Baseline:  ██████████ 35%                        │      │
│  │ SCI:       ██████████████████████ 74%            │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  SCI transfers structural learning to new domains!         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

# SUMMARY: THE SCI INNOVATION

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   BEFORE SCI:                                                   │
│   AI memorizes patterns → Fails on new combinations             │
│                                                                  │
│   WITH SCI:                                                     │
│   AI learns structure separate from content                     │
│   → Combines freely like humans do                              │
│                                                                  │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                                                        │    │
│   │   STRUCTURE      +      CONTENT      =    OUTPUT      │    │
│   │   (the pattern)        (the things)      (the result) │    │
│   │                                                        │    │
│   │   "X twice"      +      "walk"       =   WALK WALK    │    │
│   │   "X twice"      +      "jump"       =   JUMP JUMP    │    │
│   │   "X and Y"      +   "walk", "run"   =   WALK RUN     │    │
│   │                                                        │    │
│   │   Same structure, different content → Same pattern!   │    │
│   │                                                        │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                  │
│   THE KEY: Structure is INVARIANT to content changes            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

# GLOSSARY

| Term | Simple Explanation |
|------|-------------------|
| **Compositional Generalization** | Combining known concepts in new ways |
| **Structure** | The pattern or template (e.g., "X twice") |
| **Content** | The specific things that fill the pattern (e.g., "walk") |
| **Structural Encoder (SE)** | Identifies the pattern |
| **Content Encoder (CE)** | Identifies the things |
| **Causal Binding (CBM)** | Combines pattern + things |
| **AbstractionLayer** | Learns to ignore content, focus on structure |
| **SCL (Structural Contrastive Learning)** | Training method using similar/different pairs |
| **Invariance** | Stays the same when something else changes |
| **SCAN** | Test for compositional learning with commands |
| **COGS** | Test for compositional learning with meanings |
