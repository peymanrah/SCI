# SCI Training Pairs: Strategic Generation Across Compositional Benchmarks

## Executive Summary

**The most critical factor for SCI success is accurate training pairs.**

This document provides:
1. Strategic training curriculum across benchmarks
2. Precise definitions of structure vs content for each benchmark
3. Rules for generating positive pairs (same structure, different content)
4. Rules for generating negative pairs (different structure)
5. Complete code implementations with validation
6. Quality assurance checks to prevent bad pairs

---

# PART 1: UNDERSTANDING STRUCTURAL INVARIANCE BY BENCHMARK

## 1.1 What Is "Structure" vs "Content"?

```
CORE PRINCIPLE:
- STRUCTURE = The compositional pattern/template that determines HOW elements combine
- CONTENT = The specific entities/values that fill the structural slots

POSITIVE PAIR: Same structure, different content
NEGATIVE PAIR: Different structure (content may or may not differ)

INVARIANCE GOAL: 
The Structural Encoder should produce IDENTICAL representations for positive pairs
and DIFFERENT representations for negative pairs.
```

## 1.2 Structure Definitions by Benchmark

### SCAN (Synthetic Compositional Tasks)

```
STRUCTURE in SCAN = The compositional grammar template

STRUCTURAL ELEMENTS (function words):
- "twice", "thrice" → Repetition modifiers
- "and" → Conjunction (combines two commands)
- "after" → Temporal sequencing
- "around", "opposite" → Directional modifiers
- "left", "right" → Directions (structural in context)

CONTENT ELEMENTS (action words):
- "walk", "run", "jump", "look", "turn" → Actions

TEMPLATE EXTRACTION:
"walk twice" → "[ACTION] twice"
"run and jump" → "[ACTION] and [ACTION]"
"jump around left thrice" → "[ACTION] around [DIRECTION] thrice"

POSITIVE PAIR EXAMPLE:
- "walk twice" ↔ "run twice" (same template: [ACTION] twice)

NEGATIVE PAIR EXAMPLE:
- "walk twice" ↔ "walk and run" (different templates)
```

### COGS (Compositional Generalization in Semantic Parsing)

```
STRUCTURE in COGS = Syntactic constituency structure

STRUCTURAL ELEMENTS:
- Sentence structure: S → NP VP
- Verb frames: transitive, ditransitive, with-PP, to-PP
- Embedding: relative clauses, that-clauses
- Coordination patterns

CONTENT ELEMENTS:
- Specific nouns: "Emma", "cake", "dog", "teacher"
- Specific verbs: "helped", "saw", "baked"
- Specific adjectives: "small", "red", "happy"

TEMPLATE EXTRACTION:
"Emma helped the dog" → "[SUBJ] [TRANS_VERB] [OBJ]"
"The cat that Emma saw ran" → "[NP [REL_CLAUSE]] [VERB]"

POSITIVE PAIR EXAMPLE:
- "Emma helped the dog" ↔ "Max saw the cake" 
  (same: [SUBJ] [TRANS_VERB] [OBJ])

NEGATIVE PAIR EXAMPLE:
- "Emma helped the dog" ↔ "Emma gave Max the cake"
  (different: transitive vs ditransitive)
```

### GSM8K (Grade School Math)

```
STRUCTURE in GSM8K = Mathematical operation sequence

STRUCTURAL ELEMENTS:
- Operation types: addition, subtraction, multiplication, division
- Operation order: sequential, nested
- Comparison patterns: "more than", "less than", "times as many"
- Variable relationships: part-whole, rate, proportion

CONTENT ELEMENTS:
- Specific numbers: 5, 12, 100
- Entity names: "apples", "dollars", "students"
- Person names: "John", "Mary"

TEMPLATE EXTRACTION:
"John has 5 apples. Mary has 3 more. How many total?"
→ "[ENTITY1] has [X]. [ENTITY2] has [Y] more. [SUM_QUERY]"

POSITIVE PAIR EXAMPLE:
- "John has 5 apples, Mary has 3 more, total?"
- "Tom has 12 books, Sue has 7 more, total?"
  (same: add-then-sum structure)

NEGATIVE PAIR EXAMPLE:
- "John has 5 apples, Mary has 3 more, total?"
- "John has 5 apples, gives 2 to Mary, remaining?"
  (different: addition vs subtraction)
```

### DROP (Discrete Reasoning Over Paragraphs)

```
STRUCTURE in DROP = Reasoning chain pattern

STRUCTURAL ELEMENTS:
- Reasoning types: counting, comparison, arithmetic, sorting
- Multi-hop patterns: bridge, comparison, intersection
- Temporal reasoning: before/after, duration
- Aggregation: min, max, count, sum

CONTENT ELEMENTS:
- Specific entities from passage
- Specific numbers and dates
- Specific events and facts

TEMPLATE EXTRACTION:
"How many more touchdowns than field goals?"
→ "[COUNT_COMPARE] [ENTITY1] [VERSUS] [ENTITY2]"

POSITIVE PAIR EXAMPLE:
- "How many more touchdowns than field goals?"
- "How many more wins than losses?"
  (same: count-difference structure)

NEGATIVE PAIR EXAMPLE:
- "How many more touchdowns than field goals?"
- "Who scored the most touchdowns?"
  (different: difference vs max-finding)
```

### LogiQA (Logical Reasoning)

```
STRUCTURE in LogiQA = Logical inference pattern

STRUCTURAL ELEMENTS:
- Logical connectives: if-then, and, or, not
- Quantifiers: all, some, none
- Inference types: modus ponens, modus tollens, disjunctive syllogism
- Argument structure: premise-premise-conclusion

CONTENT ELEMENTS:
- Specific propositions
- Specific entities in logical statements
- Domain-specific terms

TEMPLATE EXTRACTION:
"If A then B. A is true. Therefore?"
→ "[IF P THEN Q]. [P]. [CONCLUDE Q]" (modus ponens)

POSITIVE PAIR EXAMPLE:
- "If it rains, the ground is wet. It rained. Therefore?"
- "If John studies, he passes. John studied. Therefore?"
  (same: modus ponens)

NEGATIVE PAIR EXAMPLE:
- "If it rains, ground is wet. It rained. Therefore?"
- "Either A or B. Not A. Therefore?"
  (different: modus ponens vs disjunctive syllogism)
```

### StructTest (Multi-Domain Structured Output)

```
STRUCTURE in StructTest = Output format schema + instruction pattern

STRUCTURAL ELEMENTS:
- Output format: JSON schema, HTML structure, LaTeX template, code structure
- Instruction type: extract, generate, transform, validate
- Nested structures: hierarchy depth, field relationships

CONTENT ELEMENTS:
- Specific field values
- Specific text content
- Domain-specific data

TEMPLATE EXTRACTION:
"Extract name and age as JSON"
→ "[EXTRACT] [FIELDS:name,age] [FORMAT:JSON]"

POSITIVE PAIR EXAMPLE:
- "Extract name and age as JSON" ↔ "Extract title and date as JSON"
  (same: extract-two-fields-as-JSON)

NEGATIVE PAIR EXAMPLE:
- "Extract name and age as JSON" ↔ "Generate HTML table with name and age"
  (different: extract-JSON vs generate-HTML)
```

---

# PART 2: STRATEGIC TRAINING CURRICULUM

## 2.1 Recommended Training Order

```
CURRICULUM STRATEGY: Start simple, increase complexity gradually

PHASE 1: Foundation (SCAN Template Split)
├── Why: Shortest sequences, cleanest structural patterns
├── Duration: Until >90% in-distribution accuracy
├── Focus: Learn basic structure/content separation
└── Pairs: ~10,000 positive, ~10,000 negative

PHASE 2: Length Generalization (SCAN Length Split)  
├── Why: Same domain, tests length extrapolation
├── Duration: Until >75% length OOD accuracy
├── Focus: Length-invariant structural representations
└── Pairs: Augment with length-diverse pairs

PHASE 3: Semantic Transfer (COGS)
├── Why: Natural language, richer syntax
├── Duration: Until >60% generalization split
├── Focus: Transfer structural learning to real syntax
└── Pairs: ~20,000 syntactic structure pairs

PHASE 4: Reasoning Patterns (GSM8K + DROP)
├── Why: Tests compositional reasoning, not just syntax
├── Duration: Until reasoning accuracy improves
├── Focus: Mathematical and multi-hop structure
└── Pairs: ~15,000 reasoning structure pairs each

PHASE 5: Multi-Domain (StructTest + LogiQA)
├── Why: Tests cross-domain structural transfer
├── Duration: Fine-tuning for specific domains
├── Focus: Format-invariant and logic-invariant representations
└── Pairs: ~10,000 per domain
```

## 2.2 Training Configuration by Phase

```yaml
# configs/curriculum_training.yaml

curriculum:
  phase1_scan_template:
    dataset: "scan"
    split: "template"  # addprim_jump or addprim_turn_left
    epochs: 10
    scl_weight: 0.3
    target_accuracy: 0.90
    stop_condition: "val_exact_match >= 0.90 or epochs >= 10"
    
  phase2_scan_length:
    dataset: "scan"
    split: "length"
    epochs: 20
    scl_weight: 0.35
    target_accuracy: 0.75
    # Load checkpoint from phase1
    resume_from: "phase1_best"
    # Reduce base LR since already pretrained
    base_lr: 1e-5
    sci_lr: 3e-5
    
  phase3_cogs:
    dataset: "cogs"
    split: "gen"
    epochs: 15
    scl_weight: 0.3
    target_accuracy: 0.60
    resume_from: "phase2_best"
    base_lr: 5e-6
    sci_lr: 2e-5
    
  phase4_gsm8k:
    dataset: "gsm8k"
    split: "test"
    epochs: 10
    scl_weight: 0.25
    target_accuracy: 0.50
    resume_from: "phase3_best"
    
  phase5_multi:
    datasets: ["structtest", "logiqa", "drop"]
    epochs: 5  # per dataset
    scl_weight: 0.2
    resume_from: "phase4_best"
```

---

# PART 3: PAIR GENERATION CODE

## 3.1 Base Pair Generator Class

```python
# sci/data/pair_generators/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import random
import hashlib

@dataclass
class SCLPair:
    """A single SCL training pair."""
    anchor: str
    partner: str  # positive or negative
    is_positive: bool
    anchor_template: str
    partner_template: str
    metadata: Dict = None
    
    def validate(self) -> bool:
        """Validate pair correctness."""
        if self.is_positive:
            return self.anchor_template == self.partner_template
        else:
            return self.anchor_template != self.partner_template


@dataclass 
class SCLBatch:
    """A batch of SCL pairs for training."""
    anchors: List[str]
    positives: List[str]
    negatives: List[List[str]]  # Multiple negatives per anchor
    anchor_templates: List[str]
    
    def __len__(self):
        return len(self.anchors)


class BasePairGenerator(ABC):
    """Base class for SCL pair generation."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self._template_cache: Dict[str, str] = {}
        self._examples_by_template: Dict[str, List[str]] = {}
        
    @abstractmethod
    def extract_template(self, text: str) -> str:
        """Extract structural template from text."""
        pass
    
    @abstractmethod
    def get_structure_elements(self) -> Set[str]:
        """Return set of structural elements for this benchmark."""
        pass
    
    @abstractmethod
    def get_content_elements(self) -> Set[str]:
        """Return set of content elements for this benchmark."""
        pass
    
    def build_template_index(self, examples: List[str]):
        """Build index of examples by template for efficient pair generation."""
        self._examples_by_template.clear()
        self._template_cache.clear()
        
        for example in examples:
            template = self.extract_template(example)
            self._template_cache[example] = template
            
            if template not in self._examples_by_template:
                self._examples_by_template[template] = []
            self._examples_by_template[template].append(example)
        
        # Filter templates with at least 2 examples (needed for positive pairs)
        self._examples_by_template = {
            k: v for k, v in self._examples_by_template.items() 
            if len(v) >= 2
        }
        
        print(f"Built index: {len(self._examples_by_template)} templates, "
              f"{len(examples)} examples")
    
    def generate_positive_pair(self) -> Optional[SCLPair]:
        """Generate a positive pair (same template, different content)."""
        if not self._examples_by_template:
            raise ValueError("Call build_template_index first")
        
        # Random template with at least 2 examples
        template = self.rng.choice(list(self._examples_by_template.keys()))
        examples = self._examples_by_template[template]
        
        if len(examples) < 2:
            return None
        
        anchor, partner = self.rng.sample(examples, 2)
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=template,
            partner_template=template
        )
    
    def generate_negative_pair(self) -> Optional[SCLPair]:
        """Generate a negative pair (different templates)."""
        if len(self._examples_by_template) < 2:
            raise ValueError("Need at least 2 different templates")
        
        # Random two different templates
        templates = self.rng.sample(list(self._examples_by_template.keys()), 2)
        
        anchor = self.rng.choice(self._examples_by_template[templates[0]])
        partner = self.rng.choice(self._examples_by_template[templates[1]])
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=False,
            anchor_template=templates[0],
            partner_template=templates[1]
        )
    
    def generate_batch(
        self, 
        batch_size: int = 32, 
        num_negatives: int = 5
    ) -> SCLBatch:
        """Generate a batch for SCL training."""
        anchors = []
        positives = []
        negatives = []
        anchor_templates = []
        
        for _ in range(batch_size):
            # Generate positive pair
            pos_pair = self.generate_positive_pair()
            if pos_pair is None:
                continue
            
            anchors.append(pos_pair.anchor)
            positives.append(pos_pair.partner)
            anchor_templates.append(pos_pair.anchor_template)
            
            # Generate negatives
            batch_negatives = []
            attempts = 0
            while len(batch_negatives) < num_negatives and attempts < num_negatives * 3:
                neg_pair = self.generate_negative_pair()
                if neg_pair and neg_pair.partner_template != pos_pair.anchor_template:
                    batch_negatives.append(neg_pair.partner)
                attempts += 1
            
            # Pad if needed
            while len(batch_negatives) < num_negatives:
                batch_negatives.append(self.rng.choice(
                    list(self._template_cache.keys())
                ))
            
            negatives.append(batch_negatives)
        
        return SCLBatch(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            anchor_templates=anchor_templates
        )
    
    def validate_pairs(self, pairs: List[SCLPair]) -> Dict:
        """Validate a list of pairs and return statistics."""
        stats = {
            'total': len(pairs),
            'valid': 0,
            'invalid': 0,
            'positive_correct': 0,
            'negative_correct': 0,
            'errors': []
        }
        
        for i, pair in enumerate(pairs):
            if pair.validate():
                stats['valid'] += 1
                if pair.is_positive:
                    stats['positive_correct'] += 1
                else:
                    stats['negative_correct'] += 1
            else:
                stats['invalid'] += 1
                stats['errors'].append({
                    'index': i,
                    'anchor': pair.anchor,
                    'partner': pair.partner,
                    'is_positive': pair.is_positive,
                    'anchor_template': pair.anchor_template,
                    'partner_template': pair.partner_template
                })
        
        return stats
```

## 3.2 SCAN Pair Generator

```python
# sci/data/pair_generators/scan.py

import re
from typing import Set, Dict, List, Optional
from .base import BasePairGenerator, SCLPair

class SCANPairGenerator(BasePairGenerator):
    """
    Pair generator for SCAN benchmark.
    
    SCAN Structure:
    - Actions: walk, run, jump, look, turn (CONTENT)
    - Directions: left, right (STRUCTURAL in modifiers)
    - Modifiers: twice, thrice, around, opposite (STRUCTURAL)
    - Conjunctions: and, after (STRUCTURAL)
    
    Template format: [ACTION] becomes placeholder, structural words preserved.
    """
    
    # Structural elements (preserved in template)
    STRUCTURE_WORDS = {
        'twice', 'thrice',  # Repetition
        'and', 'after',     # Conjunction
        'around', 'opposite',  # Direction modifiers
        'left', 'right'     # Directions (structural when with modifiers)
    }
    
    # Content elements (replaced with [ACTION])
    CONTENT_WORDS = {
        'walk', 'run', 'jump', 'look', 'turn'
    }
    
    def get_structure_elements(self) -> Set[str]:
        return self.STRUCTURE_WORDS
    
    def get_content_elements(self) -> Set[str]:
        return self.CONTENT_WORDS
    
    def extract_template(self, text: str) -> str:
        """
        Extract structural template from SCAN command.
        
        Examples:
        - "walk twice" → "[ACTION] twice"
        - "run and jump" → "[ACTION] and [ACTION]"
        - "jump around left thrice" → "[ACTION] around left thrice"
        - "walk opposite right twice and run left" 
          → "[ACTION] opposite right twice and [ACTION] left"
        """
        # Check cache first
        if text in self._template_cache:
            return self._template_cache[text]
        
        # Normalize
        text = text.lower().strip()
        words = text.split()
        
        template_words = []
        for word in words:
            if word in self.CONTENT_WORDS:
                template_words.append('[ACTION]')
            else:
                template_words.append(word)
        
        template = ' '.join(template_words)
        self._template_cache[text] = template
        
        return template
    
    def generate_from_template(
        self, 
        template: str, 
        exclude: Optional[Set[str]] = None
    ) -> str:
        """Generate a new example from a template."""
        exclude = exclude or set()
        
        words = template.split()
        result = []
        
        available_actions = list(self.CONTENT_WORDS - exclude)
        if not available_actions:
            available_actions = list(self.CONTENT_WORDS)
        
        for word in words:
            if word == '[ACTION]':
                action = self.rng.choice(available_actions)
                result.append(action)
                # Remove used action to encourage diversity
                if action in available_actions and len(available_actions) > 1:
                    available_actions.remove(action)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def generate_positive_pair_synthetic(self) -> SCLPair:
        """
        Generate positive pair by creating from same template.
        More reliable than sampling from corpus.
        """
        # Define common SCAN templates
        templates = [
            "[ACTION] twice",
            "[ACTION] thrice", 
            "[ACTION] and [ACTION]",
            "[ACTION] after [ACTION]",
            "[ACTION] around left",
            "[ACTION] around right",
            "[ACTION] opposite left",
            "[ACTION] opposite right",
            "[ACTION] around left twice",
            "[ACTION] around right twice",
            "[ACTION] opposite left twice",
            "[ACTION] opposite right twice",
            "[ACTION] and [ACTION] twice",
            "[ACTION] after [ACTION] twice",
            "[ACTION] around left and [ACTION]",
            "[ACTION] around right and [ACTION]",
            "[ACTION] twice and [ACTION] thrice",
            "[ACTION] opposite left twice and [ACTION] around right",
        ]
        
        template = self.rng.choice(templates)
        
        # Generate two different instances
        anchor = self.generate_from_template(template)
        
        # Ensure different content for partner
        anchor_actions = set(w for w in anchor.split() if w in self.CONTENT_WORDS)
        partner = self.generate_from_template(template, exclude=anchor_actions)
        
        # Verify they're different
        if anchor == partner:
            partner = self.generate_from_template(template)
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=template,
            partner_template=template
        )
    
    def generate_negative_pair_synthetic(self) -> SCLPair:
        """Generate negative pair with guaranteed different templates."""
        templates = [
            "[ACTION] twice",
            "[ACTION] thrice",
            "[ACTION] and [ACTION]",
            "[ACTION] after [ACTION]",
            "[ACTION] around left",
            "[ACTION] around right",
            "[ACTION] opposite left twice",
            "[ACTION] and [ACTION] twice",
            "[ACTION] around left thrice and [ACTION]",
        ]
        
        # Select two different templates
        template1, template2 = self.rng.sample(templates, 2)
        
        anchor = self.generate_from_template(template1)
        partner = self.generate_from_template(template2)
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=False,
            anchor_template=template1,
            partner_template=template2
        )


class SCANLengthPairGenerator(SCANPairGenerator):
    """
    Specialized generator for SCAN Length split.
    
    Key challenge: Generate pairs that test length invariance.
    Short and long sequences with same structure should be positive pairs.
    """
    
    # Templates of varying length complexity
    SHORT_TEMPLATES = [
        "[ACTION] twice",
        "[ACTION] and [ACTION]",
        "[ACTION] left",
    ]
    
    MEDIUM_TEMPLATES = [
        "[ACTION] around left twice",
        "[ACTION] and [ACTION] twice",
        "[ACTION] opposite right thrice",
        "[ACTION] after [ACTION] after [ACTION]",
    ]
    
    LONG_TEMPLATES = [
        "[ACTION] around left twice and [ACTION] around right twice",
        "[ACTION] opposite left thrice and [ACTION] opposite right thrice",
        "[ACTION] around left twice and [ACTION] around right twice and [ACTION]",
        "[ACTION] opposite left twice and [ACTION] around right thrice and [ACTION] twice",
    ]
    
    def generate_length_invariant_pair(self) -> SCLPair:
        """
        Generate pair where structure pattern is same despite length difference.
        
        Example: Both use "X around [DIR] twice" pattern
        - Short: "walk around left twice"
        - Long: "walk around left twice and run around right twice"
        
        Both have the SAME structural pattern (around-dir-twice), just repeated.
        """
        # Base structural unit
        base_patterns = [
            "[ACTION] around {dir} twice",
            "[ACTION] opposite {dir} twice",
            "[ACTION] around {dir} thrice",
            "[ACTION] {dir} twice",
        ]
        
        pattern = self.rng.choice(base_patterns)
        dir1 = self.rng.choice(['left', 'right'])
        dir2 = self.rng.choice(['left', 'right'])
        
        # Short version (single pattern)
        short_template = pattern.format(dir=dir1)
        
        # Long version (pattern repeated with 'and')
        long_template = f"{pattern.format(dir=dir1)} and {pattern.format(dir=dir2)}"
        
        # Note: These are DIFFERENT templates (one has 'and', one doesn't)
        # For TRUE positive pairs with length variation, we need same template
        
        # TRUE length-invariant positive pair:
        # Same template, but we'll use the data's natural variation
        template = self.rng.choice(self.MEDIUM_TEMPLATES + self.LONG_TEMPLATES)
        
        anchor = self.generate_from_template(template)
        partner = self.generate_from_template(template)
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=template,
            partner_template=template
        )
```

## 3.3 COGS Pair Generator

```python
# sci/data/pair_generators/cogs.py

import re
from typing import Set, Dict, List, Tuple, Optional
from .base import BasePairGenerator, SCLPair

class COGSPairGenerator(BasePairGenerator):
    """
    Pair generator for COGS benchmark.
    
    COGS Structure is based on syntactic patterns:
    - Verb transitivity: transitive, intransitive, ditransitive
    - Argument structure: NP, PP, clausal complements
    - Modifiers: relative clauses, PP attachments
    - Embeddings: that-clauses, infinitives
    
    Template format: Syntactic structure with slots for specific lexical items.
    """
    
    # Syntactic structure markers
    STRUCTURE_PATTERNS = {
        # Verb frames
        'transitive': r'\[SUBJ\] \[V_TRANS\] \[OBJ\]',
        'intransitive': r'\[SUBJ\] \[V_INTRANS\]',
        'ditransitive': r'\[SUBJ\] \[V_DITRANS\] \[OBJ1\] \[OBJ2\]',
        'pp_dative': r'\[SUBJ\] \[V_TRANS\] \[OBJ\] to \[RECIP\]',
        
        # Modifiers
        'rel_subj': r'\[NP\] that \[V\] \[REST\]',  # "the dog that ran"
        'rel_obj': r'\[NP\] that \[SUBJ\] \[V\]',   # "the dog that Emma saw"
        'pp_mod': r'\[NP\] \[PP\]',                  # "the dog on the mat"
    }
    
    # Content words (to be replaced in templates)
    NOUNS = {
        'emma', 'liam', 'olivia', 'noah', 'ava', 'william',
        'dog', 'cat', 'cake', 'ball', 'book', 'table',
        'teacher', 'student', 'doctor', 'artist'
    }
    
    VERBS_TRANS = {'helped', 'saw', 'found', 'wanted', 'liked', 'knew'}
    VERBS_INTRANS = {'ran', 'slept', 'smiled', 'cried', 'arrived'}
    VERBS_DITRANS = {'gave', 'sent', 'showed', 'told', 'offered'}
    
    ADJECTIVES = {'small', 'big', 'red', 'blue', 'happy', 'old', 'new'}
    PREPOSITIONS = {'on', 'in', 'beside', 'behind', 'under', 'near'}
    
    def get_structure_elements(self) -> Set[str]:
        return {'that', 'to', 'a', 'the', 'who', 'which'} | set(self.PREPOSITIONS)
    
    def get_content_elements(self) -> Set[str]:
        return self.NOUNS | self.VERBS_TRANS | self.VERBS_INTRANS | self.VERBS_DITRANS | self.ADJECTIVES
    
    def _parse_sentence_structure(self, sentence: str) -> Dict:
        """Parse COGS sentence into structural components."""
        sentence = sentence.lower().strip()
        words = sentence.split()
        
        structure = {
            'pattern': None,
            'has_relative': False,
            'has_pp': False,
            'verb_frame': None,
            'embedding_depth': 0
        }
        
        # Check for relative clauses
        if ' that ' in sentence or ' who ' in sentence:
            structure['has_relative'] = True
            structure['embedding_depth'] += 1
        
        # Check for PP
        for prep in self.PREPOSITIONS:
            if f' {prep} ' in sentence:
                structure['has_pp'] = True
                break
        
        # Determine verb frame
        for verb in self.VERBS_DITRANS:
            if verb in words:
                structure['verb_frame'] = 'ditransitive'
                break
        if structure['verb_frame'] is None:
            for verb in self.VERBS_TRANS:
                if verb in words:
                    structure['verb_frame'] = 'transitive'
                    break
        if structure['verb_frame'] is None:
            for verb in self.VERBS_INTRANS:
                if verb in words:
                    structure['verb_frame'] = 'intransitive'
                    break
        
        # Create pattern signature
        structure['pattern'] = (
            f"verb:{structure['verb_frame']}"
            f"_rel:{structure['has_relative']}"
            f"_pp:{structure['has_pp']}"
            f"_depth:{structure['embedding_depth']}"
        )
        
        return structure
    
    def extract_template(self, text: str) -> str:
        """
        Extract structural template from COGS sentence.
        
        Strategy: Replace content words with typed placeholders.
        """
        text = text.lower().strip()
        
        # Get structural analysis
        structure = self._parse_sentence_structure(text)
        
        # Build template by replacing content
        template = text
        
        # Replace proper nouns with [NAME]
        for noun in ['emma', 'liam', 'olivia', 'noah', 'ava', 'william']:
            template = re.sub(rf'\b{noun}\b', '[NAME]', template)
        
        # Replace common nouns with [NOUN]
        for noun in ['dog', 'cat', 'cake', 'ball', 'book', 'table', 
                     'teacher', 'student', 'doctor', 'artist']:
            template = re.sub(rf'\b{noun}\b', '[NOUN]', template)
        
        # Replace verbs
        for verb in self.VERBS_TRANS:
            template = re.sub(rf'\b{verb}\b', '[V_TRANS]', template)
        for verb in self.VERBS_INTRANS:
            template = re.sub(rf'\b{verb}\b', '[V_INTRANS]', template)
        for verb in self.VERBS_DITRANS:
            template = re.sub(rf'\b{verb}\b', '[V_DITRANS]', template)
        
        # Replace adjectives with [ADJ]
        for adj in self.ADJECTIVES:
            template = re.sub(rf'\b{adj}\b', '[ADJ]', template)
        
        return template
    
    def generate_from_template(
        self, 
        template: str,
        exclude: Optional[Set[str]] = None
    ) -> str:
        """Generate new sentence from template."""
        exclude = exclude or set()
        
        result = template
        
        # Replace [NAME] with random proper noun
        while '[NAME]' in result:
            name = self.rng.choice(list({'emma', 'liam', 'olivia', 'noah'} - exclude))
            result = result.replace('[NAME]', name, 1)
        
        # Replace [NOUN] with random common noun
        while '[NOUN]' in result:
            noun = self.rng.choice(list({'dog', 'cat', 'cake', 'ball', 'book'} - exclude))
            result = result.replace('[NOUN]', noun, 1)
        
        # Replace verbs
        while '[V_TRANS]' in result:
            verb = self.rng.choice(list(self.VERBS_TRANS))
            result = result.replace('[V_TRANS]', verb, 1)
        while '[V_INTRANS]' in result:
            verb = self.rng.choice(list(self.VERBS_INTRANS))
            result = result.replace('[V_INTRANS]', verb, 1)
        while '[V_DITRANS]' in result:
            verb = self.rng.choice(list(self.VERBS_DITRANS))
            result = result.replace('[V_DITRANS]', verb, 1)
        
        # Replace [ADJ]
        while '[ADJ]' in result:
            adj = self.rng.choice(list(self.ADJECTIVES))
            result = result.replace('[ADJ]', adj, 1)
        
        return result
    
    def generate_positive_pair_by_syntax(self) -> SCLPair:
        """Generate positive pair with same syntactic structure."""
        
        # Define syntactic templates
        templates = {
            'simple_trans': "[NAME] [V_TRANS] the [NOUN]",
            'simple_intrans': "the [NOUN] [V_INTRANS]",
            'ditrans': "[NAME] [V_DITRANS] [NAME] the [NOUN]",
            'rel_subj': "the [NOUN] that [V_INTRANS] [V_TRANS] the [NOUN]",
            'rel_obj': "the [NOUN] that [NAME] [V_TRANS] [V_INTRANS]",
            'pp_attach': "[NAME] [V_TRANS] the [NOUN] on the [NOUN]",
        }
        
        template_name = self.rng.choice(list(templates.keys()))
        template = templates[template_name]
        
        anchor = self.generate_from_template(template)
        partner = self.generate_from_template(template)
        
        # Ensure different
        attempts = 0
        while anchor == partner and attempts < 10:
            partner = self.generate_from_template(template)
            attempts += 1
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=template_name,
            partner_template=template_name
        )
```

## 3.4 GSM8K Pair Generator

```python
# sci/data/pair_generators/gsm8k.py

import re
from typing import Set, Dict, List, Optional
from .base import BasePairGenerator, SCLPair

class GSM8KPairGenerator(BasePairGenerator):
    """
    Pair generator for GSM8K math word problems.
    
    GSM8K Structure is based on mathematical operation patterns:
    - Operation types: +, -, *, /
    - Operation sequences: single, chain, nested
    - Problem patterns: comparison, rate, ratio, total
    - Variable relationships: part-whole, before-after, rate-quantity
    """
    
    # Mathematical structure patterns
    MATH_STRUCTURES = {
        'add_total': "Add quantities to find total",
        'sub_remain': "Subtract to find remainder", 
        'mult_rate': "Multiply rate by quantity",
        'div_share': "Divide total into equal parts",
        'compare_diff': "Find difference between two quantities",
        'compare_ratio': "Find ratio between quantities",
        'chain_add_sub': "Add then subtract",
        'chain_mult_add': "Multiply then add",
        'multi_step': "Multiple operations in sequence",
    }
    
    # Content words to replace
    ENTITIES = {
        'apples', 'oranges', 'books', 'pencils', 'dollars', 'cookies',
        'students', 'teachers', 'cars', 'bikes', 'pages', 'hours'
    }
    
    NAMES = {
        'john', 'mary', 'tom', 'jane', 'bob', 'alice', 'sam', 'emma'
    }
    
    def get_structure_elements(self) -> Set[str]:
        return {
            'more', 'less', 'than', 'times', 'total', 'each', 'every',
            'together', 'combined', 'remaining', 'left', 'gave', 'received',
            'how', 'many', 'much', 'what', 'find', 'calculate'
        }
    
    def get_content_elements(self) -> Set[str]:
        return self.ENTITIES | self.NAMES
    
    def _classify_problem_structure(self, problem: str) -> str:
        """Classify the mathematical structure of a problem."""
        problem_lower = problem.lower()
        
        # Detect operation patterns
        has_addition = any(w in problem_lower for w in 
                         ['total', 'together', 'combined', 'sum', 'plus', 'more than'])
        has_subtraction = any(w in problem_lower for w in 
                            ['remaining', 'left', 'minus', 'less than', 'gave away', 'spent'])
        has_multiplication = any(w in problem_lower for w in 
                               ['times', 'each', 'every', 'per', 'multiply'])
        has_division = any(w in problem_lower for w in 
                         ['divide', 'split', 'shared equally', 'each got'])
        has_comparison = any(w in problem_lower for w in 
                           ['more than', 'less than', 'difference', 'compare'])
        
        # Classify
        if has_comparison and has_subtraction:
            return 'compare_diff'
        elif has_multiplication and has_addition:
            return 'chain_mult_add'
        elif has_addition and has_subtraction:
            return 'chain_add_sub'
        elif has_multiplication:
            return 'mult_rate'
        elif has_division:
            return 'div_share'
        elif has_addition:
            return 'add_total'
        elif has_subtraction:
            return 'sub_remain'
        else:
            return 'multi_step'
    
    def extract_template(self, text: str) -> str:
        """Extract mathematical structure template from problem."""
        # Get structure classification
        structure = self._classify_problem_structure(text)
        
        # Also create abstracted template
        template = text.lower()
        
        # Replace numbers with [NUM]
        template = re.sub(r'\b\d+\b', '[NUM]', template)
        
        # Replace names with [NAME]
        for name in self.NAMES:
            template = re.sub(rf'\b{name}\b', '[NAME]', template)
        
        # Replace entities with [ENTITY]
        for entity in self.ENTITIES:
            template = re.sub(rf'\b{entity}\b', '[ENTITY]', template)
        
        # Return combined structure signature
        return f"{structure}::{template[:100]}"  # Truncate template
    
    def generate_positive_pair_by_math_structure(self) -> SCLPair:
        """Generate problems with same mathematical structure."""
        
        # Template library by structure
        templates = {
            'add_total': [
                "[NAME] has [NUM] [ENTITY]. [NAME] has [NUM] [ENTITY]. How many [ENTITY] do they have together?",
                "[NAME] bought [NUM] [ENTITY] and [NUM] [ENTITY]. What is the total number of [ENTITY]?",
            ],
            'sub_remain': [
                "[NAME] had [NUM] [ENTITY]. [NAME] gave [NUM] to [NAME]. How many [ENTITY] are left?",
                "[NAME] started with [NUM] [ENTITY] and used [NUM]. How many [ENTITY] remain?",
            ],
            'mult_rate': [
                "[NAME] reads [NUM] [ENTITY] every day. How many [ENTITY] in [NUM] days?",
                "Each [ENTITY] costs [NUM] [ENTITY]. How much for [NUM] [ENTITY]?",
            ],
            'compare_diff': [
                "[NAME] has [NUM] [ENTITY]. [NAME] has [NUM] more. How many more does [NAME] have?",
                "[NAME] scored [NUM] points. [NAME] scored [NUM] fewer. What is the difference?",
            ],
            'chain_mult_add': [
                "[NAME] bought [NUM] [ENTITY] at [NUM] each, plus [NUM] tax. Total cost?",
                "[NAME] earns [NUM] per hour for [NUM] hours, plus [NUM] bonus. Total earnings?",
            ],
        }
        
        # Select structure
        structure = self.rng.choice(list(templates.keys()))
        template_options = templates[structure]
        
        # Select same template or different template with same structure
        if len(template_options) >= 2:
            t1, t2 = self.rng.sample(template_options, 2)
        else:
            t1 = t2 = template_options[0]
        
        # Generate instances
        anchor = self._fill_math_template(t1)
        partner = self._fill_math_template(t2)
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=structure,
            partner_template=structure
        )
    
    def _fill_math_template(self, template: str) -> str:
        """Fill a math problem template with random values."""
        result = template
        
        # Fill names
        names = list(self.NAMES)
        self.rng.shuffle(names)
        name_idx = 0
        while '[NAME]' in result:
            result = result.replace('[NAME]', names[name_idx % len(names)].capitalize(), 1)
            name_idx += 1
        
        # Fill numbers (reasonable ranges)
        while '[NUM]' in result:
            num = self.rng.randint(2, 50)
            result = result.replace('[NUM]', str(num), 1)
        
        # Fill entities
        entities = list(self.ENTITIES)
        self.rng.shuffle(entities)
        while '[ENTITY]' in result:
            result = result.replace('[ENTITY]', entities[0], 1)
        
        return result
    
    def generate_negative_pair_by_math_structure(self) -> SCLPair:
        """Generate problems with different mathematical structures."""
        structures = list(self.MATH_STRUCTURES.keys())
        s1, s2 = self.rng.sample(structures[:5], 2)  # Use first 5 distinct structures
        
        # Simple templates per structure for negatives
        simple_templates = {
            'add_total': "[NAME] has [NUM] [ENTITY]. [NAME] has [NUM] [ENTITY]. Total?",
            'sub_remain': "[NAME] had [NUM] [ENTITY], gave away [NUM]. Remaining?",
            'mult_rate': "[NAME] reads [NUM] pages per day for [NUM] days. Total pages?",
            'div_share': "[NAME] splits [NUM] [ENTITY] among [NUM] friends. Each gets?",
            'compare_diff': "[NAME] has [NUM], [NAME] has [NUM]. Difference?",
        }
        
        anchor = self._fill_math_template(simple_templates.get(s1, simple_templates['add_total']))
        partner = self._fill_math_template(simple_templates.get(s2, simple_templates['sub_remain']))
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=False,
            anchor_template=s1,
            partner_template=s2
        )
```

## 3.5 DROP Pair Generator

```python
# sci/data/pair_generators/drop.py

from typing import Set, Dict, List, Optional
from .base import BasePairGenerator, SCLPair

class DROPPairGenerator(BasePairGenerator):
    """
    Pair generator for DROP reading comprehension.
    
    DROP Structure is based on reasoning patterns:
    - Counting: "How many..."
    - Arithmetic: "How many more/fewer...", "Total of..."
    - Sorting: "First/Last...", "Most/Least..."
    - Temporal: "Before/After...", "During..."
    - Multi-hop: Combining facts from passage
    """
    
    REASONING_PATTERNS = {
        'count': "Count instances of entity",
        'count_diff': "Count difference between two categories",
        'arithmetic_add': "Add quantities from passage",
        'arithmetic_sub': "Subtract quantities",
        'max_min': "Find maximum or minimum",
        'temporal_before': "Identify what happened before",
        'temporal_after': "Identify what happened after",
        'multi_hop': "Combine multiple facts",
    }
    
    QUESTION_STRUCTURES = {
        'count': [
            "How many [ENTITY] [VERB]?",
            "How many [ENTITY] were there?",
            "What is the number of [ENTITY]?",
        ],
        'count_diff': [
            "How many more [ENTITY1] than [ENTITY2]?",
            "How many fewer [ENTITY1] than [ENTITY2]?",
            "What is the difference between [ENTITY1] and [ENTITY2]?",
        ],
        'max_min': [
            "Which [ENTITY] had the most [PROPERTY]?",
            "Who scored the highest?",
            "What was the longest [ENTITY]?",
        ],
        'temporal': [
            "What happened before [EVENT]?",
            "What happened after [EVENT]?",
            "Which [ENTITY] came first?",
        ],
    }
    
    def get_structure_elements(self) -> Set[str]:
        return {
            'how', 'many', 'more', 'fewer', 'than', 'most', 'least',
            'first', 'last', 'before', 'after', 'during', 'between',
            'total', 'difference', 'which', 'what', 'who'
        }
    
    def get_content_elements(self) -> Set[str]:
        return {
            'touchdown', 'field goal', 'yard', 'point', 'quarter',
            'team', 'player', 'game', 'score', 'win', 'loss'
        }
    
    def _classify_question(self, question: str) -> str:
        """Classify question by reasoning type."""
        q_lower = question.lower()
        
        if 'how many more' in q_lower or 'how many fewer' in q_lower:
            return 'count_diff'
        elif 'difference' in q_lower:
            return 'count_diff'
        elif 'how many' in q_lower:
            return 'count'
        elif any(w in q_lower for w in ['most', 'highest', 'longest', 'largest']):
            return 'max_min'
        elif any(w in q_lower for w in ['least', 'lowest', 'shortest', 'smallest']):
            return 'max_min'
        elif any(w in q_lower for w in ['before', 'after', 'first', 'last']):
            return 'temporal'
        else:
            return 'multi_hop'
    
    def extract_template(self, text: str) -> str:
        """Extract reasoning structure from DROP question."""
        structure = self._classify_question(text)
        return structure
    
    def generate_positive_pair_by_reasoning(self) -> SCLPair:
        """Generate questions with same reasoning structure."""
        structure = self.rng.choice(['count', 'count_diff', 'max_min', 'temporal'])
        
        templates = self.QUESTION_STRUCTURES[structure]
        t1 = self.rng.choice(templates)
        t2 = self.rng.choice(templates)
        
        anchor = self._fill_drop_template(t1)
        partner = self._fill_drop_template(t2)
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=structure,
            partner_template=structure
        )
    
    def _fill_drop_template(self, template: str) -> str:
        """Fill DROP question template."""
        result = template
        
        entities = ['touchdowns', 'field goals', 'yards', 'points', 'wins', 'losses']
        
        while '[ENTITY]' in result:
            result = result.replace('[ENTITY]', self.rng.choice(entities), 1)
        while '[ENTITY1]' in result:
            result = result.replace('[ENTITY1]', self.rng.choice(entities), 1)
        while '[ENTITY2]' in result:
            result = result.replace('[ENTITY2]', self.rng.choice(entities), 1)
        while '[PROPERTY]' in result:
            result = result.replace('[PROPERTY]', self.rng.choice(['points', 'yards', 'scores']), 1)
        while '[EVENT]' in result:
            result = result.replace('[EVENT]', self.rng.choice(['halftime', 'the final quarter', 'overtime']), 1)
        while '[VERB]' in result:
            result = result.replace('[VERB]', self.rng.choice(['were scored', 'happened', 'occurred']), 1)
        
        return result
```

## 3.6 LogiQA Pair Generator

```python
# sci/data/pair_generators/logiqa.py

from typing import Set, Dict, List
from .base import BasePairGenerator, SCLPair

class LogiQAPairGenerator(BasePairGenerator):
    """
    Pair generator for LogiQA logical reasoning.
    
    LogiQA Structure is based on logical inference patterns:
    - Modus ponens: If P then Q. P. Therefore Q.
    - Modus tollens: If P then Q. Not Q. Therefore not P.
    - Disjunctive syllogism: P or Q. Not P. Therefore Q.
    - Hypothetical syllogism: If P then Q. If Q then R. Therefore if P then R.
    - Categorical syllogism: All A are B. X is A. Therefore X is B.
    """
    
    LOGIC_PATTERNS = {
        'modus_ponens': "If [P] then [Q]. [P]. Therefore [Q].",
        'modus_tollens': "If [P] then [Q]. Not [Q]. Therefore not [P].",
        'disjunctive': "[P] or [Q]. Not [P]. Therefore [Q].",
        'hypothetical': "If [P] then [Q]. If [Q] then [R]. Therefore if [P] then [R].",
        'categorical_all': "All [A] are [B]. [X] is [A]. Therefore [X] is [B].",
        'categorical_some': "Some [A] are [B]. [X] is [A]. Therefore [X] might be [B].",
    }
    
    # Sample propositions
    PROPOSITIONS = {
        'rain': ('it rains', 'the ground is wet'),
        'study': ('John studies', 'John passes'),
        'exercise': ('you exercise', 'you are healthy'),
        'practice': ('you practice', 'you improve'),
        'invest': ('you invest wisely', 'you gain returns'),
    }
    
    CATEGORIES = {
        'mammals': ('mammals', 'warm-blooded'),
        'birds': ('birds', 'have feathers'),
        'scientists': ('scientists', 'curious'),
        'athletes': ('athletes', 'physically fit'),
    }
    
    def get_structure_elements(self) -> Set[str]:
        return {
            'if', 'then', 'therefore', 'not', 'or', 'and', 'all', 'some',
            'none', 'must', 'might', 'cannot', 'because', 'since'
        }
    
    def get_content_elements(self) -> Set[str]:
        elements = set()
        for p, q in self.PROPOSITIONS.values():
            elements.add(p)
            elements.add(q)
        for a, b in self.CATEGORIES.values():
            elements.add(a)
            elements.add(b)
        return elements
    
    def extract_template(self, text: str) -> str:
        """Extract logical structure from LogiQA problem."""
        text_lower = text.lower()
        
        # Detect pattern
        if 'if' in text_lower and 'then' in text_lower:
            if 'not' in text_lower and text_lower.index('not') > text_lower.index('then'):
                return 'modus_tollens'
            return 'modus_ponens'
        elif ' or ' in text_lower and 'not' in text_lower:
            return 'disjunctive'
        elif 'all ' in text_lower:
            return 'categorical_all'
        elif 'some ' in text_lower:
            return 'categorical_some'
        else:
            return 'unknown'
    
    def generate_positive_pair_by_logic(self) -> SCLPair:
        """Generate problems with same logical structure."""
        
        patterns_with_props = ['modus_ponens', 'modus_tollens', 'disjunctive']
        patterns_with_cats = ['categorical_all', 'categorical_some']
        
        if self.rng.random() < 0.6:
            # Propositional logic
            pattern = self.rng.choice(patterns_with_props)
            prop1 = self.rng.choice(list(self.PROPOSITIONS.keys()))
            prop2 = self.rng.choice(list(self.PROPOSITIONS.keys()))
            
            anchor = self._fill_propositional(pattern, prop1)
            partner = self._fill_propositional(pattern, prop2)
        else:
            # Categorical logic
            pattern = self.rng.choice(patterns_with_cats)
            cat1 = self.rng.choice(list(self.CATEGORIES.keys()))
            cat2 = self.rng.choice(list(self.CATEGORIES.keys()))
            
            anchor = self._fill_categorical(pattern, cat1)
            partner = self._fill_categorical(pattern, cat2)
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=pattern,
            partner_template=pattern
        )
    
    def _fill_propositional(self, pattern: str, prop_key: str) -> str:
        """Fill propositional logic template."""
        p, q = self.PROPOSITIONS[prop_key]
        
        if pattern == 'modus_ponens':
            return f"If {p} then {q}. {p.capitalize()}. Therefore, {q}."
        elif pattern == 'modus_tollens':
            return f"If {p} then {q}. {q.capitalize()} is not the case. Therefore, {p} is not the case."
        elif pattern == 'disjunctive':
            return f"Either {p} or {q}. {p.capitalize()} is not the case. Therefore, {q}."
        return ""
    
    def _fill_categorical(self, pattern: str, cat_key: str) -> str:
        """Fill categorical logic template."""
        a, b = self.CATEGORIES[cat_key]
        x = self.rng.choice(['Socrates', 'John', 'The object', 'This entity'])
        
        if pattern == 'categorical_all':
            return f"All {a} are {b}. {x} is a {a[:-1] if a.endswith('s') else a}. Therefore, {x} is {b}."
        elif pattern == 'categorical_some':
            return f"Some {a} are {b}. {x} is a {a[:-1] if a.endswith('s') else a}. Therefore, {x} might be {b}."
        return ""
```

## 3.7 StructTest Pair Generator

```python
# sci/data/pair_generators/structtest.py

from typing import Set, Dict, List
from .base import BasePairGenerator, SCLPair

class StructTestPairGenerator(BasePairGenerator):
    """
    Pair generator for StructTest multi-domain structured output.
    
    StructTest Structure is based on:
    - Output format: JSON, HTML, LaTeX, code
    - Operation type: extract, generate, transform
    - Schema structure: flat, nested, array
    """
    
    OUTPUT_FORMATS = ['json', 'html', 'latex', 'python', 'sql']
    OPERATIONS = ['extract', 'generate', 'transform', 'validate']
    SCHEMA_TYPES = ['flat', 'nested', 'array', 'mixed']
    
    INSTRUCTION_TEMPLATES = {
        ('json', 'extract', 'flat'): [
            "Extract the [FIELD1] and [FIELD2] as JSON",
            "Parse and return [FIELD1], [FIELD2] in JSON format",
        ],
        ('json', 'extract', 'nested'): [
            "Extract [FIELD1] with nested [FIELD2] details as JSON",
            "Return JSON with [FIELD1] containing [FIELD2] object",
        ],
        ('html', 'generate', 'flat'): [
            "Generate an HTML table with [FIELD1] and [FIELD2] columns",
            "Create HTML list showing [FIELD1] and [FIELD2]",
        ],
        ('html', 'generate', 'nested'): [
            "Generate nested HTML divs with [FIELD1] containing [FIELD2]",
            "Create HTML structure with [FIELD1] parent and [FIELD2] children",
        ],
        ('python', 'generate', 'flat'): [
            "Write a Python function that takes [FIELD1] and returns [FIELD2]",
            "Generate Python code to process [FIELD1] into [FIELD2]",
        ],
        ('latex', 'generate', 'flat'): [
            "Format [FIELD1] and [FIELD2] as a LaTeX table",
            "Create LaTeX equation with [FIELD1] and [FIELD2]",
        ],
    }
    
    FIELDS = [
        'name', 'age', 'title', 'date', 'price', 'quantity',
        'description', 'category', 'status', 'id', 'value', 'type'
    ]
    
    def get_structure_elements(self) -> Set[str]:
        return set(self.OUTPUT_FORMATS) | set(self.OPERATIONS) | {
            'as', 'with', 'into', 'from', 'to', 'format',
            'extract', 'generate', 'parse', 'create', 'return'
        }
    
    def get_content_elements(self) -> Set[str]:
        return set(self.FIELDS)
    
    def extract_template(self, text: str) -> str:
        """Extract structure from StructTest instruction."""
        text_lower = text.lower()
        
        # Detect format
        fmt = 'unknown'
        for f in self.OUTPUT_FORMATS:
            if f in text_lower:
                fmt = f
                break
        
        # Detect operation
        op = 'unknown'
        for o in self.OPERATIONS:
            if o in text_lower:
                op = o
                break
        
        # Detect schema type
        schema = 'flat'
        if any(w in text_lower for w in ['nested', 'containing', 'with']):
            schema = 'nested'
        elif any(w in text_lower for w in ['array', 'list', 'multiple']):
            schema = 'array'
        
        return f"{fmt}_{op}_{schema}"
    
    def generate_positive_pair_by_structure(self) -> SCLPair:
        """Generate instructions with same structural pattern."""
        
        # Select structure
        fmt = self.rng.choice(self.OUTPUT_FORMATS[:4])  # json, html, latex, python
        op = self.rng.choice(['extract', 'generate'])
        schema = self.rng.choice(['flat', 'nested'])
        
        structure = (fmt, op, schema)
        
        if structure in self.INSTRUCTION_TEMPLATES:
            templates = self.INSTRUCTION_TEMPLATES[structure]
        else:
            templates = [f"{op.capitalize()} data as {fmt} in {schema} format"]
        
        t1 = self.rng.choice(templates)
        t2 = self.rng.choice(templates)
        
        # Fill with different fields
        fields1 = self.rng.sample(self.FIELDS, 2)
        fields2 = self.rng.sample(self.FIELDS, 2)
        
        anchor = t1.replace('[FIELD1]', fields1[0]).replace('[FIELD2]', fields1[1])
        partner = t2.replace('[FIELD1]', fields2[0]).replace('[FIELD2]', fields2[1])
        
        template_key = f"{fmt}_{op}_{schema}"
        
        return SCLPair(
            anchor=anchor,
            partner=partner,
            is_positive=True,
            anchor_template=template_key,
            partner_template=template_key
        )
```

---

# PART 4: UNIFIED PAIR GENERATION SYSTEM

## 4.1 Master Generator

```python
# sci/data/pair_generators/master.py

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import json
import os

from .scan import SCANPairGenerator, SCANLengthPairGenerator
from .cogs import COGSPairGenerator
from .gsm8k import GSM8KPairGenerator
from .drop import DROPPairGenerator
from .logiqa import LogiQAPairGenerator
from .structtest import StructTestPairGenerator
from .base import SCLPair, SCLBatch, BasePairGenerator

@dataclass
class PairGenerationConfig:
    """Configuration for pair generation."""
    benchmark: str
    num_positive_pairs: int = 10000
    num_negative_pairs: int = 10000
    num_negatives_per_anchor: int = 5
    seed: int = 42
    validation_ratio: float = 0.1
    output_dir: str = "data/scl_pairs"


class MasterPairGenerator:
    """
    Unified system for generating SCL pairs across all benchmarks.
    
    Usage:
        generator = MasterPairGenerator()
        pairs = generator.generate_for_benchmark("scan_length", config)
        generator.save_pairs(pairs, "scan_length_pairs.json")
    """
    
    GENERATORS = {
        'scan_simple': SCANPairGenerator,
        'scan_template': SCANPairGenerator,
        'scan_length': SCANLengthPairGenerator,
        'cogs': COGSPairGenerator,
        'gsm8k': GSM8KPairGenerator,
        'drop': DROPPairGenerator,
        'logiqa': LogiQAPairGenerator,
        'structtest': StructTestPairGenerator,
    }
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._generators: Dict[str, BasePairGenerator] = {}
    
    def get_generator(self, benchmark: str) -> BasePairGenerator:
        """Get or create generator for benchmark."""
        if benchmark not in self._generators:
            if benchmark not in self.GENERATORS:
                raise ValueError(f"Unknown benchmark: {benchmark}. "
                               f"Available: {list(self.GENERATORS.keys())}")
            
            self._generators[benchmark] = self.GENERATORS[benchmark](seed=self.seed)
        
        return self._generators[benchmark]
    
    def generate_for_benchmark(
        self, 
        benchmark: str,
        config: Optional[PairGenerationConfig] = None,
        corpus: Optional[List[str]] = None
    ) -> Dict[str, List[SCLPair]]:
        """
        Generate SCL pairs for a specific benchmark.
        
        Args:
            benchmark: Name of benchmark
            config: Generation configuration
            corpus: Optional corpus to index for pair generation
            
        Returns:
            Dict with 'positive' and 'negative' pair lists
        """
        config = config or PairGenerationConfig(benchmark=benchmark)
        generator = self.get_generator(benchmark)
        
        # Build index if corpus provided
        if corpus:
            generator.build_template_index(corpus)
        
        # Generate pairs
        positive_pairs = []
        negative_pairs = []
        
        print(f"Generating {config.num_positive_pairs} positive pairs for {benchmark}...")
        for _ in range(config.num_positive_pairs):
            # Use synthetic generation methods
            if hasattr(generator, 'generate_positive_pair_synthetic'):
                pair = generator.generate_positive_pair_synthetic()
            elif hasattr(generator, 'generate_positive_pair_by_syntax'):
                pair = generator.generate_positive_pair_by_syntax()
            elif hasattr(generator, 'generate_positive_pair_by_math_structure'):
                pair = generator.generate_positive_pair_by_math_structure()
            elif hasattr(generator, 'generate_positive_pair_by_reasoning'):
                pair = generator.generate_positive_pair_by_reasoning()
            elif hasattr(generator, 'generate_positive_pair_by_logic'):
                pair = generator.generate_positive_pair_by_logic()
            elif hasattr(generator, 'generate_positive_pair_by_structure'):
                pair = generator.generate_positive_pair_by_structure()
            else:
                pair = generator.generate_positive_pair()
            
            if pair:
                positive_pairs.append(pair)
        
        print(f"Generating {config.num_negative_pairs} negative pairs for {benchmark}...")
        for _ in range(config.num_negative_pairs):
            if hasattr(generator, 'generate_negative_pair_synthetic'):
                pair = generator.generate_negative_pair_synthetic()
            elif hasattr(generator, 'generate_negative_pair_by_math_structure'):
                pair = generator.generate_negative_pair_by_math_structure()
            else:
                pair = generator.generate_negative_pair()
            
            if pair:
                negative_pairs.append(pair)
        
        return {
            'positive': positive_pairs,
            'negative': negative_pairs
        }
    
    def validate_pairs(
        self, 
        pairs: Dict[str, List[SCLPair]],
        benchmark: str
    ) -> Dict:
        """Validate generated pairs."""
        generator = self.get_generator(benchmark)
        
        all_pairs = pairs['positive'] + pairs['negative']
        stats = generator.validate_pairs(all_pairs)
        
        # Add benchmark-specific validation
        stats['benchmark'] = benchmark
        stats['positive_count'] = len(pairs['positive'])
        stats['negative_count'] = len(pairs['negative'])
        
        # Check template diversity
        positive_templates = set(p.anchor_template for p in pairs['positive'])
        negative_templates = set(p.anchor_template for p in pairs['negative'])
        
        stats['unique_positive_templates'] = len(positive_templates)
        stats['unique_negative_templates'] = len(negative_templates)
        
        return stats
    
    def save_pairs(
        self, 
        pairs: Dict[str, List[SCLPair]], 
        output_path: str
    ):
        """Save pairs to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'positive': [
                {
                    'anchor': p.anchor,
                    'partner': p.partner,
                    'anchor_template': p.anchor_template,
                    'partner_template': p.partner_template
                }
                for p in pairs['positive']
            ],
            'negative': [
                {
                    'anchor': p.anchor,
                    'partner': p.partner,
                    'anchor_template': p.anchor_template,
                    'partner_template': p.partner_template
                }
                for p in pairs['negative']
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(pairs['positive'])} positive and "
              f"{len(pairs['negative'])} negative pairs to {output_path}")
    
    def load_pairs(self, input_path: str) -> Dict[str, List[SCLPair]]:
        """Load pairs from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        pairs = {
            'positive': [
                SCLPair(
                    anchor=p['anchor'],
                    partner=p['partner'],
                    is_positive=True,
                    anchor_template=p['anchor_template'],
                    partner_template=p['partner_template']
                )
                for p in data['positive']
            ],
            'negative': [
                SCLPair(
                    anchor=p['anchor'],
                    partner=p['partner'],
                    is_positive=False,
                    anchor_template=p['anchor_template'],
                    partner_template=p['partner_template']
                )
                for p in data['negative']
            ]
        }
        
        return pairs


def generate_all_benchmark_pairs(output_dir: str = "data/scl_pairs"):
    """Generate pairs for all benchmarks."""
    generator = MasterPairGenerator(seed=42)
    
    benchmarks = [
        'scan_simple',
        'scan_template', 
        'scan_length',
        'cogs',
        'gsm8k',
        'drop',
        'logiqa',
        'structtest'
    ]
    
    all_stats = {}
    
    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Generating pairs for: {benchmark}")
        print('='*60)
        
        config = PairGenerationConfig(
            benchmark=benchmark,
            num_positive_pairs=10000,
            num_negative_pairs=10000
        )
        
        pairs = generator.generate_for_benchmark(benchmark, config)
        
        # Validate
        stats = generator.validate_pairs(pairs, benchmark)
        all_stats[benchmark] = stats
        
        print(f"\nValidation results for {benchmark}:")
        print(f"  - Total pairs: {stats['total']}")
        print(f"  - Valid: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
        print(f"  - Unique positive templates: {stats['unique_positive_templates']}")
        print(f"  - Unique negative templates: {stats['unique_negative_templates']}")
        
        if stats['invalid'] > 0:
            print(f"  - ⚠️ Invalid pairs: {stats['invalid']}")
            for error in stats['errors'][:3]:
                print(f"    - {error}")
        
        # Save
        output_path = os.path.join(output_dir, f"{benchmark}_pairs.json")
        generator.save_pairs(pairs, output_path)
    
    # Save summary
    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Generation complete! Summary saved to:", summary_path)
    print('='*60)
    
    return all_stats


if __name__ == "__main__":
    generate_all_benchmark_pairs()
```

---

# PART 5: QUALITY ASSURANCE

## 5.1 Pair Quality Validation

```python
# sci/data/pair_validators.py

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from .pair_generators.base import SCLPair

@dataclass
class ValidationResult:
    """Result of pair validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict


class PairQualityValidator:
    """
    Comprehensive validator for SCL pairs.
    
    Checks:
    1. Template correctness (positive pairs have same template)
    2. Content diversity (pairs have different content)
    3. Template diversity (training set has diverse templates)
    4. No trivial pairs (anchor != partner)
    5. No duplicate pairs
    """
    
    def validate_batch(
        self, 
        pairs: List[SCLPair],
        expected_positive_ratio: float = 0.5
    ) -> ValidationResult:
        """Validate a batch of pairs."""
        errors = []
        warnings = []
        
        # 1. Template correctness
        for i, pair in enumerate(pairs):
            if not pair.validate():
                errors.append(f"Pair {i}: Template mismatch - "
                            f"positive={pair.is_positive}, "
                            f"templates={pair.anchor_template} vs {pair.partner_template}")
        
        # 2. No trivial pairs
        trivial_count = 0
        for i, pair in enumerate(pairs):
            if pair.anchor == pair.partner:
                warnings.append(f"Pair {i}: Trivial pair (anchor == partner)")
                trivial_count += 1
        
        # 3. Duplicate check
        seen = set()
        duplicates = 0
        for pair in pairs:
            key = (pair.anchor, pair.partner, pair.is_positive)
            if key in seen:
                duplicates += 1
            seen.add(key)
        
        if duplicates > len(pairs) * 0.01:
            warnings.append(f"High duplicate rate: {duplicates} ({duplicates/len(pairs)*100:.1f}%)")
        
        # 4. Template diversity
        positive_pairs = [p for p in pairs if p.is_positive]
        negative_pairs = [p for p in pairs if not p.is_positive]
        
        positive_templates = set(p.anchor_template for p in positive_pairs)
        negative_templates = set(p.anchor_template for p in negative_pairs)
        
        if len(positive_templates) < 5:
            warnings.append(f"Low positive template diversity: {len(positive_templates)}")
        if len(negative_templates) < 5:
            warnings.append(f"Low negative template diversity: {len(negative_templates)}")
        
        # 5. Ratio check
        actual_positive_ratio = len(positive_pairs) / len(pairs) if pairs else 0
        if abs(actual_positive_ratio - expected_positive_ratio) > 0.1:
            warnings.append(f"Unexpected positive ratio: {actual_positive_ratio:.2f} "
                          f"(expected {expected_positive_ratio:.2f})")
        
        # Compile stats
        stats = {
            'total_pairs': len(pairs),
            'positive_pairs': len(positive_pairs),
            'negative_pairs': len(negative_pairs),
            'unique_positive_templates': len(positive_templates),
            'unique_negative_templates': len(negative_templates),
            'trivial_pairs': trivial_count,
            'duplicates': duplicates,
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def validate_template_extraction(
        self,
        examples: List[Tuple[str, str]],  # (input, expected_template)
        generator
    ) -> ValidationResult:
        """Validate template extraction accuracy."""
        errors = []
        correct = 0
        
        for input_text, expected in examples:
            actual = generator.extract_template(input_text)
            if actual == expected:
                correct += 1
            else:
                errors.append(f"Template mismatch: '{input_text}' -> "
                            f"'{actual}' (expected '{expected}')")
        
        stats = {
            'total': len(examples),
            'correct': correct,
            'accuracy': correct / len(examples) if examples else 0
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[],
            stats=stats
        )


# Test template extraction for each benchmark
TEMPLATE_TEST_CASES = {
    'scan': [
        ("walk twice", "[ACTION] twice"),
        ("run and jump", "[ACTION] and [ACTION]"),
        ("jump around left thrice", "[ACTION] around left thrice"),
        ("walk opposite right twice and run", "[ACTION] opposite right twice and [ACTION]"),
    ],
    'cogs': [
        ("emma helped the dog", "[NAME] [V_TRANS] the [NOUN]"),
        ("the cat ran", "the [NOUN] [V_INTRANS]"),
    ],
    'gsm8k': [
        # These use structure classification rather than exact templates
    ],
}
```

## 5.2 Running Validation

```python
# scripts/validate_pairs.py

import argparse
import json
from sci.data.pair_generators.master import MasterPairGenerator
from sci.data.pair_validators import PairQualityValidator, TEMPLATE_TEST_CASES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--pairs_file', required=True)
    args = parser.parse_args()
    
    # Load pairs
    generator = MasterPairGenerator()
    pairs = generator.load_pairs(args.pairs_file)
    
    # Validate
    validator = PairQualityValidator()
    
    all_pairs = pairs['positive'] + pairs['negative']
    result = validator.validate_batch(all_pairs)
    
    print(f"\n{'='*60}")
    print(f"Validation Results for {args.benchmark}")
    print('='*60)
    
    print(f"\nValid: {'✅ YES' if result.is_valid else '❌ NO'}")
    
    print(f"\nStatistics:")
    for key, value in result.stats.items():
        print(f"  - {key}: {value}")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:10]:
            print(f"  ❌ {error}")
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings[:10]:
            print(f"  ⚠️ {warning}")
    
    # Test template extraction
    if args.benchmark in TEMPLATE_TEST_CASES:
        print(f"\nTemplate Extraction Test:")
        gen = generator.get_generator(args.benchmark)
        template_result = validator.validate_template_extraction(
            TEMPLATE_TEST_CASES[args.benchmark],
            gen
        )
        print(f"  Accuracy: {template_result.stats['accuracy']*100:.1f}%")
        if template_result.errors:
            for error in template_result.errors[:5]:
                print(f"  ❌ {error}")


if __name__ == "__main__":
    main()
```

---

# PART 6: TRAINING INTEGRATION

## 6.1 SCL Dataset Class

```python
# sci/data/scl_dataset.py

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import json
import random

class SCLDataset(Dataset):
    """
    Dataset for SCL training that provides (anchor, positive, negatives) tuples.
    """
    
    def __init__(
        self,
        pairs_file: str,
        tokenizer,
        max_length: int = 128,
        num_negatives: int = 5,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)
        
        # Load pairs
        with open(pairs_file, 'r') as f:
            data = json.load(f)
        
        self.positive_pairs = data['positive']
        self.negative_examples = [p['partner'] for p in data['negative']]
        
        # Build template index for hard negative mining
        self._build_template_index(data)
    
    def _build_template_index(self, data):
        """Index examples by template for hard negative mining."""
        self.by_template = {}
        
        for p in data['positive']:
            template = p['anchor_template']
            if template not in self.by_template:
                self.by_template[template] = []
            self.by_template[template].append(p['anchor'])
            self.by_template[template].append(p['partner'])
    
    def __len__(self):
        return len(self.positive_pairs)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a training item."""
        pos_pair = self.positive_pairs[idx]
        
        anchor = pos_pair['anchor']
        positive = pos_pair['partner']
        anchor_template = pos_pair['anchor_template']
        
        # Sample negatives (from different templates)
        negatives = []
        other_templates = [t for t in self.by_template.keys() if t != anchor_template]
        
        for _ in range(self.num_negatives):
            if other_templates:
                neg_template = self.rng.choice(other_templates)
                neg = self.rng.choice(self.by_template[neg_template])
            else:
                neg = self.rng.choice(self.negative_examples)
            negatives.append(neg)
        
        # Tokenize
        anchor_enc = self.tokenizer(
            anchor,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive_enc = self.tokenizer(
            positive,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        negative_encs = [
            self.tokenizer(
                neg,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            for neg in negatives
        ]
        
        return {
            'anchor_input_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_enc['attention_mask'].squeeze(0),
            'positive_input_ids': positive_enc['input_ids'].squeeze(0),
            'positive_attention_mask': positive_enc['attention_mask'].squeeze(0),
            'negative_input_ids': torch.stack([e['input_ids'].squeeze(0) for e in negative_encs]),
            'negative_attention_mask': torch.stack([e['attention_mask'].squeeze(0) for e in negative_encs]),
        }


def collate_scl_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate SCL batch."""
    return {
        'anchor_input_ids': torch.stack([b['anchor_input_ids'] for b in batch]),
        'anchor_attention_mask': torch.stack([b['anchor_attention_mask'] for b in batch]),
        'positive_input_ids': torch.stack([b['positive_input_ids'] for b in batch]),
        'positive_attention_mask': torch.stack([b['positive_attention_mask'] for b in batch]),
        'negative_input_ids': torch.stack([b['negative_input_ids'] for b in batch]),
        'negative_attention_mask': torch.stack([b['negative_attention_mask'] for b in batch]),
    }
```

---

# SUMMARY: KEY TAKEAWAYS

## Critical Rules for SCL Pair Generation

1. **STRUCTURE vs CONTENT must be clearly defined per benchmark**
   - SCAN: Function words (twice, and, around) are structure; action words are content
   - COGS: Syntactic patterns are structure; lexical items are content
   - GSM8K: Operation sequences are structure; numbers/entities are content

2. **POSITIVE pairs MUST have identical templates**
   - Same structural pattern, different surface content
   - Validate with template extraction

3. **NEGATIVE pairs MUST have different templates**
   - Different structural pattern
   - Content can be same or different (doesn't matter)

4. **Template extraction must be precise**
   - Test with known examples before generating
   - Log template diversity statistics

5. **Training curriculum matters**
   - Start with SCAN Template (simpler)
   - Progress to SCAN Length (harder)
   - Transfer to COGS, GSM8K, etc.

6. **Always validate before training**
   - Run validation script on generated pairs
   - Check template diversity
   - Check for trivial/duplicate pairs

---

**END OF DOCUMENT**
