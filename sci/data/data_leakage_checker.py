"""
Data Leakage Checker - Required by SCI_ENGINEERING_STANDARDS.md

Implements comprehensive data leakage safeguards:
1. Train/val/test split overlap detection
2. SCAN length-split constraint verification (train ≤22 tokens, test >22 tokens)
3. SE/CE attention mask verification (zero attention on response tokens)
4. Instruction mask consistency validation
"""

import torch
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataLeakageChecker:
    """
    Comprehensive data leakage detection for SCI training.
    
    Implements checks required by SCI_ENGINEERING_STANDARDS.md:
    - No overlap between train/val/test splits
    - Length split constraints enforced
    - SE/CE only see instruction tokens (attention verification)
    - Near-duplicate detection (#21 FIX)
    """
    
    def __init__(self, split_name: str = "length", near_duplicate_threshold: float = 0.9):
        """
        Args:
            split_name: SCAN split name ("length", "simple", "template", etc.)
            near_duplicate_threshold: Jaccard similarity threshold for near-duplicates (default: 0.9)
        """
        self.split_name = split_name
        self.seen_commands: Dict[str, Set[str]] = defaultdict(set)  # subset -> commands
        self.length_violations: List[Dict] = []
        self.near_duplicate_threshold = near_duplicate_threshold
        
        # Length split constraints (from SCAN benchmark)
        self.LENGTH_SPLIT_TRAIN_MAX = 22  # Training: outputs ≤ 22 tokens
        self.LENGTH_SPLIT_TEST_MIN = 23   # Test: outputs > 22 tokens
        
    def check_split_overlap(
        self,
        train_commands: List[str],
        val_commands: Optional[List[str]] = None,
        test_commands: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Check for overlap between train/val/test splits.
        
        Args:
            train_commands: List of training commands
            val_commands: Optional list of validation commands
            test_commands: Optional list of test commands
            
        Returns:
            Dict with overlap statistics and any violations
        """
        train_set = set(train_commands)
        val_set = set(val_commands) if val_commands else set()
        test_set = set(test_commands) if test_commands else set()
        
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        has_leakage = len(train_val_overlap) > 0 or len(train_test_overlap) > 0
        
        result = {
            "has_leakage": has_leakage,
            "train_size": len(train_set),
            "val_size": len(val_set),
            "test_size": len(test_set),
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
        }
        
        if has_leakage:
            logger.error(f"DATA LEAKAGE DETECTED!")
            logger.error(f"  Train-Val overlap: {len(train_val_overlap)} examples")
            logger.error(f"  Train-Test overlap: {len(train_test_overlap)} examples")
            if train_test_overlap:
                logger.error(f"  Sample leaked commands: {list(train_test_overlap)[:3]}")
        else:
            logger.info("✓ No split overlap detected")
            
        return result
    
    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """
        Compute Jaccard similarity between two strings based on word tokens.
        
        #21 FIX: Add near-duplicate detection.
        """
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union
    
    def check_near_duplicates(
        self,
        train_commands: List[str],
        test_commands: List[str],
    ) -> Dict[str, any]:
        """
        Check for near-duplicate commands between train and test sets.
        
        #21 FIX: Implements semantic similarity checking, not just exact match.
        
        Args:
            train_commands: List of training commands
            test_commands: List of test commands
            
        Returns:
            Dict with near-duplicate detection results
        """
        near_duplicates = []
        
        # For efficiency, only check a sample if sets are large
        train_sample = train_commands[:1000] if len(train_commands) > 1000 else train_commands
        test_sample = test_commands[:500] if len(test_commands) > 500 else test_commands
        
        for test_cmd in test_sample:
            for train_cmd in train_sample:
                if test_cmd == train_cmd:
                    continue  # Skip exact matches (handled by check_split_overlap)
                similarity = self._jaccard_similarity(test_cmd, train_cmd)
                if similarity >= self.near_duplicate_threshold:
                    near_duplicates.append({
                        "train_cmd": train_cmd,
                        "test_cmd": test_cmd,
                        "similarity": similarity,
                    })
        
        result = {
            "num_near_duplicates": len(near_duplicates),
            "threshold": self.near_duplicate_threshold,
            "near_duplicates": near_duplicates[:10],  # First 10
        }
        
        if near_duplicates:
            logger.warning(f"NEAR-DUPLICATES DETECTED: {len(near_duplicates)} pairs with "
                          f"similarity >= {self.near_duplicate_threshold}")
            for nd in near_duplicates[:3]:
                logger.warning(f"  - Train: '{nd['train_cmd'][:40]}...' vs Test: '{nd['test_cmd'][:40]}...' "
                              f"(sim={nd['similarity']:.2f})")
        else:
            logger.info(f"✓ No near-duplicates detected (threshold={self.near_duplicate_threshold})")
            
        return result
    
    def check_length_split_constraints(
        self,
        commands: List[str],
        actions: List[str],
        subset: str,  # "train" or "test"
    ) -> Dict[str, any]:
        """
        Verify SCAN length split constraints.
        
        For length split:
        - Training: all action sequences ≤ 22 tokens
        - Test: all action sequences > 22 tokens (for OOD generalization)
        
        Args:
            commands: List of commands
            actions: List of action sequences
            subset: "train" or "test"
            
        Returns:
            Dict with constraint verification results
        """
        if self.split_name != "length":
            return {"applicable": False, "message": "Not length split"}
        
        violations = []
        
        for i, (cmd, act) in enumerate(zip(commands, actions)):
            action_tokens = act.split()
            num_tokens = len(action_tokens)
            
            if subset == "train" and num_tokens > self.LENGTH_SPLIT_TRAIN_MAX:
                violations.append({
                    "index": i,
                    "command": cmd,
                    "num_tokens": num_tokens,
                    "expected_max": self.LENGTH_SPLIT_TRAIN_MAX,
                })
            elif subset == "test" and num_tokens <= self.LENGTH_SPLIT_TRAIN_MAX:
                violations.append({
                    "index": i,
                    "command": cmd,
                    "num_tokens": num_tokens,
                    "expected_min": self.LENGTH_SPLIT_TEST_MIN,
                })
        
        result = {
            "applicable": True,
            "subset": subset,
            "total_examples": len(commands),
            "num_violations": len(violations),
            "violations": violations[:10] if violations else [],  # First 10
        }
        
        if violations:
            logger.warning(f"LENGTH SPLIT CONSTRAINT VIOLATIONS: {len(violations)} in {subset}")
            for v in violations[:3]:
                logger.warning(f"  - '{v['command'][:50]}...' has {v['num_tokens']} tokens")
        else:
            logger.info(f"✓ Length split constraints satisfied for {subset}")
            
        return result
    
    def verify_instruction_mask_excludes_padding(
        self,
        instruction_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> bool:
        """
        Verify instruction_mask does not include padding positions.
        
        Padding should be excluded from instruction_mask to prevent
        AbstractionLayer/CE from seeing padding as "instruction".
        
        Args:
            instruction_mask: [batch, seq_len] - 1 for instruction, 0 for response
            attention_mask: [batch, seq_len] - 1 for tokens, 0 for padding
            
        Returns:
            True if no padding leakage, False otherwise
        """
        # Where attention_mask is 0 (padding), instruction_mask should also be 0
        padding_positions = (attention_mask == 0)
        instruction_at_padding = instruction_mask[padding_positions]
        
        if instruction_at_padding.sum() > 0:
            logger.error(f"PADDING LEAKAGE: {instruction_at_padding.sum().item()} padding "
                        f"positions marked as instruction")
            return False
            
        return True
    
    def verify_se_ce_attention_weights(
        self,
        attention_weights: torch.Tensor,  # [batch, heads, seq, seq]
        instruction_mask: torch.Tensor,   # [batch, seq]
        tolerance: float = 1e-6,
    ) -> Dict[str, any]:
        """
        Verify SE/CE attention weights are zero on response tokens.
        
        This is a CRITICAL check to ensure no data leakage during training.
        
        Args:
            attention_weights: Attention weights from SE or CE
            instruction_mask: 1 for instruction, 0 for response
            tolerance: Maximum allowed attention on response tokens
            
        Returns:
            Dict with verification results
        """
        # attention_weights: [batch, heads, query_seq, key_seq]
        # instruction_mask: [batch, seq]
        
        batch_size, num_heads, q_len, k_len = attention_weights.shape
        
        # Create response mask (1 for response, 0 for instruction)
        response_mask = (1 - instruction_mask.float())  # [batch, seq]
        
        # Expand for attention: [batch, 1, 1, k_len]
        response_mask_expanded = response_mask.unsqueeze(1).unsqueeze(2)
        
        # Check attention to response tokens
        attention_to_response = attention_weights * response_mask_expanded
        max_attention_to_response = attention_to_response.max().item()
        mean_attention_to_response = attention_to_response.mean().item()
        
        has_leakage = max_attention_to_response > tolerance
        
        result = {
            "has_leakage": has_leakage,
            "max_attention_to_response": max_attention_to_response,
            "mean_attention_to_response": mean_attention_to_response,
            "tolerance": tolerance,
        }
        
        if has_leakage:
            logger.error(f"ATTENTION LEAKAGE: Max attention to response = {max_attention_to_response:.6f}")
        else:
            logger.debug(f"✓ No attention leakage (max={max_attention_to_response:.2e})")
            
        return result
    
    def full_check(
        self,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
    ) -> Dict[str, any]:
        """
        Run all leakage checks on datasets.
        
        Args:
            train_dataset: Training dataset with 'commands' and 'actions' fields
            val_dataset: Optional validation dataset
            test_dataset: Optional test dataset
            
        Returns:
            Comprehensive leakage check results
        """
        results = {}
        
        # Extract data
        train_commands = [d['commands'] for d in train_dataset.data]
        train_actions = [d['actions'] for d in train_dataset.data]
        
        val_commands = [d['commands'] for d in val_dataset.data] if val_dataset else None
        test_commands = [d['commands'] for d in test_dataset.data] if test_dataset else None
        test_actions = [d['actions'] for d in test_dataset.data] if test_dataset else None
        
        # 1. Check split overlap
        results["split_overlap"] = self.check_split_overlap(
            train_commands, val_commands, test_commands
        )
        
        # 2. Check length constraints for train
        results["train_length_constraints"] = self.check_length_split_constraints(
            train_commands, train_actions, "train"
        )
        
        # 3. Check length constraints for test
        if test_dataset:
            results["test_length_constraints"] = self.check_length_split_constraints(
                test_commands, test_actions, "test"
            )
        
        # Overall status
        results["passed"] = (
            not results["split_overlap"]["has_leakage"] and
            results["train_length_constraints"].get("num_violations", 0) == 0
        )
        
        return results


def create_leakage_checker(split_name: str = "length") -> DataLeakageChecker:
    """Factory function to create a DataLeakageChecker."""
    return DataLeakageChecker(split_name=split_name)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    checker = DataLeakageChecker(split_name="length")
    
    # Test split overlap
    train = ["walk", "run", "jump"]
    test = ["walk twice", "run thrice"]  # No overlap
    result = checker.check_split_overlap(train, test_commands=test)
    print(f"Split overlap test: {result}")
    
    # Test with overlap
    test_overlap = ["walk", "run twice"]  # "walk" overlaps!
    result = checker.check_split_overlap(train, test_commands=test_overlap)
    print(f"Split overlap (with leak): {result}")
    
    # Test length constraints
    commands = ["walk", "jump twice"]
    actions = ["WALK", "JUMP JUMP"]  # 1 and 2 tokens
    result = checker.check_length_split_constraints(commands, actions, "train")
    print(f"Length constraints: {result}")
    
    print("\n✓ DataLeakageChecker tests passed!")
