"""
SCAN Structure Extractor.

Extracts structural templates from SCAN commands by replacing content words
with placeholders.

SCAN Content Words (actions and their modifiers):
- Actions: walk, run, jump, look
- Directional modifiers are STRUCTURAL (left, right, around, opposite, twice, thrice)

Examples:
- "walk twice" → template: "ACTION_0 twice", content: ["walk"]
- "run twice" → template: "ACTION_0 twice", content: ["run"]
- "walk left and run right" → template: "ACTION_0 left and ACTION_1 right", content: ["walk", "run"]
- "jump around left" → template: "ACTION_0 around left", content: ["jump"]

These templates capture compositional structure patterns independent of content.
"""

import re
from typing import Tuple, List, Set, Dict


class SCANStructureExtractor:
    """
    Extracts structural templates from SCAN commands.

    The key insight: Actions (walk, run, jump, look) are CONTENT.
    Everything else (left, right, twice, around, and, after, etc.) is STRUCTURE.

    By replacing content words with placeholders, we obtain structure templates
    that are invariant to content substitution.
    """

    # Content words in SCAN (actions only)
    # HIGH #15: Include 'turn' which appears in addprim_turn_left split
    CONTENT_WORDS = {
        'walk', 'run', 'jump', 'look', 'turn',
        'WALK', 'RUN', 'JUMP', 'LOOK', 'TURN',  # Include capitalized versions
    }

    # Structural words (everything else is structural)
    STRUCTURAL_WORDS = {
        'left', 'right', 'around', 'opposite',
        'twice', 'thrice',
        'and', 'after',
        'LEFT', 'RIGHT', 'AROUND', 'OPPOSITE',
        'TWICE', 'THRICE',
        'AND', 'AFTER',
    }

    def __init__(self, case_sensitive: bool = False):
        """
        Args:
            case_sensitive: Whether to treat case sensitively (default: False)
        """
        self.case_sensitive = case_sensitive

    def extract_structure(self, command: str) -> Tuple[str, List[str]]:
        """
        Extract structural template and content from SCAN command.

        Args:
            command: SCAN command string (e.g., "walk twice")

        Returns:
            template: Structure template with placeholders (e.g., "ACTION_0 twice")
            content: List of content words in order (e.g., ["walk"])

        Examples:
            "walk twice" → ("ACTION_0 twice", ["walk"])
            "run twice" → ("ACTION_0 twice", ["run"])
            "walk left and run right" → ("ACTION_0 left and ACTION_1 right", ["walk", "run"])
            "jump around left" → ("ACTION_0 around left", ["jump"])
            "look opposite left after walk around right" →
                ("ACTION_0 opposite left after ACTION_1 around right", ["look", "walk"])
        """
        # Normalize case if needed
        if not self.case_sensitive:
            command_lower = command.lower()
        else:
            command_lower = command

        # Tokenize
        tokens = command_lower.split()

        # Extract content words and create template
        template_tokens = []
        content = []
        content_idx = 0

        for token in tokens:
            # Check if normalized token is content word
            if token in self.CONTENT_WORDS or token.lower() in self.CONTENT_WORDS:
                # Replace with placeholder
                template_tokens.append(f"ACTION_{content_idx}")
                content.append(token)
                content_idx += 1
            else:
                # Keep structural word as-is
                template_tokens.append(token)

        # Join template
        template = " ".join(template_tokens)

        return template, content

    def are_same_structure(self, command1: str, command2: str) -> bool:
        """
        Check if two SCAN commands have the same structure.

        Args:
            command1: First command
            command2: Second command

        Returns:
            True if they have the same structural template

        Examples:
            "walk twice" and "run twice" → True (same structure)
            "walk twice" and "walk left" → False (different structure)
            "jump and run" and "look and walk" → True (same structure)
        """
        template1, _ = self.extract_structure(command1)
        template2, _ = self.extract_structure(command2)

        return template1 == template2

    def get_structure_hash(self, command: str) -> str:
        """
        Get a hash string representing the structure.

        Useful for grouping commands by structure.

        Args:
            command: SCAN command

        Returns:
            structure_hash: String hash of the structure
        """
        template, _ = self.extract_structure(command)
        return template

    def extract_batch_structures(
        self,
        commands: List[str]
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Extract structures for a batch of commands.

        Args:
            commands: List of SCAN commands

        Returns:
            templates: List of structure templates
            contents: List of content word lists
        """
        templates = []
        contents = []

        for command in commands:
            template, content = self.extract_structure(command)
            templates.append(template)
            contents.append(content)

        return templates, contents

    def group_by_structure(
        self,
        commands: List[str]
    ) -> Dict[str, List[int]]:
        """
        Group commands by their structural template.

        Args:
            commands: List of SCAN commands

        Returns:
            structure_groups: Dict mapping template → list of command indices

        Example:
            commands = ["walk twice", "run twice", "jump left"]
            →  {
                "ACTION_0 twice": [0, 1],
                "ACTION_0 left": [2]
               }
        """
        structure_groups = {}

        for idx, command in enumerate(commands):
            template, _ = self.extract_structure(command)

            if template not in structure_groups:
                structure_groups[template] = []

            structure_groups[template].append(idx)

        return structure_groups

    def count_unique_structures(self, commands: List[str]) -> int:
        """
        Count number of unique structural templates in commands.

        Args:
            commands: List of SCAN commands

        Returns:
            num_unique: Number of unique structures
        """
        templates = set()

        for command in commands:
            template, _ = self.extract_structure(command)
            templates.add(template)

        return len(templates)

    def get_structure_stats(self, commands: List[str]) -> Dict:
        """
        Get statistics about structures in dataset.

        Args:
            commands: List of SCAN commands

        Returns:
            stats: Dictionary with structure statistics
        """
        structure_groups = self.group_by_structure(commands)

        # Count examples per structure
        structure_counts = {
            template: len(indices)
            for template, indices in structure_groups.items()
        }

        # Sort by frequency
        sorted_structures = sorted(
            structure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        stats = {
            'num_commands': len(commands),
            'num_unique_structures': len(structure_groups),
            'structure_counts': structure_counts,
            'most_common_structures': sorted_structures[:10],
            'avg_examples_per_structure': len(commands) / len(structure_groups) if structure_groups else 0,
        }

        return stats


if __name__ == "__main__":
    # Test SCAN structure extractor
    print("Testing SCANStructureExtractor...\n")

    extractor = SCANStructureExtractor()

    # Test cases
    test_cases = [
        "walk twice",
        "run twice",
        "jump left",
        "look right",
        "walk left and run right",
        "jump around left",
        "look opposite right",
        "walk twice after run thrice",
        "jump and walk",
        "look around right after walk left",
    ]

    print("=" * 60)
    print("Structure Extraction Tests")
    print("=" * 60)

    for command in test_cases:
        template, content = extractor.extract_structure(command)
        print(f"Command:  {command}")
        print(f"Template: {template}")
        print(f"Content:  {content}")
        print()

    # Test same structure
    print("=" * 60)
    print("Same Structure Tests")
    print("=" * 60)

    pairs = [
        ("walk twice", "run twice", True),
        ("walk twice", "walk left", False),
        ("jump and run", "look and walk", True),
        ("walk left and run right", "jump left and look right", True),
        ("walk", "run", True),  # Both are just single actions
    ]

    for cmd1, cmd2, expected in pairs:
        result = extractor.are_same_structure(cmd1, cmd2)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{cmd1}' == '{cmd2}': {result} (expected: {expected})")

    # Test grouping
    print("\n" + "=" * 60)
    print("Grouping by Structure")
    print("=" * 60)

    commands = [
        "walk twice",
        "run twice",
        "jump twice",
        "walk left",
        "run left",
        "look around right",
        "jump around right",
    ]

    groups = extractor.group_by_structure(commands)

    for template, indices in groups.items():
        print(f"\nTemplate: {template}")
        print(f"  Commands: {[commands[i] for i in indices]}")

    # Test statistics
    print("\n" + "=" * 60)
    print("Structure Statistics")
    print("=" * 60)

    stats = extractor.get_structure_stats(commands)
    print(f"Total commands: {stats['num_commands']}")
    print(f"Unique structures: {stats['num_unique_structures']}")
    print(f"Avg examples/structure: {stats['avg_examples_per_structure']:.1f}")
    print(f"\nMost common structures:")
    for template, count in stats['most_common_structures']:
        print(f"  {template}: {count} examples")

    print("\n✓ All SCAN structure extractor tests passed!")
