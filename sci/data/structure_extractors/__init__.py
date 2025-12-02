"""
Structure Extractors for different datasets.

Each extractor defines:
1. CONTENT_WORDS: Set of words considered content (not structure)
2. extract_structure(text): Extract structural template from text
3. are_same_structure(text1, text2): Check if two texts have same structure
"""

from sci.data.structure_extractors.scan_extractor import SCANStructureExtractor

__all__ = ['SCANStructureExtractor']
