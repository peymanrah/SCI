"""
SCI: Structural Causal Invariance for Compositional Generalization

This package implements the SCI architecture for achieving compositional
generalization on sequence-to-sequence tasks.
"""

__version__ = "0.1.0"

from sci.models import SCITinyLlama
from sci.config import load_config

__all__ = ["SCITinyLlama", "load_config"]
