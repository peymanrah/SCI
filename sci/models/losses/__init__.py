"""Loss functions for SCI training."""

from sci.models.losses.scl_loss import StructuralContrastiveLoss
from sci.models.losses.orthogonality_loss import orthogonality_loss
from sci.models.losses.eos_loss import eos_token_loss

__all__ = [
    "StructuralContrastiveLoss",
    "orthogonality_loss",
    "eos_token_loss",
]
