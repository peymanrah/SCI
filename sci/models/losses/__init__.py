"""Loss functions for SCI training."""

from sci.models.losses.scl_loss import StructuralContrastiveLoss
from sci.models.losses.eos_loss import eos_token_loss, EOSLoss
from sci.models.losses.combined_loss import SCICombinedLoss

__all__ = [
    "StructuralContrastiveLoss",
    "eos_token_loss",
    "EOSLoss",
    "SCICombinedLoss",
]
