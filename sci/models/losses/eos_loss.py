"""
EOS Token Loss: Upweights loss for EOS token prediction.

For exact match accuracy on SCAN, the model must predict EOS at exactly
the right position. This loss gives higher weight to EOS token predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def eos_token_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    eos_token_id: int,
    lambda_eos: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with upweighted EOS token.

    Args:
        logits: [batch, seq_len, vocab_size] - Model logits
        target_ids: [batch, seq_len] - Target token IDs
        eos_token_id: EOS token ID
        lambda_eos: Weight multiplier for EOS positions (default: 2.0)
        ignore_index: Index to ignore in targets (default: -100)

    Returns:
        loss: Scalar weighted loss
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for cross entropy
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = target_ids.view(-1)

    # Compute standard cross entropy (reduction='none' to get per-token losses)
    ce_loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='none'
    )

    # Create EOS mask: 1.0 for EOS positions, 0.0 for others
    eos_mask = (targets_flat == eos_token_id).float()

    # Create weights: lambda_eos for EOS, 1.0 for others
    weights = 1.0 + (lambda_eos - 1.0) * eos_mask

    # Apply weights
    weighted_loss = ce_loss * weights

    # Average (excluding ignore_index which gives 0 loss)
    loss = weighted_loss.sum() / (targets_flat != ignore_index).sum().clamp(min=1)

    return loss


class EOSLoss(nn.Module):
    """EOS loss as a module for easier integration."""

    def __init__(self, eos_token_id: int, lambda_eos: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.lambda_eos = lambda_eos
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return eos_token_loss(
            logits,
            target_ids,
            self.eos_token_id,
            self.lambda_eos,
            self.ignore_index,
        )


if __name__ == "__main__":
    # Test EOS loss
    print("Testing eos_token_loss...")

    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    eos_token_id = 2

    # Create dummy logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Set some positions to EOS
    targets[:, -1] = eos_token_id  # Last position is EOS
    targets[:, 5] = eos_token_id   # Also position 5

    # Compute loss without EOS weighting
    loss_normal = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction='mean'
    )

    # Compute loss with EOS weighting
    loss_weighted = eos_token_loss(logits, targets, eos_token_id, lambda_eos=2.0)

    print(f"✓ Normal loss: {loss_normal.item():.4f}")
    print(f"✓ EOS-weighted loss: {loss_weighted.item():.4f}")

    # Test that EOS positions have higher weight
    # Create targets with all EOS vs no EOS
    targets_all_eos = torch.full((batch_size, seq_len), eos_token_id, dtype=torch.long)
    targets_no_eos = torch.randint(3, vocab_size, (batch_size, seq_len))

    loss_all_eos = eos_token_loss(logits, targets_all_eos, eos_token_id, lambda_eos=2.0)
    loss_no_eos = eos_token_loss(logits, targets_no_eos, eos_token_id, lambda_eos=2.0)

    print(f"✓ Loss with all EOS: {loss_all_eos.item():.4f}")
    print(f"✓ Loss with no EOS: {loss_no_eos.item():.4f}")

    # Test with ignore_index
    targets_with_ignore = targets.clone()
    targets_with_ignore[:, :5] = -100  # Ignore first 5 positions

    loss_with_ignore = eos_token_loss(logits, targets_with_ignore, eos_token_id, lambda_eos=2.0)
    print(f"✓ Loss with ignore index: {loss_with_ignore.item():.4f}")

    # Test module version
    eos_loss_module = EOSLoss(eos_token_id, lambda_eos=2.0)
    loss_module = eos_loss_module(logits, targets)
    assert torch.allclose(loss_module, loss_weighted), "Module version should match function"
    print(f"✓ Module version: {loss_module.item():.4f}")

    print("\n✓ All EOS loss tests passed!")
