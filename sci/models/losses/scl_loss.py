"""
Structural Contrastive Learning (SCL) Loss.

This loss enforces structural invariance by training the model to produce
similar structural representations for examples with the same structure
but different content.

Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralContrastiveLoss(nn.Module):
    """
    Structural Contrastive Learning loss using NT-Xent.

    Positive pairs: Same structure, different content
    Negative pairs: Different structure

    Example:
        Positive: "walk twice" <-> "run twice" (same [X twice] structure)
        Negative: "walk twice" <-> "walk and run" (different structure)

    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
        lambda_weight: Loss weight (default: 0.3)

    Input:
        structural_repr_i: [batch, num_slots, d_model] or [batch, d_model]
        structural_repr_j: [batch, num_slots, d_model] or [batch, d_model]
        pair_labels: [batch, batch] - 1 for positive pairs, 0 for negative

    Output:
        loss: Scalar contrastive loss
    """

    def __init__(self, temperature: float = 0.07, lambda_weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight

    def forward(
        self,
        structural_repr_i: torch.Tensor,
        structural_repr_j: torch.Tensor,
        pair_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SCL loss.

        Args:
            structural_repr_i: [batch, num_slots, d_model] or [batch, d_model]
            structural_repr_j: [batch, num_slots, d_model] or [batch, d_model]
            pair_labels: [batch, batch] where 1 = same structure, 0 = different

        Returns:
            loss: Scalar contrastive loss
        """
        # Pool to [batch, d_model] if slots dimension exists
        if structural_repr_i.dim() == 3:
            repr_i = structural_repr_i.mean(dim=1)  # Average over slots
        else:
            repr_i = structural_repr_i

        if structural_repr_j.dim() == 3:
            repr_j = structural_repr_j.mean(dim=1)
        else:
            repr_j = structural_repr_j

        batch_size = repr_i.shape[0]

        # Normalize representations
        repr_i = F.normalize(repr_i, dim=-1)
        repr_j = F.normalize(repr_j, dim=-1)

        # Compute similarity matrix
        # [batch, batch] where element [i, j] is cosine similarity
        similarity_matrix = torch.matmul(repr_i, repr_j.T) / self.temperature

        # Create labels for contrastive learning
        # Positive pairs should have label 1, negative pairs should have label 0
        # pair_labels is already in this format

        # NT-Xent loss computation
        # For each anchor i, we want to maximize similarity to positive pairs
        # and minimize similarity to negative pairs

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=repr_i.device)

        # Get positive pairs mask
        positive_mask = pair_labels.bool() & ~self_mask

        # Check if there are any positive pairs
        if positive_mask.sum() == 0:
            # No positive pairs in batch - return zero loss
            return torch.tensor(0.0, device=repr_i.device, requires_grad=True)

        # Compute loss for each anchor
        losses = []

        for i in range(batch_size):
            # Get positive pairs for anchor i
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0:
                continue  # No positive pairs for this anchor

            # Similarities for anchor i
            sim_i = similarity_matrix[i]

            # For each positive pair
            for pos_idx in pos_indices:
                # Numerator: exp(sim(i, positive))
                numerator = torch.exp(sim_i[pos_idx])

                # Denominator: sum of exp(sim(i, k)) for all k != i
                # Include both positive and negative pairs
                denominator_mask = torch.ones(batch_size, dtype=torch.bool, device=repr_i.device)
                denominator_mask[i] = False  # Exclude self

                denominator = torch.exp(sim_i[denominator_mask]).sum()

                # Loss for this positive pair
                loss_ij = -torch.log(numerator / (denominator + 1e-8))
                losses.append(loss_ij)

        if len(losses) == 0:
            # No valid positive pairs
            return torch.tensor(0.0, device=repr_i.device, requires_grad=True)

        # Average loss over all positive pairs
        loss = torch.stack(losses).mean()

        return self.lambda_weight * loss


class SimplifiedSCLLoss(nn.Module):
    """
    Simplified SCL loss using InfoNCE formulation.

    This is computationally more efficient for large batches.
    """

    def __init__(self, temperature: float = 0.07, lambda_weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight

    def forward(
        self,
        structural_repr: torch.Tensor,
        pair_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified SCL using single batch of representations.

        Args:
            structural_repr: [batch, d_model] - Structural representations
            pair_labels: [batch, batch] - 1 for same structure, 0 for different

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = structural_repr.shape[0]

        # Normalize
        repr_norm = F.normalize(structural_repr, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(repr_norm, repr_norm.T) / self.temperature

        # Mask out diagonal
        mask = torch.eye(batch_size, dtype=torch.bool, device=structural_repr.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Convert pair_labels to float and mask diagonal
        labels = pair_labels.float()
        labels = labels.masked_fill(mask, 0)

        # Check if there are positive pairs
        if labels.sum() == 0:
            return torch.tensor(0.0, device=structural_repr.device, requires_grad=True)

        # Compute log softmax over similarities
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # Weight by positive pairs and average
        # For each row, average log probability of positive pairs
        positive_log_prob = (log_prob * labels).sum(dim=1)
        num_positives = labels.sum(dim=1).clamp(min=1)

        loss = -(positive_log_prob / num_positives).mean()

        return self.lambda_weight * loss


if __name__ == "__main__":
    # Test SCL Loss
    print("Testing StructuralContrastiveLoss...")

    batch_size = 8
    num_slots = 8
    d_model = 512

    # Create SCL loss
    scl = StructuralContrastiveLoss(temperature=0.07, lambda_weight=0.3)

    # Test case 1: With slot representations
    repr_i = torch.randn(batch_size, num_slots, d_model)
    repr_j = torch.randn(batch_size, num_slots, d_model)

    # Create pair labels
    # Let's say examples 0,1 have same structure, 2,3 have same structure, etc.
    pair_labels = torch.zeros(batch_size, batch_size)
    pair_labels[0, 1] = 1
    pair_labels[1, 0] = 1
    pair_labels[2, 3] = 1
    pair_labels[3, 2] = 1
    pair_labels[4, 5] = 1
    pair_labels[5, 4] = 1

    loss = scl(repr_i, repr_j, pair_labels)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.requires_grad, "Loss should require gradients"
    print(f"✓ Loss with slots (should be > 0): {loss.item():.4f}")

    # Test case 2: Similar positive pairs should give lower loss
    repr_i_similar = torch.randn(batch_size, d_model)
    repr_j_similar = repr_i_similar + 0.01 * torch.randn(batch_size, d_model)  # Very similar

    repr_i_dissimilar = torch.randn(batch_size, d_model)
    repr_j_dissimilar = torch.randn(batch_size, d_model)  # Random

    loss_similar = scl(repr_i_similar, repr_j_similar, pair_labels)
    loss_dissimilar = scl(repr_i_dissimilar, repr_j_dissimilar, pair_labels)

    print(f"✓ Loss with similar pairs: {loss_similar.item():.4f}")
    print(f"✓ Loss with dissimilar pairs: {loss_dissimilar.item():.4f}")
    assert loss_similar < loss_dissimilar, "Similar pairs should have lower loss"

    # Test case 3: No positive pairs
    pair_labels_none = torch.zeros(batch_size, batch_size)
    loss_none = scl(repr_i, repr_j, pair_labels_none)
    assert loss_none.item() == 0.0, "No positive pairs should give zero loss"
    print(f"✓ Loss with no positive pairs: {loss_none.item():.4f}")

    # Test temperature effect
    scl_low_temp = StructuralContrastiveLoss(temperature=0.01, lambda_weight=0.3)
    scl_high_temp = StructuralContrastiveLoss(temperature=0.5, lambda_weight=0.3)

    loss_low_temp = scl_low_temp(repr_i, repr_j, pair_labels)
    loss_high_temp = scl_high_temp(repr_i, repr_j, pair_labels)

    print(f"✓ Loss with low temperature: {loss_low_temp.item():.4f}")
    print(f"✓ Loss with high temperature: {loss_high_temp.item():.4f}")
    assert loss_low_temp != loss_high_temp, "Temperature should affect loss"

    # Test simplified SCL
    print("\nTesting SimplifiedSCLLoss...")
    scl_simple = SimplifiedSCLLoss(temperature=0.07, lambda_weight=0.3)

    repr_single = torch.randn(batch_size, d_model)
    loss_simple = scl_simple(repr_single, pair_labels)

    assert loss_simple.dim() == 0
    assert loss_simple.requires_grad
    print(f"✓ Simplified SCL loss: {loss_simple.item():.4f}")

    print("\n✓ All SCL loss tests passed!")
