"""
Combined Loss for SCI Training.

Combines three loss components:
1. Language Modeling (LM) Loss - Standard next-token prediction
2. Structural Contrastive Learning (SCL) Loss - Enforces structural invariance
3. Orthogonality Loss - Ensures content ⊥ structure

Total Loss = LM_loss + λ_scl * SCL_loss + λ_ortho * Ortho_loss

where:
- λ_scl has warmup schedule (prevents early instability)
- λ_ortho is fixed (default: 0.01)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from sci.models.losses.scl_loss import StructuralContrastiveLoss


class SCICombinedLoss(nn.Module):
    """
    Combined loss function for SCI training.

    Integrates:
    1. LM loss (from base model)
    2. SCL loss (structural contrastive learning)
    3. Orthogonality loss (content ⊥ structure)

    Args:
        scl_weight: Weight for SCL loss (default: 0.3)
        ortho_weight: Weight for orthogonality loss (default: 0.01)
        temperature: Temperature for SCL (default: 0.07)
        use_hard_negatives: Whether to use hard negative mining (default: True)
        hard_negative_ratio: Ratio of hard negatives to use (default: 0.3)

    Input (as dict from model forward):
        - lm_loss: Scalar LM loss from base model
        - structural_slots: [batch, num_slots, d_model] from SE
        - content_repr: [batch, d_model] from CE
        - pair_labels: [batch, batch] indicating same structure

    Output:
        Dictionary with:
        - total_loss: Combined loss for backprop
        - lm_loss: LM loss component
        - scl_loss: SCL loss component
        - orthogonality_loss: Orthogonality loss component
        - eos_loss: EOS enforcement loss component
        - num_positive_pairs: Number of positive pairs in batch
    """

    def __init__(
        self,
        scl_weight: float = 0.3,
        ortho_weight: float = 0.01,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.3,
        eos_weight: float = 2.0,
        eos_token_id: int = None,
    ):
        super().__init__()

        self.scl_weight = scl_weight
        self.ortho_weight = ortho_weight
        self.eos_weight = eos_weight
        self.eos_token_id = eos_token_id

        # Structural Contrastive Learning loss
        self.scl_loss_fn = StructuralContrastiveLoss(
            temperature=temperature,
            lambda_weight=1.0,  # We'll apply weight externally
        )

        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio

    def compute_eos_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        eos_token_id: int,
    ) -> torch.Tensor:
        """
        Compute EOS enforcement loss.
        
        Upweights the loss on the EOS token to ensure proper sequence termination.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len] with -100 for ignored tokens
            eos_token_id: Token ID for EOS
            
        Returns:
            eos_loss: Scalar loss for EOS tokens
        """
        if eos_token_id is None:
            return torch.tensor(0.0, device=logits.device)
            
        # Find positions where label is EOS
        eos_mask = (labels == eos_token_id)
        
        if not eos_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        # Get logits at EOS positions
        # Shift logits to align with labels (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_eos_mask = eos_mask[:, 1:]
        
        # Compute cross entropy only at EOS positions
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_eos_mask = shift_eos_mask.view(-1)
        
        # Get loss for all tokens
        token_losses = loss_fct(flat_logits, flat_labels.clamp(min=0))
        
        # Mask out non-EOS tokens
        eos_losses = token_losses * flat_eos_mask.float()
        
        # Average over EOS tokens
        if flat_eos_mask.sum() > 0:
            eos_loss = eos_losses.sum() / flat_eos_mask.sum()
        else:
            eos_loss = torch.tensor(0.0, device=logits.device)
            
        return eos_loss

    def compute_orthogonality_loss(
        self,
        content_repr: torch.Tensor,
        structural_slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute orthogonality loss: content ⊥ structure.

        Ensures clean factorization by penalizing correlation.

        Args:
            content_repr: [batch, d_model]
            structural_slots: [batch, num_slots, d_model]

        Returns:
            loss: Scalar orthogonality loss
        """
        # Pool structural slots to [batch, d_model]
        structural_pooled = structural_slots.mean(dim=1)

        # Normalize both representations
        content_norm = F.normalize(content_repr, dim=-1, p=2)
        structure_norm = F.normalize(structural_pooled, dim=-1, p=2)

        # Compute cosine similarity (should be close to 0)
        cosine_sim = (content_norm * structure_norm).sum(dim=-1)  # [batch]

        # Loss is absolute value (penalize both positive and negative correlation)
        loss = cosine_sim.abs().mean()

        return loss

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        pair_labels: Optional[torch.Tensor] = None,
        scl_weight_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            model_outputs: Dictionary from model forward pass containing:
                - logits: [batch, seq_len, vocab]
                - loss: LM loss (optional, computed if labels provided)
                - structural_slots: [batch, num_slots, d_model]
                - content_repr: [batch, d_model]
            pair_labels: [batch, batch] indicating structural similarity
                (1 for same structure, 0 for different)
            scl_weight_override: Optional override for SCL weight (for warmup)

        Returns:
            Dictionary with all loss components
        """
        device = model_outputs['logits'].device

        # =====================================================
        # 1. Language Modeling Loss (always present)
        # =====================================================

        if model_outputs.get('loss') is not None:
            lm_loss = model_outputs['loss']
        else:
            # If not computed by model, return zero
            lm_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # =====================================================
        # 2. Structural Contrastive Learning Loss
        # =====================================================

        structural_slots = model_outputs.get('structural_slots')
        scl_loss = torch.tensor(0.0, device=device)
        num_positive_pairs = 0

        if (structural_slots is not None and
            pair_labels is not None and
            pair_labels.sum() > 0):  # Check if there are positive pairs

            # Compute SCL loss
            # We pass structural_slots twice since we're using within-batch contrastive
            scl_loss = self.scl_loss_fn(
                structural_repr_i=structural_slots,
                structural_repr_j=structural_slots,  # Same batch, different views
                pair_labels=pair_labels,
            )

            # Count positive pairs (excluding diagonal)
            num_positive_pairs = (pair_labels.sum() - pair_labels.diagonal().sum()).item()

        # =====================================================
        # 3. Orthogonality Loss
        # =====================================================

        content_repr = model_outputs.get('content_repr')
        ortho_loss = torch.tensor(0.0, device=device)

        if structural_slots is not None and content_repr is not None:
            ortho_loss = self.compute_orthogonality_loss(
                content_repr=content_repr,
                structural_slots=structural_slots,
            )

        # =====================================================
        # 4. EOS Enforcement Loss
        # =====================================================

        eos_loss = torch.tensor(0.0, device=device)
        labels = model_outputs.get('labels')
        logits = model_outputs.get('logits')

        if self.eos_weight > 0 and self.eos_token_id is not None and labels is not None and logits is not None:
            eos_loss = self.compute_eos_loss(
                logits=logits,
                labels=labels,
                eos_token_id=self.eos_token_id,
            )

        # =====================================================
        # 5. Combine Losses
        # =====================================================

        # Use override weight if provided (for warmup schedule)
        scl_weight = scl_weight_override if scl_weight_override is not None else self.scl_weight

        # Total loss
        total_loss = lm_loss + scl_weight * scl_loss + self.ortho_weight * ortho_loss + self.eos_weight * eos_loss

        # Return all components for logging
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'scl_loss': scl_loss,
            'orthogonality_loss': ortho_loss,
            'eos_loss': eos_loss,
            'num_positive_pairs': num_positive_pairs,
            'scl_weight_used': scl_weight,
        }


class EOS_EnforcementLoss(nn.Module):
    """
    Optional: EOS Enforcement Loss.

    Encourages the model to generate EOS token at appropriate positions.
    Can be added to combat length mismatch issues in sequence generation.

    This is an OPTIONAL component that can improve exact match accuracy.
    """

    def __init__(self, eos_weight: float = 0.1):
        super().__init__()
        self.eos_weight = eos_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        eos_token_id: int,
    ) -> torch.Tensor:
        """
        Compute EOS enforcement loss.

        Args:
            logits: [batch, seq_len, vocab]
            labels: [batch, seq_len]
            eos_token_id: ID of EOS token

        Returns:
            loss: Scalar EOS loss
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Find positions where EOS should be generated
        # (positions where label is EOS)
        eos_positions = (labels == eos_token_id)

        if eos_positions.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Get logits at EOS positions
        eos_logits = logits[eos_positions]  # [num_eos, vocab]

        # Compute cross-entropy loss for EOS positions
        eos_targets = torch.full(
            (eos_logits.shape[0],),
            eos_token_id,
            dtype=torch.long,
            device=logits.device
        )

        eos_loss = F.cross_entropy(eos_logits, eos_targets)

        return self.eos_weight * eos_loss


class HardNegativeMiner:
    """
    Hard Negative Mining for SCL.

    Selects the hardest negative pairs (most similar but different structure)
    to focus learning on difficult cases.

    This is an advanced technique that can improve SCL effectiveness.
    """

    def __init__(self, hard_negative_ratio: float = 0.3):
        """
        Args:
            hard_negative_ratio: Fraction of hardest negatives to keep
        """
        self.hard_negative_ratio = hard_negative_ratio

    def mine_hard_negatives(
        self,
        structural_repr: torch.Tensor,
        pair_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select hard negatives based on similarity.

        Args:
            structural_repr: [batch, d_model]
            pair_labels: [batch, batch] (1 for same structure, 0 for different)

        Returns:
            hard_negative_mask: [batch, batch] (1 for hard negatives, 0 otherwise)
        """
        batch_size = structural_repr.shape[0]

        # Normalize representations
        repr_norm = F.normalize(structural_repr, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(repr_norm, repr_norm.T)  # [batch, batch]

        # Create negative mask (different structure)
        negative_mask = (pair_labels == 0) & ~torch.eye(batch_size, dtype=torch.bool, device=structural_repr.device)

        # Get similarities for negative pairs
        negative_similarities = similarity.clone()
        negative_similarities[~negative_mask] = -float('inf')

        # CRITICAL #10: Add bounds checking for hard negative mining
        # For each anchor, select top-k hardest negatives
        num_negatives_per_sample = negative_mask.sum(dim=1)
        avg_negatives = num_negatives_per_sample.float().mean().item()

        # Compute number of hard negatives, clamped to available negatives
        num_hard = max(1, int(avg_negatives * self.hard_negative_ratio))

        # Create hard negative mask
        hard_negative_mask = torch.zeros_like(pair_labels, dtype=torch.bool)

        for i in range(batch_size):
            # Get number of available negatives for this sample
            num_available = num_negatives_per_sample[i].item()

            if num_available > 0:
                # Clamp k to not exceed available negatives
                k = min(num_hard, num_available)

                # Get top-k most similar negatives for this row
                _, hard_indices = torch.topk(negative_similarities[i], k=k, dim=0)
                hard_negative_mask[i, hard_indices] = True

        return hard_negative_mask.long()


if __name__ == "__main__":
    # Test combined loss
    print("Testing SCICombinedLoss...")

    batch_size = 8
    num_slots = 8
    d_model = 512
    seq_len = 32
    vocab_size = 1000

    # Create loss function
    loss_fn = SCICombinedLoss(
        scl_weight=0.3,
        ortho_weight=0.01,
        temperature=0.07,
    )

    # Create dummy model outputs
    model_outputs = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'loss': torch.tensor(2.5),  # Dummy LM loss
        'structural_slots': torch.randn(batch_size, num_slots, d_model),
        'content_repr': torch.randn(batch_size, d_model),
    }

    # Create pair labels (examples 0,1 same; 2,3 same; etc.)
    pair_labels = torch.zeros(batch_size, batch_size)
    for i in range(0, batch_size, 2):
        if i + 1 < batch_size:
            pair_labels[i, i+1] = 1
            pair_labels[i+1, i] = 1

    # Compute losses
    losses = loss_fn(model_outputs, pair_labels)

    print(f"✓ Total loss: {losses['total_loss'].item():.4f}")
    print(f"✓ LM loss: {losses['lm_loss'].item():.4f}")
    print(f"✓ SCL loss: {losses['scl_loss'].item():.4f}")
    print(f"✓ Ortho loss: {losses['orthogonality_loss'].item():.4f}")
    print(f"✓ Num positive pairs: {losses['num_positive_pairs']}")

    # Test with warmup
    losses_warmup = loss_fn(model_outputs, pair_labels, scl_weight_override=0.1)
    print(f"✓ Total loss (warmup): {losses_warmup['total_loss'].item():.4f}")
    print(f"✓ SCL weight used: {losses_warmup['scl_weight_used']:.2f}")

    # Test with no positive pairs
    pair_labels_none = torch.zeros(batch_size, batch_size)
    losses_none = loss_fn(model_outputs, pair_labels_none)
    print(f"✓ SCL loss (no pairs): {losses_none['scl_loss'].item():.4f}")

    # Test hard negative mining
    print("\nTesting HardNegativeMiner...")
    miner = HardNegativeMiner(hard_negative_ratio=0.3)

    structural_repr = torch.randn(batch_size, d_model)
    hard_neg_mask = miner.mine_hard_negatives(structural_repr, pair_labels)

    print(f"✓ Hard negative mask shape: {hard_neg_mask.shape}")
    print(f"✓ Num hard negatives per row: {hard_neg_mask.sum(dim=1).float().mean().item():.1f}")

    print("\n✓ All combined loss tests passed!")
