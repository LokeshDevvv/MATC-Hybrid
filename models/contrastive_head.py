"""Supervised Contrastive Learning head for MATC-Net.

Projects fused representations into a normalized embedding space and computes
SupCon loss to pull same-class samples together and push different classes apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjectionHead(nn.Module):
    """MLP projection head that maps fused representations to a unit sphere.

    Architecture:
        Linear(d_model -> 512) -> BatchNorm -> ReLU -> Linear(512 -> projection_dim) -> L2-Norm
    """

    def __init__(self, d_model: int = 768, projection_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, h_fused):
        """
        Args:
            h_fused: (batch, d_model)
        Returns:
            z: (batch, projection_dim), L2-normalized
        """
        z = self.projector(h_fused)
        z = F.normalize(z, p=2, dim=-1)
        return z


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor, pulls together all positives (same class) and pushes apart
    all negatives (different class) in the projected embedding space.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (batch, projection_dim) — L2-normalized projections
            labels: (batch,) — class labels
        Returns:
            loss: scalar SupCon loss
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pairwise cosine similarity (features already L2-normalized)
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask for positive pairs: same label, excluding self
        labels_col = labels.unsqueeze(0)  # (1, B)
        labels_row = labels.unsqueeze(1)  # (B, 1)
        positive_mask = torch.eq(labels_row, labels_col).float()  # (B, B)
        positive_mask.fill_diagonal_(0.0)  # exclude self-pairs

        # Check if there are any positive pairs at all
        num_positives = positive_mask.sum(dim=1)
        if (num_positives == 0).all():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerical stability: subtract max logit
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask out self from denominator
        self_mask = torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * (1.0 - self_mask)

        # Log-softmax denominator
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log probability over positive pairs
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

        # Only average over anchors that have at least one positive
        valid_anchors = (num_positives > 0).float()
        loss = -(mean_log_prob * valid_anchors).sum() / (valid_anchors.sum() + 1e-8)

        return loss
