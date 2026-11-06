
import torch
import torch.nn as nn
import torch.nn.functional as F

class HardNegativeSuppressionLoss(nn.Module):
    """
    Computes BCE in float32 outside autocast (preventing NaN) while heavily penalizing
    high-confidence false positives (y == 0) on focus classes and confounder scans.
    """
    def __init__(self, focus_indices, fp_weight=3.5, gamma=2.0):
        super().__init__()
        self.focus_indices = focus_indices
        self.fp_weight = fp_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        y = y.float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        probs = torch.sigmoid(logits).detach()

        weights = torch.ones_like(bce)
        is_abn = y[:, -1:]  # (B, 1) -> 1 if scan has any abnormality

        for idx in self.focus_indices:
            neg_mask = (y[:, idx] == 0.0).float()
            conf_penalty = (probs[:, idx] ** self.gamma) * self.fp_weight * (1.0 + 0.5 * is_abn.squeeze(1))
            weights[:, idx] = 1.0 + neg_mask * conf_penalty

        return (bce * weights).mean()
