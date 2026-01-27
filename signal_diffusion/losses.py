"""Loss functions for signal diffusion training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard negative examples. It helps address extreme class
    imbalance by down-weighting easy examples and focusing on hard examples.

    Reference: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor for the rare class in range [0, 1].
            Lower values put more weight on the rare class.
            Default: 0.25
        gamma: Exponent of the modulating factor (1 - p_t)^gamma.
            Higher values focus more on hard examples.
            Default: 2.0
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits tensor of shape (B, C) where B is batch size
                and C is number of classes.
            targets: Target class indices of shape (B,).

        Returns:
            Scalar loss value.
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get the probability of the target class using gather
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute cross entropy loss
        ce = F.cross_entropy(inputs, targets, reduction="none")

        # Compute focal loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
