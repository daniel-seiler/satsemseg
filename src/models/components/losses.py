import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss for dense prediction, focuses learning on hard examples."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)

        targets_expanded = targets.unsqueeze(1)
        log_pt = torch.gather(log_probs, 1, targets_expanded.clamp(min=0)).squeeze(1)
        pt = torch.exp(log_pt)

        mask = (targets != self.ignore_index).float()
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        return (loss * mask).sum() / (mask.sum() + 1e-6)


class DiceLoss(nn.Module):
    """Multiclass Dice Loss computed over softmax probabilities."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        num_classes = probs.shape[1]

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        probs_flat = probs.reshape(probs.size(0), num_classes, -1)
        targets_flat = targets_one_hot.reshape(targets_one_hot.size(0), num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Weighted sum of Focal Loss and Dice Loss."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(inputs, targets) + self.dice_weight * self.dice(
            inputs, targets
        )


if __name__ == "__main__":
    _ = FocalDiceLoss()
