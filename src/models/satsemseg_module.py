from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex


class SatSemSegLitModule(LightningModule):
    """`LightningModule` for satellite semantic segmentation.

    Produces per-pixel class logits of shape ``[B, C, H, W]`` and tracks
    mean Intersection-over-Union (mIoU) across classes.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        num_classes: int = 6,
        ignore_index: int = 255,
        compile: bool = False,
    ) -> None:
        """Initialize a `SatSemSegLitModule`.

        :param net: Segmentation network producing logits ``[B, C, H, W]``.
        :param optimizer: Optimizer partial (Hydra ``_partial_: true``).
        :param scheduler: LR scheduler partial (Hydra ``_partial_: true``).
        :param criterion: Pixel-wise loss module, e.g. ``FocalDiceLoss``.
        :param num_classes: Number of segmentation classes.
        :param ignore_index: Pixel value in targets to ignore in metrics/loss.
        :param compile: Whether to ``torch.compile`` the network on ``fit``.
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net
        self.criterion = criterion

        iou_kwargs = dict(num_classes=num_classes, average="macro", ignore_index=ignore_index)
        self.train_iou = MulticlassJaccardIndex(**iou_kwargs)
        self.val_iou = MulticlassJaccardIndex(**iou_kwargs)
        self.test_iou = MulticlassJaccardIndex(**iou_kwargs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_iou_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_iou.reset()
        self.val_iou_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, masks = batch
        masks = masks.long()
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, masks

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_iou(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_iou(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        iou = self.val_iou.compute()
        self.val_iou_best(iou)
        self.log("val/iou_best", self.val_iou_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_iou(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/iou",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SatSemSegLitModule(None, None, None, None)
