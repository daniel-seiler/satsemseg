from typing import Any, Dict, List, Sequence, Tuple

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# ImageNet mean/std used to normalize inputs in AerialDataset.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and return an HWC uint8 array."""
    img = image.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map an ``[H, W]`` class-index array to an ``[H, W, 3]`` uint8 RGB image.

    Indices outside the palette (e.g. ``ignore_index``) are rendered black.
    """
    flat = mask.reshape(-1)
    valid = flat < len(palette)
    rgb = np.zeros((flat.size, 3), dtype=np.uint8)
    rgb[valid] = palette[flat[valid]]
    return rgb.reshape(mask.shape[0], mask.shape[1], 3)


def log_qualitative_examples(
    model: LightningModule,
    datamodule: LightningDataModule,
    loggers: Sequence[Logger],
    class_rgb: Sequence[Sequence[int]],
    num_examples: int = 4,
) -> None:
    """Run inference on a few test samples and log input + colored prediction to Aim."""
    try:
        from aim import Image as AimImage
        from aim.pytorch_lightning import AimLogger
    except ImportError:
        log.warning("aim is not installed; skipping qualitative example logging.")
        return

    aim_logger = next((lg for lg in loggers if isinstance(lg, AimLogger)), None)
    if aim_logger is None:
        log.warning("No AimLogger found; skipping qualitative example logging.")
        return

    datamodule.setup(stage="test")
    dataset = datamodule.data_test
    if dataset is None or len(dataset) == 0:
        log.warning("Test dataset is empty; skipping qualitative example logging.")
        return

    n = min(num_examples, len(dataset))
    samples = [dataset[i] for i in range(n)]
    images = torch.stack([s[0] for s in samples])
    gt_masks = torch.stack([s[1] for s in samples]).cpu().numpy()

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    if was_training:
        model.train()

    palette = np.array(class_rgb, dtype=np.uint8)
    run = aim_logger.experiment

    for i in range(n):
        input_rgb = _denormalize_image(images[i])
        pred_rgb = _colorize_mask(preds[i], palette)
        gt_rgb = _colorize_mask(gt_masks[i], palette)
        panel = np.concatenate([input_rgb, gt_rgb, pred_rgb], axis=1)
        run.track(
            AimImage(panel, caption=f"input | ground truth | prediction (sample {i})"),
            name="test/qualitative",
            step=i,
            context={"subset": "test"},
        )

    log.info(f"Logged {n} qualitative examples to Aim under 'test/qualitative'.")


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    if trainer.is_global_zero:
        log.info("Logging qualitative inference examples!")
        log_qualitative_examples(
            model=model,
            datamodule=datamodule,
            loggers=trainer.loggers,
            class_rgb=cfg.data.class_rgb,
            num_examples=cfg.get("num_qualitative_examples", 4),
        )

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
