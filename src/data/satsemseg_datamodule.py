from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.aerial_dataset import AerialDataset, collect_tile_files


class SatSemSegDataModule(LightningDataModule):
    """`LightningDataModule` for the Satellite Semantic Segmentation dataset.

    Expects the data directory to follow the structure::

        <data_dir>/
            Tile 1/
                images/image_part_001.jpg
                masks/image_part_001.png
                ...
            Tile 2/
                ...

    Mask files are RGB PNGs where each color maps to a class index via
    ``class_rgb`` (index in the list == class id).
    """

    def __init__(
        self,
        data_dir: str = "data/satsemseg",
        tiles: Optional[Sequence[str]] = None,
        class_rgb: Optional[Sequence[Sequence[int]]] = None,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        image_size: int = 512,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        augment_train: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize a `SatSemSegDataModule`.

        :param data_dir: Root directory containing the tile folders.
        :param tiles: Names of the tile subdirectories to load. Defaults to ``Tile 1`` … ``Tile 8``.
        :param class_rgb: List of per-class ``(R, G, B)`` triples; the list index is the class id.
        :param train_val_test_split: Fractions for the train / val / test split (must sum to 1).
        :param image_size: Size that images and masks are resized to (square).
        :param batch_size: Batch size across all devices.
        :param num_workers: DataLoader workers.
        :param pin_memory: Whether to pin memory in the DataLoader.
        :param augment_train: Whether to apply data augmentation on the training split.
        :param seed: RNG seed used for the deterministic split.
        """
        super().__init__()

        if tiles is None:
            tiles = [f"Tile {i}" for i in range(1, 9)]
        if class_rgb is None:
            class_rgb = [
                (226, 169, 41),   # Water
                (132, 41, 246),   # Land (unpaved area)
                (110, 193, 228),  # Road
                (60, 16, 152),    # Building
                (254, 221, 58),   # Vegetation
                (155, 155, 155),  # Unlabeled
            ]

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return len(self.hparams.class_rgb)

    def prepare_data(self) -> None:
        """No download required; data is provided locally under ``data_dir``."""
        pass

    def _rgb_to_class_map(self) -> Dict[Tuple[int, int, int], int]:
        return {tuple(int(c) for c in rgb): idx for idx, rgb in enumerate(self.hparams.class_rgb)}

    def setup(self, stage: Optional[str] = None) -> None:
        """Collect files, split deterministically, and create train/val/test datasets."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of "
                    f"devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is not None or self.data_val is not None or self.data_test is not None:
            return

        image_files, mask_files = collect_tile_files(
            data_dir=self.hparams.data_dir,
            tiles=list(self.hparams.tiles),
        )
        if len(image_files) == 0:
            raise RuntimeError(
                f"No image/mask pairs found under {self.hparams.data_dir} for tiles "
                f"{list(self.hparams.tiles)}"
            )

        train_idx, val_idx, test_idx = self._split_indices(len(image_files))

        rgb_to_class = self._rgb_to_class_map()

        def subset(indices: List[int], augment: bool) -> AerialDataset:
            imgs = [image_files[i] for i in indices]
            msks = [mask_files[i] for i in indices]
            return AerialDataset(
                image_files=imgs,
                mask_files=msks,
                rgb_to_class=rgb_to_class,
                image_size=self.hparams.image_size,
                augment=augment,
            )

        self.data_train = subset(train_idx, augment=self.hparams.augment_train)
        self.data_val = subset(val_idx, augment=False)
        self.data_test = subset(test_idx, augment=False)

    def _split_indices(self, n: int) -> Tuple[List[int], List[int], List[int]]:
        fractions = tuple(self.hparams.train_val_test_split)
        if not abs(sum(fractions) - 1.0) < 1e-6:
            raise ValueError(f"train_val_test_split must sum to 1.0, got {fractions}")

        generator = torch.Generator().manual_seed(int(self.hparams.seed))
        perm = torch.randperm(n, generator=generator).tolist()

        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        n_train = max(n_train, 1) if n >= 1 else 0
        n_val = max(n_val, 1) if n - n_train >= 1 else n_val
        n_test = n - n_train - n_val
        if n_test <= 0 and n_val > 1:
            n_val -= 1
            n_test = n - n_train - n_val

        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
        return train_idx, val_idx, test_idx

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = SatSemSegDataModule()
