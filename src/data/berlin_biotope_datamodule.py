from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.berlin_biotope_dataset import BerlinBiotopeDataset, collect_biotope_files


class BerlinBiotopeDataModule(LightningDataModule):
    """`LightningDataModule` for the Berlin Biotope dataset.

    Expects data at::

        <data_dir>/
            core/
                images/tile_XXXXX_rgb.png
                nir/tile_XXXXX_nir.png
                height/tile_XXXXX_height.npy
                labels/tile_XXXXX_label.png

    Labels are PNGs with class IDs 0-12 stored identically in all RGB channels.
    """

    def __init__(
        self,
        data_dir: str = "data/geo",
        use_nir: bool = False,
        use_height: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        image_size: int = 512,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        augment_train: bool = True,
        seed: int = 42,
        rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        nir_mean: float = 0.5,
        nir_std: float = 0.25,
        height_mean: float = 5.0,
        height_std: float = 8.0,
        class_rgb: Optional[List[List[int]]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 13

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is not None or self.data_val is not None or self.data_test is not None:
            return

        rgb_files, nir_files, height_files, label_files = collect_biotope_files(
            data_dir=self.hparams.data_dir,
            use_nir=self.hparams.use_nir,
            use_height=self.hparams.use_height,
        )
        if len(rgb_files) == 0:
            raise RuntimeError(f"No tiles found under {self.hparams.data_dir}/core/images/")

        train_idx, val_idx, test_idx = self._split_indices(len(rgb_files))

        def subset(indices: List[int], augment: bool) -> BerlinBiotopeDataset:
            return BerlinBiotopeDataset(
                rgb_files=[rgb_files[i] for i in indices],
                label_files=[label_files[i] for i in indices],
                nir_files=[nir_files[i] for i in indices] if nir_files is not None else None,
                height_files=[height_files[i] for i in indices] if height_files is not None else None,
                image_size=self.hparams.image_size,
                augment=augment,
                rgb_mean=tuple(self.hparams.rgb_mean),
                rgb_std=tuple(self.hparams.rgb_std),
                nir_mean=float(self.hparams.nir_mean),
                nir_std=float(self.hparams.nir_std),
                height_mean=float(self.hparams.height_mean),
                height_std=float(self.hparams.height_std),
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
    _ = BerlinBiotopeDataModule()
