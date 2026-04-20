import glob
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


def collect_tile_files(
    data_dir: str,
    tiles: Sequence[str],
    image_ext: str = ".jpg",
    mask_ext: str = ".png",
) -> Tuple[List[str], List[str]]:
    """Collect (image, mask) path pairs from the given tile subdirectories.

    Each tile is expected to contain ``images/`` and ``masks/`` folders whose files share
    the same basename. Pairs are returned in sorted order so that splits are deterministic.
    """
    images: List[str] = []
    masks: List[str] = []
    for tile in tiles:
        tile_dir = os.path.join(data_dir, tile)
        if not os.path.isdir(tile_dir):
            continue
        img_files = sorted(glob.glob(os.path.join(tile_dir, "images", f"*{image_ext}")))
        for img_path in img_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(tile_dir, "masks", f"{base}{mask_ext}")
            if os.path.exists(mask_path):
                images.append(img_path)
                masks.append(mask_path)
    return images, masks


class AerialDataset(Dataset):
    """Dataset for aerial/satellite semantic segmentation.

    Each sample is ``(image, mask)`` where image is a float tensor of shape
    ``[3, H, W]`` normalized by ImageNet statistics and mask is a long tensor
    of shape ``[H, W]`` with per-pixel class indices.
    """

    def __init__(
        self,
        image_files: Sequence[str],
        mask_files: Sequence[str],
        rgb_to_class: Dict[Tuple[int, int, int], int],
        image_size: int = 512,
        augment: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        assert len(image_files) == len(mask_files), "image/mask count mismatch"
        self.image_files = list(image_files)
        self.mask_files = list(mask_files)
        self.rgb_to_class = {tuple(int(c) for c in k): int(v) for k, v in rgb_to_class.items()}
        self.image_size = int(image_size)
        self.augment = bool(augment)

        transforms_list = [v2.Resize((self.image_size, self.image_size))]
        if self.augment:
            transforms_list += [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=10),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        self.transform = v2.Compose(transforms_list)

        self.register_buffer_not_applicable = True  # keep as plain tensors
        self._mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.image_files)

    def _rgb_mask_to_class(self, mask_rgb: np.ndarray) -> np.ndarray:
        h, w = mask_rgb.shape[:2]
        flat = mask_rgb.reshape(-1, 3)
        class_mask = np.zeros(flat.shape[0], dtype=np.int64)
        for rgb, cls in self.rgb_to_class.items():
            matches = np.all(flat == np.asarray(rgb, dtype=np.uint8), axis=1)
            class_mask[matches] = cls
        return class_mask.reshape(h, w)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_pil = Image.open(self.image_files[idx]).convert("RGB")
        mask_rgb = np.array(Image.open(self.mask_files[idx]).convert("RGB"))
        class_mask = self._rgb_mask_to_class(mask_rgb)

        image_np = np.array(image_pil)  # HWC uint8
        image_tv = tv_tensors.Image(torch.from_numpy(image_np).permute(2, 0, 1).contiguous())
        mask_tv = tv_tensors.Mask(torch.from_numpy(class_mask))

        image_tv, mask_tv = self.transform(image_tv, mask_tv)

        image_out = image_tv.to(torch.float32) / 255.0
        image_out = (image_out - self._mean) / self._std

        return image_out, mask_tv.long()


if __name__ == "__main__":
    _ = AerialDataset([], [], {})
