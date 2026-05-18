import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


def collect_biotope_files(
    data_dir: str,
    use_nir: bool = False,
    use_height: bool = False,
) -> Tuple[List[str], Optional[List[str]], Optional[List[str]], List[str]]:
    """Scan data_dir/core/ and return sorted file lists for each requested layer.

    Only tiles where all requested layer files exist are included.
    Returns (rgb_files, nir_files, height_files, label_files) where
    nir_files and height_files are None when the respective layer is not requested.
    """
    core_dir = os.path.join(data_dir, "core")
    rgb_paths = sorted(glob.glob(os.path.join(core_dir, "images", "tile_*_rgb.png")))

    rgb_out: List[str] = []
    nir_out: List[str] = []
    height_out: List[str] = []
    label_out: List[str] = []

    for rgb_path in rgb_paths:
        m = re.match(r"(tile_\d+)_rgb\.png", os.path.basename(rgb_path))
        if not m:
            continue
        tile_id = m.group(1)

        label_path = os.path.join(core_dir, "labels", f"{tile_id}_label.png")
        if not os.path.exists(label_path):
            continue

        nir_path = os.path.join(core_dir, "nir", f"{tile_id}_nir.png")
        if use_nir and not os.path.exists(nir_path):
            continue

        height_path = os.path.join(core_dir, "height", f"{tile_id}_height.npy")
        if use_height and not os.path.exists(height_path):
            continue

        rgb_out.append(rgb_path)
        label_out.append(label_path)
        if use_nir:
            nir_out.append(nir_path)
        if use_height:
            height_out.append(height_path)

    return (
        rgb_out,
        nir_out if use_nir else None,
        height_out if use_height else None,
        label_out,
    )


class BerlinBiotopeDataset(Dataset):
    """Dataset for Berlin biotope semantic segmentation.

    Each sample is ``(image, mask)`` where image is a float32 tensor of shape
    ``[C, H, W]`` (C=3/4/5 depending on enabled layers, normalized) and mask is
    a long tensor of shape ``[H, W]`` with per-pixel class indices 0-12.
    """

    def __init__(
        self,
        rgb_files: List[str],
        label_files: List[str],
        nir_files: Optional[List[str]] = None,
        height_files: Optional[List[str]] = None,
        image_size: int = 512,
        augment: bool = False,
        rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        nir_mean: float = 0.5,
        nir_std: float = 0.25,
        height_mean: float = 5.0,
        height_std: float = 8.0,
    ) -> None:
        super().__init__()
        assert len(rgb_files) == len(label_files), "rgb/label count mismatch"
        if nir_files is not None:
            assert len(nir_files) == len(rgb_files), "nir count mismatch"
        if height_files is not None:
            assert len(height_files) == len(rgb_files), "height count mismatch"

        self.rgb_files = list(rgb_files)
        self.label_files = list(label_files)
        self.nir_files = list(nir_files) if nir_files is not None else None
        self.height_files = list(height_files) if height_files is not None else None
        self.image_size = int(image_size)
        self.augment = bool(augment)

        spatial: list = [v2.Resize((self.image_size, self.image_size))]
        if self.augment:
            spatial += [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=10),
            ]
        self.spatial_transform = v2.Compose(spatial)
        self.color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2)

        self._rgb_mean = torch.tensor(rgb_mean, dtype=torch.float32).view(3, 1, 1)
        self._rgb_std = torch.tensor(rgb_std, dtype=torch.float32).view(3, 1, 1)
        self._nir_mean = torch.tensor([nir_mean], dtype=torch.float32).view(1, 1, 1)
        self._nir_std = torch.tensor([nir_std], dtype=torch.float32).view(1, 1, 1)
        self._height_mean = torch.tensor([height_mean], dtype=torch.float32).view(1, 1, 1)
        self._height_std = torch.tensor([height_std], dtype=torch.float32).view(1, 1, 1)

    def __len__(self) -> int:
        return len(self.rgb_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_np = np.array(Image.open(self.rgb_files[idx]).convert("RGB"), dtype=np.float32)
        label_np = np.array(Image.open(self.label_files[idx]).convert("RGB"), dtype=np.uint8)[:, :, 0]

        # RGB and NIR are scaled to [0,1]; height stays in metres
        channels = [torch.from_numpy(rgb_np).permute(2, 0, 1) / 255.0]  # [3,H,W]

        if self.nir_files is not None:
            nir_np = np.array(Image.open(self.nir_files[idx]).convert("L"), dtype=np.float32)
            channels.append(torch.from_numpy(nir_np).unsqueeze(0) / 255.0)  # [1,H,W]

        if self.height_files is not None:
            h_np = np.clip(np.load(self.height_files[idx]).astype(np.float32), 0.0, None)
            channels.append(torch.from_numpy(h_np).unsqueeze(0))  # [1,H,W] in metres

        image = torch.cat(channels, dim=0)  # [C, H, W]
        mask = torch.from_numpy(label_np.astype(np.int64))  # [H, W]

        # Spatial transforms applied jointly to all channels and the mask
        image_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)
        image_tv, mask_tv = self.spatial_transform(image_tv, mask_tv)
        image = image_tv.as_subclass(torch.Tensor)

        # Color jitter on RGB slice only (values in [0,1])
        if self.augment:
            rgb_jittered = self.color_jitter(tv_tensors.Image(image[:3])).as_subclass(torch.Tensor)
            image = (
                torch.cat([rgb_jittered, image[3:]], dim=0) if image.shape[0] > 3 else rgb_jittered
            )

        # Normalize per channel group
        rgb_norm = (image[:3] - self._rgb_mean) / self._rgb_std
        parts = [rgb_norm]
        offset = 3

        if self.nir_files is not None:
            parts.append((image[offset : offset + 1] - self._nir_mean) / self._nir_std)
            offset += 1

        if self.height_files is not None:
            parts.append((image[offset : offset + 1] - self._height_mean) / self._height_std)

        return torch.cat(parts, dim=0), mask_tv.long()


if __name__ == "__main__":
    _ = BerlinBiotopeDataset([], [])
