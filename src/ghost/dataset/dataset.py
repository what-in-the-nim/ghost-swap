import random
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class GhostDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        augmentation_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        p_same_person: float = 0.8,
    ) -> None:
        super().__init__()
        image_dir: Path = Path(image_dir)

        # Find all image files in the directory
        self.files = list(image_dir.glob("*.[jpg,png,jpeg]"))
        self.augmentation_transform = augmentation_transform
        self.p_same_person = p_same_person

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.files)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the image pair and the label."""
        dataset_length = len(self.files)

        # Get the target and source image index
        source_index = index
        if random.random() < self.p_same_person:
            # Get the same person
            target_index = source_index
        else:
            # Get a different person
            target_index = random.randrange(dataset_length)

        # Generate the same label.
        same = torch.ones(1) if target_index == source_index else torch.zeros(1)

        target_image = self.open_image(self.files[target_index])
        source_image = self.open_image(self.files[source_index])

        if self.augmentation_transform is not None:
            target_image = self.augmentation_transform(target_image)
            source_image = self.augmentation_transform(source_image)

        return target_image, source_image, same

    def open_image(self, file: str | Path) -> Image.Image:
        return Image.open(file).convert("RGB")
