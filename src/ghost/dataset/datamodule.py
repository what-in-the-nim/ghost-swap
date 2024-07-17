from typing import Callable, Optional

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import GhostDataset


class GhostDataModule(LightningDataModule):
    def __init__(
        self,
        train_image_dir: str,
        val_image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        augmentation_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        super().__init__()
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_transform = augmentation_transform

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = GhostDataset(
                self.train_image_dir, self.augmentation_transform
            )
            self.val_dataset = GhostDataset(self.val_image_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
