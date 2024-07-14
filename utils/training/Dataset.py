import random
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import TensorDataset


class FaceEmbed(TensorDataset):
    def __init__(self, image_dir: str, p_return_same_person: float = 0.8) -> None:
        image_dir: Path = Path(image_dir)
        # Check if image dir exists
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory {image_dir} not found.")

        # Recursively get all image paths in the directory.
        self.datasets = list(image_dir.glob("**/*.jpg"))
        self.p_return_same_person = p_return_same_person

        self.transforms_arcface = transforms.Compose(
            [
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.transforms_base = transforms.Compose(
            [
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.datasets)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get the image tensors and whether they are the same person or not"""
        # Get the source image
        source_path = self.datasets[idx]
        # Read the image
        source_image = cv2.imread(str(source_path))
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        source_image = Image.fromarray(source_image)
        # Randomly decide if the target image will be the same person or not.
        same_person = random.random() < self.p_return_same_person
        if same_person:
            # Same person
            target_image = source_image.copy()
        else:
            # Different person
            target_idx = random.randint(0, len(self.datasets) - 1)
            # If accidentally picked the same image, pick the next one.
            if target_idx == idx:
                target_idx = (target_idx + 1) % len(self.datasets)
            # Get the target image
            target_path = self.datasets[target_idx]
            # Read the image
            target_image = cv2.imread(str(target_path))
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_image = Image.fromarray(target_image)

        return (
            self.transforms_arcface(source_image),
            self.transforms_base(source_image),
            self.transforms_base(target_image),
            int(same_person),
        )
