import os.path as op
from dataclasses import dataclass

import numpy as np
from typing import Optional, Sequence

from .base import BaseProcessor

import cv2
FILE_DIR = op.dirname(__file__)


@dataclass
class Mask:
    """
    Dataclass for face attributes.

    Attributes:
    ----------
        - landmark (np.ndarray): Landmarks of the face with shape (106, 2).
            See more about the landmarks here: https://github.com/nttstar/insightface-resources/blob/master/alignment/images/2d106markup.jpg
    """
    landmark: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        points = np.array(self.landmark, np.int32)
        convexhull = cv2.convexHull(points)
        return convexhull

class FaceMasker(BaseProcessor):
    def __init__(self, providers: Optional[Sequence[str]] = None) -> None:
        """Initializes the FaceDetector class."""
        super().__init__("landmark", providers)

    def mask(self, image: np.ndarray) -> list[Mask]:
        """Detects faces in an image."""
        # Detect faces in the image
        faces = self.model.get(image)
        # Return None if no faces are detected
        if faces is None:
            return
        # Convert the detected faces to Face objects
        faces = [Mask(
            landmark=face.landmarks,
        ) for face in faces]
        return faces