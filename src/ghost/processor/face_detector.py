import os.path as op
from dataclasses import dataclass

import numpy as np
from typing import Optional, Sequence

from .base import BaseProcessor

FILE_DIR = op.dirname(__file__)


@dataclass
class FaceAttributes:
    """
    Dataclass for face attributes.

    Attributes:
    ----------
        - bbox (np.ndarray): Bounding box of the face with shape (4,).
        - keypoint (np.ndarray): Keypoints of the face with shape (5, 2).
            Keypoints are in the following order:
                1. Outer left eye corner (35)
                2. Outer right eye corner (93)
                3. Nose tip (86)
                4. Mouth left corner (52)
                5. Mouth right corner (61)
        - confidence (float): Confidence score of the face detection between 0 and 1.
    """

# - landmark (np.ndarray): Landmarks of the face with shape (106, 2).
#     See more about the landmarks here: https://github.com/nttstar/insightface-resources/blob/master/alignment/images/2d106markup.jpg
    bbox: np.ndarray
    keypoint: np.ndarray
    confidence: float

    def __str__(self) -> str:
        return f"FaceAttributes(bbox={self.bbox}"


class FaceDetector(BaseProcessor):
    def __init__(self, providers: Optional[Sequence[str]] = None) -> None:
        """Initializes the FaceDetector class."""
        super().__init__("detection", providers)

    def detect(self, image: np.ndarray) -> list[FaceAttributes]:
        """Detects faces in an image."""
        # Detect faces in the image
        faces = self.model.get(image)
        # Return None if no faces are detected
        if faces is None:
            return
        # Convert the detected faces to Face objects
        faces = [FaceAttributes(
            bbox=face.bbox,
            keypoint=face.kps,
            confidence=face.det_score,
        ) for face in faces]
        return faces
    
    def crop(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crops the face from the image."""
        # Convert bbox to int
        bbox = bbox.astype(int)
        # Crop the face from the image
        face = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return face 