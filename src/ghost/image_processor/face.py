import os.path as op
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import ArcFaceONNX
from insightface.utils import face_align
from onnxruntime import InferenceSession
import torch

FILE_DIR = op.dirname(__file__)


@dataclass
class FaceAttributes:
    """
    Dataclass for face information.

    Attributes:
    ----------
        - is_female (bool): Whether the face is female.
        - is_male(bool): Whether the face is male.
        - age (int): Age of the face.
        - bbox (np.ndarray): Bounding box of the face with shape (4,).
        - keypoint (np.ndarray): Keypoints of the face with shape (5, 2).
            Keypoints are in the following order:
                1. Outer left eye corner (35)
                2. Outer right eye corner (93)
                3. Nose tip (86)
                4. Mouth left corner (52)
                5. Mouth right corner (61)
        - landmark (np.ndarray): Landmarks of the face with shape (106, 2).
            See more about the landmarks here: https://github.com/nttstar/insightface-resources/blob/master/alignment/images/2d106markup.jpg
        - confidence (float): Confidence score of the face detection between 0 and 1.
    """

    is_female: bool
    age: int
    bbox: np.ndarray
    keypoint: np.ndarray
    landmark: np.ndarray
    confidence: float

    @property
    def is_male(self) -> bool:
        return not self.is_female

    def __str__(self) -> str:
        gender = "female" if self.is_female else "male"
        return f"FaceInfo(gender={gender}, age{self.age}, confidence={self.confidence:.2f})"


class FaceProcessor:
    def __init__(
        self,
        arcface_weight_path: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the FaceProcessor class."""
        self.detector = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106", "genderage"],
            providers=providers,
        )
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

        if arcface_weight_path is None:
            arcface_weight_path = op.join(FILE_DIR, "../../../weights/arcface.onnx")
        session = InferenceSession(arcface_weight_path, providers=providers)
        self.embedder = ArcFaceONNX(arcface_weight_path, session=session)

    def detect_faces(self, image: np.ndarray) -> list[FaceAttributes]:
        """Detects faces in an image."""
        # Detect faces in the image
        faces = self.detector.get(image)
        # Return None if no faces are detected
        if faces is None:
            return
        # Convert the detected faces to Face objects
        faces = [
            FaceAttributes(
                is_female=face.gender,
                age=face.age,
                bbox=face.bbox,
                keypoint=face.kps,
                landmark=face.landmark_2d_106,
                confidence=face.det_score,
            )
            for face in faces
        ]
        return faces

    def extract_faces(
        self,
        image: np.ndarray,
        face_attributes: Sequence[FaceAttributes],
        face_size: int = 256,
    ) -> list[np.ndarray]:
        """
        Extracts faces from an image with the given Face objects.

        Parameters:
        -----------
            - image (np.ndarray): The input image.
            - faces (Sequence[FaceInfo]): The list of Face objects.
            - face_size (int): The size of the extracted face. Must be a multiple of 112 or 128.
        """
        # Check if the face_size is a multiple of 128 or 112
        if face_size % 128 != 0 and face_size % 112 != 0:
            raise ValueError(
                f"face_size must be a multiple of 128 or 112. Got: {face_size}"
            )
        # Extract faces from the image
        faces = [
            face_align.norm_crop(image, attr.keypoint, face_size)
            for attr in face_attributes
        ]
        return faces

    def embed_faces(self, faces: Sequence[np.ndarray]) -> list[torch.Tensor]:
        """Embeds the faces using the ArcFace model."""
        embeddings = [self.embedder.get_feat(face) for face in faces]
        return embeddings

    def inverse_transform_faces(
        self, frames: Sequence[np.ndarray], faces: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        """Inverse transforms the faces to the original image."""
        return [
            face_align.norm_crop_inverse(frame, face)
            for frame, face in zip(frames, faces)
        ]
