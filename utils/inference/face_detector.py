import cv2
import numpy as np
from face_alignment import FaceAlignment


class FaceDetector(FaceAlignment):
    """
    Subclass of FaceAlignment from face_alignment library

    Implements additional methods for face alignment
    """
    def __init__(self, *args, **kwargs):
        if "face_detector" not in kwargs:
            # Set default face detector
            kwargs["face_detector"] = "blazeface"
        if "landmarks_type" not in kwargs:
            # Set default landmarks type
            kwargs["landmarks_type"] = 1  # 2D landmarks
        super().__init__(*args, **kwargs)

    def align(self, image: np.ndarray, landmarks: np.ndarray, chip_size: int = 224, scaling: float = 0.9) -> np.ndarray:
        """Align face using landmarks"""
        chip_corners = np.float32(
            [[0, 0], [chip_size, 0], [0, chip_size], [chip_size, chip_size]]
        )
        # Compute the Anchor Landmarks
        # This ensures the eyes and chin will not move within the chip
        right_eye_mean = np.mean(landmarks[36:42], axis=0)
        left_eye_mean = np.mean(landmarks[42:47], axis=0)
        middle_eye = (right_eye_mean + left_eye_mean) / 2
        chin = landmarks[8]

        # Compute the chip center and up/side vectors
        mean = middle_eye[:2]

        # Compute the up_vector and right_vector
        up_vector = (chin - middle_eye)[:2] * scaling
        right_vector = np.array([up_vector[1], -up_vector[0]])

        # Compute the corners of the facial chip
        image_corners = np.float32(
            [
                (mean + ((-right_vector - up_vector))),
                (mean + ((right_vector - up_vector))),
                (mean + ((-right_vector + up_vector))),
                (mean + ((right_vector + up_vector))),
            ]
        )

        # Compute the Perspective Homography and Extract the chip from the image
        chipMatrix = cv2.getPerspectiveTransform(image_corners, chip_corners)
        align_image = cv2.warpPerspective(image, chipMatrix, (chip_size, chip_size))
        return align_image
