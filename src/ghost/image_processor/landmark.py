from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance


@dataclass
class Landmark:
    """
    Dataclass for face landmarks.

    Attributes:
    ----------
        - landmarks (np.ndarray): Landmarks of the face with shape (106, 2).
            See more about the landmarks here:


    """

    landmarks: np.ndarray

    def __str__(self) -> str:
        return f"Landmark(landmarks={self.landmarks})"

    def __repr__(self) -> str:
        return str(self)

    def smooth_landmarks(self, keypoints, window=2) -> list:
        """Smoothe the landmarks using moving average."""
        smooth_keypoints = []
        for keypoint in keypoints:
            kps_arr_s = [[keypoint[0]]]
            for i in range(1, len(keypoint)):
                if (len(keypoint[i]) == 0) or (len(keypoint[i - 1]) == 0):
                    kps_arr_s.append([keypoint[i]])
                elif (distance.euclidean(keypoint[i][0], keypoint[i - 1][0]) > 5) or (
                    distance.euclidean(keypoint[i][2], keypoint[i - 1][2]) > 5
                ):
                    kps_arr_s.append([keypoint[i]])
                else:
                    kps_arr_s[-1].append(keypoint[i])

            kps_arr_smooth = []

            for a in kps_arr_s:
                a_smooth = []
                for i in range(len(a)):
                    q = min(i - 0, len(a) - i - 1, window)
                    a_smooth.append(np.mean(np.array(a[i - q : i + 1 + q]), axis=0))

                kps_arr_smooth += a_smooth
            smooth_keypoints.append(kps_arr_smooth)
        return smooth_keypoints
