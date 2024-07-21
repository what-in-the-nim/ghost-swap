from typing import Callable, List

import cv2
import numpy as np
import torch
from insightface.utils import face_align

from .face_detector import FaceDetector
from .masks import face_mask_static


def crop_face(image_full: np.ndarray, app: Callable) -> np.ndarray:
    """
    Crop face from image and resize
    """
    kps = app.get(image_full, 224)
    M, _ = face_align.estimate_norm(kps[0], 224, mode="None")
    align_img = cv2.warpAffine(image_full, M, (224, 224), borderValue=0.0)
    return [align_img]


def normalize_and_torch(image: np.ndarray) -> torch.Tensor:
    """
    Normalize image and transform to torch
    """
    image = torch.tensor(image.copy(), dtype=torch.float32)
    if image.max() > 1.0:
        image = image / 255.0

    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


def normalize_and_torch_batch(frames: np.ndarray) -> torch.Tensor:
    """
    Normalize batch images and transform to torch
    """
    batch_frames = torch.from_numpy(frames.copy())
    if batch_frames.max() > 1.0:
        batch_frames = batch_frames / 255.0

    batch_frames = batch_frames.permute(0, 3, 1, 2)
    batch_frames = (batch_frames - 0.5) / 0.5

    return batch_frames


def get_final_image(
    face_detector: FaceDetector,
    final_frames: List[np.ndarray],
    crop_frames: List[np.ndarray],
    full_frame: np.ndarray,
    tfm_arrays: List[np.ndarray],
) -> None:
    """
    Create final video from frames
    """
    final = full_frame.copy()
    params = [None for i in range(len(final_frames))]

    for i in range(len(final_frames)):
        frame = cv2.resize(final_frames[i][0], (224, 224))

        landmarks = face_detector.get_landmarks(frame)
        landmarks_tgt = face_detector.get_landmarks(crop_frames[i][0])

        mask, _ = face_mask_static(
            crop_frames[i][0], landmarks[0], landmarks_tgt[0], params[i]
        )
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])

        swap_t = cv2.warpAffine(
            frame,
            mat_rev,
            (full_frame.shape[1], full_frame.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask_t = cv2.warpAffine(
            mask, mat_rev, (full_frame.shape[1], full_frame.shape[0])
        )
        mask_t = np.expand_dims(mask_t, 2)

        final = mask_t * swap_t + (1 - mask_t) * final
    final = np.array(final, dtype="uint8")
    return final
def smooth_landmarks(kps_arr, n = 2):
    kps_arr_smooth_final = []
    for ka in kps_arr:
        kps_arr_s = [[ka[0]]]
        for i in range(1, len(ka)):
            if (len(ka[i])==0) or (len(ka[i-1])==0):
                kps_arr_s.append([ka[i]])
            elif (distance.euclidean(ka[i][0], ka[i-1][0]) > 5) or (distance.euclidean(ka[i][2], ka[i-1][2]) > 5):
                kps_arr_s.append([ka[i]])
            else:
                kps_arr_s[-1].append(ka[i])

        kps_arr_smooth = []

        for a in kps_arr_s:
            a_smooth = []
            for i in range(len(a)):
                q = min(i-0, len(a)-i-1, n)      
                a_smooth.append(np.mean( np.array(a[i-q:i+1+q]), axis=0 ) )

            kps_arr_smooth += a_smooth
        kps_arr_smooth_final.append(kps_arr_smooth)      
    return kps_arr_smooth_final