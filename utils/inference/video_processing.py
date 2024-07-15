import os
import traceback
from typing import Any, Callable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from insightface.utils import face_align
from PIL import Image
from scipy.spatial import distance
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .face_detector import FaceDetector
from .image_processing import normalize_and_torch_batch
from .masks import face_mask_static


def add_audio_from_another_video(
    video_with_sound: str,
    video_without_sound: str,
    audio_name: str = "audio",
    fast_cpu=True,
    gpu=False,
) -> None:

    if not os.path.exists("./examples/audio/"):
        os.makedirs("./examples/audio/")
    fast_cmd = "-c:v libx264 -preset ultrafast -crf 18" if fast_cpu else ""
    gpu_cmd = "-c:v h264_nvenc" if gpu else ""
    os.system(
        f"ffmpeg -v -8 -i {video_with_sound} -vn -vcodec h264_nvenc ./examples/audio/{audio_name}.m4a"
    )
    os.system(
        f"ffmpeg -v -8 -i {video_without_sound} -i ./examples/audio/{audio_name}.m4a {fast_cmd} {gpu_cmd}{video_without_sound[:-4]}_audio.mp4 -y"
    )
    os.system(f"rm -rf ./examples/audio/{audio_name}.m4a")
    os.system(f"mv {video_without_sound[:-4]}_audio.mp4 {video_without_sound}")


def read_video(path: str) -> tuple[list[np.ndarray], float]:
    """Read video into frames and get fps"""
    # Create the capture object.
    cap = cv2.VideoCapture(path)
    # Raise an error if we failed to open the video file.
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {path}")
    # Get the frames per second and the total number of frames.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get frame by frame.
    frames = []
    for _ in range(frames_count):
        ret, frame = cap.read()
        # Stop if video is over.
        if not ret:
            break
        frames.append(frame)

    # Release the capture object.
    cap.release()

    return frames, fps


def smooth_landmarks(kps_arr, n=2):
    kps_arr_smooth_final = []
    for ka in kps_arr:
        kps_arr_s = [[ka[0]]]
        for i in range(1, len(ka)):
            if (len(ka[i]) == 0) or (len(ka[i - 1]) == 0):
                kps_arr_s.append([ka[i]])
            elif (distance.euclidean(ka[i][0], ka[i - 1][0]) > 5) or (
                distance.euclidean(ka[i][2], ka[i - 1][2]) > 5
            ):
                kps_arr_s.append([ka[i]])
            else:
                kps_arr_s[-1].append(ka[i])

        kps_arr_smooth = []

        for a in kps_arr_s:
            a_smooth = []
            for i in range(len(a)):
                q = min(i - 0, len(a) - i - 1, n)
                a_smooth.append(np.mean(np.array(a[i - q : i + 1 + q]), axis=0))

            kps_arr_smooth += a_smooth
        kps_arr_smooth_final.append(kps_arr_smooth)
    return kps_arr_smooth_final


def crop_frames_and_get_transforms(
    full_frames: List[np.ndarray],
    target_embeds: torch.Tensor,
    face_detector: FaceDetector,
    netArc: Callable,
    crop_size: int,
    set_target: bool,
    similarity_th: float,
) -> Tuple[List[Any], List[Any]]:
    """
    Crop faces from frames and get respective tranforms
    """

    crop_frames = [[] for _ in range(target_embeds.shape[0])]
    transform_arrays = [[] for _ in range(target_embeds.shape[0])]
    keypoints_arrays = [[] for _ in range(target_embeds.shape[0])]

    target_embeds = F.normalize(target_embeds)
    for frame in tqdm(full_frames, desc="Finding face in frames"):
        try:
            landmarks = face_detector.get_landmarks(frame)
            kps = face_detector.get_keypoints(landmarks)
            if len(landmarks) > 1 or set_target:
                faces = []
                for landmark in landmarks:
                    align_face = face_detector.align(frame, landmark=landmark)
                    faces.append(align_face)

                face_norm = normalize_and_torch_batch(np.array(faces))
                face_norm.to(target_embeds.device)
                face_norm = F.interpolate(
                    face_norm, scale_factor=0.5, mode="bilinear", align_corners=True
                )
                face_embeds = netArc(face_norm)
                face_embeds = F.normalize(face_embeds)

                # Find the best face that matches the target
                similarity = face_embeds @ target_embeds.T
                best_idxs = similarity.argmax(0).detach().cpu().numpy()
                for idx, best_idx in enumerate(best_idxs):

                    if similarity[best_idx][idx] > similarity_th:
                        keypoints_arrays[idx].append(kps[best_idx])
                    else:
                        keypoints_arrays[idx].append([])

            else:
                keypoints_arrays[0].append(kps[0])

        except TypeError as e:
            print("Error in frame with error", e)
            for q in range(len(target_embeds)):
                keypoints_arrays[0].append([])

    smooth_kps = smooth_landmarks(keypoints_arrays, n=2)

    for i, frame in tqdm(enumerate(full_frames)):
        for q in range(len(target_embeds)):
            try:
                M, _ = face_align.estimate_norm(
                    smooth_kps[q][i], crop_size, mode="None"
                )
                align_img = cv2.warpAffine(
                    frame, M, (crop_size, crop_size), borderValue=0.0
                )
                crop_frames[q].append(align_img)
                transform_arrays[q].append(M)
            except Exception as e:
                print("Error in frame", i, "and face", q, "with error", e)
                crop_frames[q].append([])
                transform_arrays[q].append([])

    return crop_frames, transform_arrays


def resize_frames(
    frames: list[np.ndarray], new_size: tuple[int, int] = (256, 256)
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Resize frames to new size

    Args:
        frames: list of frames. Frame shape is (H, W, C)
        new_size: tuple[int, int]
    """
    resized_frames: list[np.ndarray] = []
    have_face = np.ones(len(frames))

    for frame_idx, frame in tqdm(
        enumerate(frames), total=len(frames), desc="Resizing frames"
    ):
        if isinstance(frame, list):
            print(f"Frame {frame_idx} contains no face")
            have_face[frame_idx] = 0
        else:
            resized_frame = cv2.resize(frame, new_size)
            resized_frames.append(resized_frame)

    return resized_frames, have_face


def get_final_video(
    face_detector: FaceDetector,
    final_frames: List[np.ndarray],
    crop_frames: List[np.ndarray],
    full_frames: List[np.ndarray],
    tfm_array: List[np.ndarray],
    output_name: str,
    fps: float,
) -> None:
    """
    Create final video from frames
    """
    # Get output frame size.
    sample_frame = full_frames[0]
    frame_size = sample_frame.shape[1], sample_frame.shape[0]
    # Define the codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Create the video writer object.
    writer = cv2.VideoWriter(
        filename=output_name,
        fourcc=fourcc,
        fps=fps,
        frameSize=frame_size,
    )

    params = [None] * len(crop_frames)
    result_frames = full_frames.copy()
    # debug_dir = Path("debug")
    # debug_mask_dir = debug_dir / "mask"
    # debug_swap_dir = debug_dir / "swap"
    # debug_full_dir = debug_dir / "full"
    # debug_final_dir = debug_dir / "final"
    # debug_mask_dir.mkdir(parents=True, exist_ok=True)
    # debug_swap_dir.mkdir(parents=True, exist_ok=True)
    # debug_full_dir.mkdir(parents=True, exist_ok=True)
    # debug_final_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(full_frames)), desc="Writing video"):
        for j in range(len(crop_frames)):
            try:
                swap = cv2.resize(final_frames[j][i], (224, 224))

                if len(crop_frames[j][i]) == 0:
                    params[j] = None
                    continue

                landmarks = face_detector.get_landmarks(swap)
                landmarks_tgt = face_detector.get_landmarks(crop_frames[j][i])

                if params[j] == None:
                    mask, params[j] = face_mask_static(
                        swap, landmarks[0], landmarks_tgt[0], params[j]
                    )
                else:
                    mask = face_mask_static(
                        swap, landmarks[0], landmarks_tgt[0], params[j]
                    )

                # Save image for debugging
                # cv2.imwrite(
                #     str(debug_mask_dir / f"mask_{i}_{j}.png"),
                #     (mask * 255).astype(np.uint8),
                # )
                # cv2.imwrite(
                #     str(debug_swap_dir / f"swap_{i}_{j}.png"),
                #     (swap * 255).astype(np.uint8),
                # )
                # cv2.imwrite(
                #     str(debug_full_dir / f"full_{i}_{j}.png"),
                #     (full_frames[i] * 255).astype(np.uint8),
                # )
                # print("swap", swap.shape, np.min(swap), np.max(swap))
                # print("mask", mask.shape, np.min(mask), np.max(mask))
                # print("full_frame", full_frames[i].shape, np.min(full_frames[i]), np.max(full_frames[i]))

                # Read inputs
                swap = swap.astype(np.float32)
                mask = mask.astype(np.float32)
                full_frame = result_frames[i].astype(np.float32)
                mat = tfm_array[j][i]

                # Invert the affine transformation matrix
                mat_rev = cv2.invertAffineTransform(mat)

                # Apply affine transformation using OpenCV
                swap_t = cv2.warpAffine(
                    swap, mat_rev, frame_size, flags=cv2.INTER_LINEAR
                )
                mask_t = cv2.warpAffine(
                    mask, mat_rev, frame_size, flags=cv2.INTER_LINEAR
                )

                # Ensure mask_t has the same number of channels as swap_t
                mask_t = mask_t[:, :, np.newaxis]

                # Final composition
                final = (mask_t * swap_t + (1 - mask_t) * full_frame).astype(np.uint8)

                # # Save image for debugging
                # cv2.imwrite(
                #     str(debug_final_dir / f"final_{i}_{j}.png"),
                #     (final * 255).astype(np.uint8),
                # )

                result_frames[i] = final

            except Exception as e:
                print("Error in frame", i, "and face", j)
                print(traceback.format_exc())
                continue

        writer.write(result_frames[i])

    writer.release()


class Frames(Dataset):
    def __init__(self, frames_list):
        self.frames_list = frames_list

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        frame = Image.fromarray(self.frames_list[idx][:, :, ::-1])

        return self.transforms(frame)

    def __len__(self):
        return len(self.frames_list)


def face_enhancement(final_frames: List[np.ndarray], model) -> List[np.ndarray]:
    enhanced_frames_all = []
    for i in range(len(final_frames)):
        enhanced_frames = final_frames[i].copy()
        face_idx = [i for i, x in enumerate(final_frames[i]) if not isinstance(x, list)]
        face_frames = [
            x for i, x in enumerate(final_frames[i]) if not isinstance(x, list)
        ]
        ff_i = 0

        dataset = Frames(face_frames)
        dataloader = DataLoader(
            dataset, batch_size=20, shuffle=False, num_workers=1, drop_last=False
        )

        for data in tqdm(dataloader):
            frames = data
            data = {"image": frames, "label": frames}
            generated = model(data, mode="inference2")
            generated = torch.clamp(generated * 255, 0, 255)
            generated = (
                (generated).type(torch.uint8).permute(0, 2, 3, 1).cpu().detach().numpy()
            )
            for generated_frame in generated:
                enhanced_frames[face_idx[ff_i]] = generated_frame[:, :, ::-1]
                ff_i += 1
        enhanced_frames_all.append(enhanced_frames)

    return enhanced_frames_all
