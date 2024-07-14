import os
import time
from typing import Literal, Sequence

import cv2
import numpy as np
import torch

from arcface_model.iresnet import IResNet, iresnet100
from models.config_sr import TestOptions
from models.pix2pix_model import Pix2PixModel
from network.AEI_Net import AEI_Net
from utils.inference.core import model_inference
from utils.inference.face_detector import FaceDetector
from utils.inference.image_processing import get_final_image
from utils.inference.video_processing import (
    add_audio_from_another_video,
    face_enhancement,
    get_final_video,
    read_video,
)


def load_aei_net(
    G_path: str = "weights/G_unet_2blocks.pth",
    backbone: str = "unet",
    num_blocks: int = 2,
) -> AEI_Net:
    G = AEI_Net(backbone, num_blocks=num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(G_path, map_location=torch.device("cpu")))
    return G


def load_arcface_model(path: str = "arcface_model/backbone.pth") -> IResNet:
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    netArc.eval()
    return netArc


def find_first_face_in_frames(
    face_detector: FaceDetector, frames: Sequence[np.ndarray]
) -> np.ndarray | None:
    """Find the first face in a list of frames."""
    for frame in frames:
        landmarks = face_detector.get_landmarks(frame)
        if len(landmarks) > 0:
            # Face(s) found in frame.
            if len(landmarks) > 1:
                print("Multiple faces found in the input frames, using the first face")
            landmark = landmarks[0]
            face = face_detector.align(frame, landmark=landmark)
            return face
    return None


def inference(
    source_paths: Sequence[str],
    target_paths: Sequence[str],
    frames: Sequence[np.ndarray],
    batch_size: int,
    apply_super_resolution: bool,
    similarity_threshold: float,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Check if all source images are valid.
    invalid_paths = []
    for source_path in source_paths:
        if not os.path.exists(source_path):
            invalid_paths.append(source_path)
    # Raise error if all source images are invalid.
    if len(invalid_paths) == len(source_paths):
        invalid_paths_string = "\n- ".join(source_paths)
        raise FileNotFoundError(
            f"None of the source images exist. Invalid paths:\n- {invalid_paths_string}"
        )
    elif invalid_paths:
        invalid_paths_string = "\n- ".join(invalid_paths)
        print(
            f"Some source images do not exist. Invalid paths:\n- {invalid_paths_string}"
        )
    else:
        print(f"Source images are valid. Found: {len(source_paths)} images")

    # Initialize face detector
    face_detector = FaceDetector(device=device)

    ######################
    # Load source images #
    ######################

    source_images: list[np.ndarray] = []
    for source_path in source_paths:
        # Open image.
        image = cv2.imread(source_path)
        # Warn if image is invalid.
        if image is None:
            print(f"Bad source image: {source_path}")
            continue
        # Convert image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Align face in image.
        landmarks = face_detector.get_landmarks(image)
        if len(landmarks) == 0:
            print(f"No face found in {source_path}")
            continue
        elif len(landmarks) > 1:
            print(f"Multiple faces found in {source_path}, using the first face")
        aligned_image = face_detector.align(image, landmark=landmarks[0])
        source_images.append(aligned_image)

    # Raise error if no valid source images are found.
    if not source_images:
        raise ValueError("None of the source images have a face or are valid")
    else:
        print("Source images loaded")

    ######################
    # Load target images #
    ######################

    # Load target images if provided.
    target_images: list[np.ndarray] = []
    auto_target = (
        False  # Flag to indicate that the target face is automatically selected.
    )
    if target_paths:
        for target_path in target_paths:
            # Open image.
            image = cv2.imread(target_path)
            if image is None:
                print(f"Bad target image: {target_path}")
                continue
            # Convert image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Align face in image.
            landmarks = face_detector.get_landmarks(image)
            if len(landmarks) == 0:
                print(f"No face found in {target_path}")
                continue
            elif len(landmarks) > 1:
                print(f"Multiple faces found in {target_path}, using the first face")
            aligned_image = face_detector.align(image, landmark=landmarks[0])
            target_images.append(target_images)

        # Raise error if no valid target images are found.
        if not target_images:
            raise ValueError("None of the target images have a face or are valid")
        else:
            print("Target images loaded")

    # Automatically find the first face in the target video as the target if no target images are provided.
    else:
        auto_target = True
        target = find_first_face_in_frames(face_detector, frames)
        # Raise error if no face is found in the target video.
        if target is None:
            raise ValueError("No face found in the input frames")
        target_images.append(target)
        print("Target face found in the input frames")

    ###############
    # Inferencing #
    ###############

    arcface_net = load_arcface_model()
    aei_net = load_aei_net()

    final_frames, crop_frames, full_frames, transform_arrays = model_inference(
        frames,
        source_images,
        target_images,
        arcface_net,
        aei_net,
        face_detector,
        not auto_target,
        similarity_th=similarity_threshold,
        batch_size=batch_size,
    )
    if apply_super_resolution:
        print("Applying super resolution to the final result")
        opt = TestOptions()
        model = Pix2PixModel(opt)
        model.netG.train()
        final_frames = face_enhancement(final_frames, model)

    return final_frames, crop_frames, full_frames, transform_arrays


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description="Swap faces from source image to target image or video",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--source_paths",
        required=True,
        nargs="+",
        help="Path to the source image(s) to swap the face from",
    )
    parser.add_argument(
        "-r",
        "--reference_target_paths",
        default=None,
        nargs="+",
        help="Path to the reference target image(s) to detect which face to place the source face on",
    )
    parser.add_argument(
        "--batch_size", default=40, type=int, help="Batch size for inference"
    )
    parser.add_argument(
        "-s",
        "--similarity_threshold",
        default=0.15,
        type=float,
        help="Threshold for selecting a face similar to the target",
    )
    parser.add_argument(
        "-sr",
        "--super_resolution",
        action="store_true",
        help="Apply super resolution to the final result",
    )
    parser.add_argument(
        "-d", "--device", default="cpu", type=str, help="Device to run the model on"
    )

    # Add subparsers
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Subparser for image
    image_parser = subparsers.add_parser("image", help="Swap faces to an image")
    image_parser.add_argument(
        "-t",
        "--target_image",
        type=str,
        required=True,
        help="Path to the target image to swap the face to",
    )
    image_parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default="output.jpg",
        help="Path to save the result image",
    )

    # Subparser for video
    video_parser = subparsers.add_parser("video", help="Swap faces to a video")
    video_parser.add_argument(
        "-t",
        "--target_video",
        type=str,
        required=True,
        help="Path to the target video to swap the face to",
    )
    video_parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default="output.mp4",
        help="Path to save the result video",
    )

    args = parser.parse_args()

    start = time.perf_counter()

    # Load the target source
    if args.command == "image":
        # Load the target image
        target_image = cv2.imread(args.target_image)
        # Raise error if the target image is invalid.
        if target_image is None:
            raise FileNotFoundError(f"Invalid target image path: {args.target_image}")
        # Convert image to RGB
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        frames = [target_image]
    elif args.command == "video":
        frames, fps = read_video(args.target_video)

    # Run the inference
    final_frames, crop_frames, full_frames, transform_arrays = inference(
        source_paths=args.source_paths,
        target_paths=args.reference_target_paths,
        frames=frames,
        batch_size=args.batch_size,
        apply_super_resolution=args.super_resolution,
        similarity_threshold=args.similarity_threshold,
        device=args.device,
    )

    # Apply the result back to the video or image
    if args.command == "image":
        image = full_frames[0]
        swapped_image = get_final_image(
            final_frames, crop_frames, image, transform_arrays
        )
        swapped_image = cv2.cvtColor(swapped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_name, swapped_image)

    elif args.command == "video":
        get_final_video(
            final_frames,
            crop_frames,
            full_frames,
            transform_arrays,
            args.output_name,
            fps,
        )

        add_audio_from_another_video(
            video_with_sound=args.target_video,
            video_without_sound=args.output_name,
        )
        print(f"Video saved with path {args.output_name}")

    stop = time.perf_counter()

    print("Total time: ", stop - start)
