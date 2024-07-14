import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from utils.inference.face_detector import FaceDetector


def process_image(
    image_dir: str,
    output_dir: Optional[str],
    crop_size: int,
    confidence: float = 0.7,
    device: str = "cpu",
) -> None:
    detector = FaceDetector(device=device)

    image_dir: Path = Path(image_dir)

    if output_dir is None:
        output_dir = image_dir / "faces"
        output_dir.mkdir(exist_ok=True)

    image_paths = list(image_dir.glob("*.jpg"))
    for path in tqdm(image_paths, desc="Processing images"):
        # Load image
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_idx = 0

        landmarks, landmarks_score, _ = detector.get_landmarks(
            image, return_landmark_score=True
        )
        if landmarks is None:
            continue

        for landmark, score in zip(landmarks, landmarks_score):
            if np.average(score) < confidence:
                continue

            face = detector.align(image, landmark, crop_size)
            save_filename = path.stem + f"_{face_idx}.jpg"
            save_path = output_dir / save_filename
            cv2.imwrite(
                str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save cropped faces",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
        help="Size of cropped face",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for face detection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run face detection on",
    )

    args = parser.parse_args()

    process_image(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        confidence=args.confidence,
        device=args.device,
    )
