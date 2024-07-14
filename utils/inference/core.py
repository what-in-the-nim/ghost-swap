from typing import Any, Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .image_processing import normalize_and_torch, normalize_and_torch_batch
from .video_processing import crop_frames_and_get_transforms, resize_frames

def model_inference(
    full_frames: list[np.ndarray],
    source_images: list[np.ndarray],
    target_images: list[np.ndarray],
    netArc: Callable,
    G: Callable,
    app: Callable,
    set_target: bool,
    similarity_th=0.15,
    crop_size=224,
    batch_size=60,
    half=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Using original frames get faceswaped frames and transofrmations
    """
    # Get Arcface embeddings of target image
    target_norm = normalize_and_torch_batch(np.array(target_images))
    target_embeds = netArc(
        F.interpolate(
            target_norm, scale_factor=0.5, mode="bilinear", align_corners=True
        )
    )

    # Get the cropped faces from original frames and transformations to get those crops
    crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(
        full_frames,
        target_embeds,
        app,
        netArc,
        crop_size,
        set_target,
        similarity_th=similarity_th,
    )

    # Normalize source images and transform to torch and get Arcface embeddings
    source_embeds = []
    for source_curr in source_images:
        source_curr = normalize_and_torch(source_curr)
        source_embeds.append(
            netArc(
                F.interpolate(
                    source_curr, scale_factor=0.5, mode="bilinear", align_corners=True
                )
            )
        )

    final_frames_list = []
    for crop_frames, source_embed in zip(crop_frames_list, source_embeds):
        # Resize croped frames and get vector which shows on which frames there were faces
        resized_frs, present = resize_frames(crop_frames)
        resized_frs = np.array(resized_frs)

        # transform embeds of Xs and target frames to use by model
        target_batch_rs = torch.from_numpy(resized_frs.copy())
        target_batch_rs = target_batch_rs[:, :, :, [2, 1, 0]] / 255.0

        if half:
            target_batch_rs = target_batch_rs.half()

        target_batch_rs = (target_batch_rs - 0.5) / 0.5  # normalize
        target_batch_rs = target_batch_rs.permute(0, 3, 1, 2)

        if half:
            source_embed = source_embed.half()

        # run model
        size = target_batch_rs.shape[0]
        model_output = []

        for i in tqdm(range(0, size, batch_size), desc="Inference batch size"):
            bs = target_batch_rs[i : i + batch_size].shape[0]

            if bs > 1:
                source_embed = torch.cat([source_embed] * bs)

            with torch.no_grad():
                Y_st, _ = G(target_batch_rs[i : i + batch_size], source_embed)
                Y_st = (Y_st.permute(0, 2, 3, 1) * 0.5 + 0.5) * 255
                Y_st = Y_st[:, :, :, [2, 1, 0]].type(torch.uint8)
                Y_st = Y_st.cpu().detach().numpy()
            model_output.append(Y_st)
        # torch.cuda.empty_cache()
        model_output = np.concatenate(model_output)

        # create list of final frames with transformed faces
        final_frames = []
        idx_fs = 0

        for pres in tqdm(present, desc="Final frames"):
            if pres == 1:
                final_frames.append(model_output[idx_fs])
                idx_fs += 1
            else:
                final_frames.append([])
        final_frames_list.append(final_frames)

    return final_frames_list, crop_frames_list, full_frames, tfm_array_list
