import argparse
import os
import time

import cv2
import torch
from arcface_model.iresnet import iresnet100

from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from models.config_sr import TestOptions
from models.pix2pix_model import Pix2PixModel
from network.AEI_Net import AEI_Net
from utils.inference.core import model_inference
from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import (
    add_audio_from_another_video,
    face_enhancement,
    get_final_video,
    get_target,
    read_video,
)
from typing import Sequence


def init_models(G_path: str, backbone: str, num_blocks: int, use_sr: bool) -> tuple:
    # model for face cropping
    app = Face_detect_crop(name="antelope", root="./insightface_func/models")
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    # main model for generation
    G = AEI_Net(backbone, num_blocks=num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(G_path, map_location=torch.device("cpu")))
    # G = G.cuda()
    G = G.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load("arcface_model/backbone.pth", map_location=torch.device("cpu")))
    # netArc = netArc.cuda()
    netArc.eval()

    # model to get face landmarks
    handler = Handler("./coordinate_reg/model/2d106det", 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
    if use_sr:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        # opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
    else:
        model = None

    return app, G, netArc, handler, model


def main(
    source_paths: Sequence[str],
    target_faces_paths: Sequence[str],
    target_video: str,
    out_video_name: str,
    image_to_image: bool,
    target_image: str,
    out_image_name: str,
    G_path: str,
    backbone: str,
    num_blocks: int,
    batch_size: int,
    crop_size: int,
    use_sr: bool,
    similarity_th: float,
):
    app, G, netArc, handler, model = init_models(G_path, backbone, num_blocks, use_sr)

    # get crops from source images
    print("List of source paths: ", source_paths)
    source = []
    try:
        for source_path in source_paths:
            img = cv2.imread(source_path)
            img = crop_face(img, app, crop_size)[0]
            source.append(img[:, :, ::-1])
    except TypeError:
        print("Bad source images!")
        exit()

    # get full frames from video
    if not image_to_image:
        full_frames, fps = read_video(target_video)
    else:
        target_full = cv2.imread(target_image)
        full_frames = [target_full]

    # get target faces that are used for swap
    set_target = True
    print("List of target paths: ", target_faces_paths)
    if not target_faces_paths:
        target = get_target(full_frames, app, crop_size)
        set_target = False
    else:
        target = []
        try:
            for target_faces_path in target_faces_paths:
                img = cv2.imread(target_faces_path)
                img = crop_face(img, app, crop_size)[0]
                target.append(img)
        except TypeError:
            print("Bad target images!")
            exit()

    start = time.time()
    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
        full_frames,
        source,
        target,
        netArc,
        G,
        app,
        set_target,
        similarity_th=similarity_th,
        crop_size=crop_size,
        BS=batch_size,
    )
    if use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)

    if not image_to_image:
        get_final_video(
            final_frames_list,
            crop_frames_list,
            full_frames,
            tfm_array_list,
            out_video_name,
            fps,
            handler,
        )

        add_audio_from_another_video(target_video, out_video_name, "audio")
        print(f"Video saved with path {out_video_name}")
    else:
        result = get_final_image(
            final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler
        )
        cv2.imwrite(out_image_name, result)
        print(f"Swapped Image saved with path {out_image_name}")

    print("Total time: ", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generator params
    parser.add_argument(
        "--G_path",
        default="weights/G_unet_2blocks.pth",
        type=str,
        help="Path to weights for G",
    )
    parser.add_argument(
        "--backbone",
        default="unet",
        const="unet",
        nargs="?",
        choices=["unet", "linknet", "resnet"],
        help="Backbone for attribute encoder",
    )
    parser.add_argument(
        "--num_blocks", default=2, type=int, help="Numbers of AddBlocks at AddResblock"
    )

    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--crop_size", default=224, type=int, help="Don't change this")
    parser.add_argument(
        "--use_sr",
        default=False,
        type=bool,
        help="True for super resolution on swap images",
    )
    parser.add_argument(
        "--similarity_th",
        default=0.15,
        type=float,
        help="Threshold for selecting a face similar to the target",
    )

    parser.add_argument(
        "--source_paths",
        default=["examples/images/mark.jpg", "examples/images/elon_musk.jpg"],
        nargs="+",
    )
    parser.add_argument(
        "--target_faces_paths",
        default=[],
        nargs="+",
        help="It's necessary to set the face/faces in the video to which the source face/faces is swapped. You can skip this parametr, and then any face is selected in the target video for swap.",
    )

    # parameters for image to video
    parser.add_argument(
        "--target_video",
        default="examples/videos/nggyup.mp4",
        type=str,
        help="It's necessary for image to video swap",
    )
    parser.add_argument(
        "--out_video_name",
        default="examples/results/result.mp4",
        type=str,
        help="It's necessary for image to video swap",
    )

    # parameters for image to image
    parser.add_argument(
        "--image_to_image",
        default=False,
        type=bool,
        help="True for image to image swap, False for swap on video",
    )
    parser.add_argument(
        "--target_image",
        default="examples/images/beckham.jpg",
        type=str,
        help="It's necessary for image to image swap",
    )
    parser.add_argument(
        "--out_image_name",
        default="examples/results/result.png",
        type=str,
        help="It's necessary for image to image swap",
    )

    args = parser.parse_args()
    main(
        source_paths=args.source_paths,
        target_faces_paths=args.target_faces_paths,
        target_video=args.target_video,
        out_video_name=args.out_video_name,
        image_to_image=args.image_to_image,
        target_image=args.target_image,
        out_image_name=args.out_image_name,
        G_path=args.G_path,
        backbone=args.backbone,
        num_blocks=args.num_blocks,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        use_sr=args.use_sr,
        similarity_th=args.similarity_th,
    )