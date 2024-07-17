import os.path as op
from argparse import ArgumentParser

from cv2 import transform
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from ghost.dataset import GhostDataModule
from ghost.network import Ghost


def main(args):
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            ])

    model = Ghost(
        arcface_ckpt_path=args.arcface_ckpt_path,
        arcface_vector_size=args.arcface_vector_size,
        learning_rate_E_G=args.learning_rate_E_G,
        learning_rate_D=args.learning_rate_D,
        eye_penalty_weight=args.eye_penalty_weight,
        input_nc=args.input_nc,
    )
    datamodule = GhostDataModule(
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_transform=transform,
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=args.checkpoint_dir,
        monitor='val_loss',
        verbose=True,
        save_top_k=args.save_top_k,
    )

    trainer = Trainer(
        logger=TensorBoardLogger(args.checkpoint_dir),
        checkpoint_callback=checkpoint_callback,
        accelerator=args.accelerator,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_interval,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('-g', '--gpus', type=str, default=None,
                        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the run.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")
    parser.add_argument('-s', '--save_top_k', type=int, default=-1,
                        help="save top k checkpoints, default(-1): save all")
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
                        help="fast run for debugging purpose")
    parser.add_argument('--val_interval', type=float, default=0.01,
                        help="run val loop every * training epochs")

    args = parser.parse_args()

    main(args)