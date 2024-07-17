from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import transforms

from ghost.dataset import GhostDataModule
from ghost.network import Ghost


def main(args):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ]
    )

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
        monitor="val_loss",
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


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, description="Train Ghost model"
    )
    parser.add_argument(
        "train_image_dir",
        type=str,
        help="Directory containing training images",
    )
    parser.add_argument(
        "val_image_dir",
        type=str,
        help="Directory containing validation images",
    )
    parser.add_argument(
        "arcface_ckpt_path",
        type=str,
        help="Path to ArcFace checkpoint",
    )
    parser.add_argument(
        "--arcface_vector_size",
        type=int,
        default=256,
        help="Size of ArcFace output vector",
    )
    parser.add_argument(
        "--learning_rate_E_G",
        type=float,
        default=0.0004,
        help="Learning rate for encoder and generator",
    )
    parser.add_argument(
        "--learning_rate_D",
        type=float,
        default=0.0004,
        help="Learning rate for discriminator",
    )
    parser.add_argument(
        "--eye_penalty_weight",
        type=float,
        default=0.0,
        help="Weight for eye loss",
    )
    parser.add_argument(
        "--input_nc",
        type=int,
        default=3,
        help="Number of input channels",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        help="Save top k checkpoints",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="ddp",
        help="Training accelerator",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Validation interval",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )

    args = parser.parse_args()

    main(args)
