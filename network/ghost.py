from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from arcface_model import iresnet100
from network import AEI_Net, MultiscaleDiscriminator


class GhostSwap(LightningModule):
    def __init__(
        self,
        arcface_path: str,
        generator_lr: float,
        discriminator_lr: float,
        betas: tuple[float, float] = (0, 0.999),
        weight_decay: float = 1e-4,
        scheduler_step: int = 5000,
        scheduler_gamma: float = 0.2,
    ) -> None:
        """Initialize the model"""
        super().__init__()
        self.save_hyperparameters()
        # Load the face embedder model
        arcface_state_dict = torch.load(arcface_path, map_location=torch.device("cpu"))
        self.face_embedder = iresnet100(fp16=False)
        self.face_embedder.load_state_dict(arcface_state_dict)
        self.face_embedder.eval()

        self.generator = AEI_Net()
        self.discriminator = MultiscaleDiscriminator()

        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

        # Define loss
        self.eye_loss = 

    def forward(self, X: torch.Tensor, z_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network"""
        attributes = self.get_attr(X)
        Y = self.generator(attributes, z_id)
        return Y, attributes
    
    @torch.no_grad()
    def embed_face(self, X: torch.Tensor) -> torch.Tensor:
        """Get the identity embeddings of the input image"""
        return self.face_embedder(F.interpolate(X, [112, 112], mode="bilinear", align_corners=False))
    
    def get_attr(self, X: torch.Tensor) -> torch.Tensor:
        """Get the attributes of the input image"""
        return self.encoder(X)
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step for the model"""
        Xs_orig, Xs, Xt, same_person = batch

        # Get the identity embeddings of Xs
        face_embeddings = self.embed_face(Xs_orig)

        diff_person = torch.ones_like(same_person)

        if args.diff_eq_same:
            same_person = diff_person

        # Training Generator
        optimizer_G.zero_grad()

        Y, Xt_attributes = self.generator(Xt, face_embeddings)
        Di = self.discriminator(Y)
        ZY = self.embed_face(Y)

        if args.eye_detector_loss:
            Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(Xt, model_ft)
            Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, model_ft)
            eye_heatmaps = [
                Xt_heatmap_left,
                Xt_heatmap_right,
                Y_heatmap_left,
                Y_heatmap_right,
            ]
        else:
            eye_heatmaps = None

        lossG, loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes = (
            compute_generator_losses(
                G,
                Y,
                Xt,
                Xt_attributes,
                Di,
                embed,
                ZY,
                eye_heatmaps,
                loss_adv_accumulated,
                diff_person,
                same_person,
                args,
            )
        )

        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward()
        opt_G.step()
        if args.scheduler:
            scheduler_G.step()

        # Training Discriminator
        opt_D.zero_grad()
        lossD = compute_discriminator_loss(D, Y, Xs, diff_person)
        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()

        if (not args.discr_force) or (loss_adv_accumulated < 4.0):
            opt_D.step()
        if args.scheduler:
            scheduler_D.step()

        return lossG


    def configure_optimizers(self) -> tuple[list[Adam], list[StepLR]]:
        """Configure the optimizer and learning rate scheduler"""
        optimizer_G = Adam(
            self.generator.parameters(), lr=self.generator_lr, betas=self.betas, weight_decay=self.weight_decay
        )
        optimizer_D = Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr, betas=self.betas, weight_decay=self.weight_decay
        )
        scheduler_G = StepLR(optimizer_G, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        scheduler_D = StepLR(optimizer_D, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]
    
    def optimizer_zero_grad(self) -> None:
        """Zero the gradients of the optimizer"""
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()


        if iteration % args.show_step == 0:
            images = [Xs, Xt, Y]
            if args.eye_detector_loss:
                Xt_eyes_img = paint_eyes(Xt, Xt_eyes)
                Yt_eyes_img = paint_eyes(Y, Y_eyes)
                images.extend([Xt_eyes_img, Yt_eyes_img])
            image = make_image_list(images)
            if args.use_wandb:
                wandb.log(
                    {
                        "gen_images": wandb.Image(
                            image, caption=f"{epoch:03}" + "_" + f"{iteration:06}"
                        )
                    }
                )
            else:
                cv2.imwrite("./images/generated_image.jpg", image[:, :, ::-1])

        if iteration % 10 == 0:
            print(f"epoch: {epoch}    {iteration} / {len(dataloader)}")
            print(
                f"lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s"
            )
            print(
                f"L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}"
            )
            if args.eye_detector_loss:
                print(f"L_l2_eyes: {L_l2_eyes.item()}")
            print(f"loss_adv_accumulated: {loss_adv_accumulated}")
            if args.scheduler:
                print(
                    f"scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()}"
                )

        if args.use_wandb:
            if args.eye_detector_loss:
                wandb.log({"loss_eyes": L_l2_eyes.item()}, commit=False)
            wandb.log(
                {
                    "loss_id": L_id.item(),
                    "lossD": lossD.item(),
                    "lossG": lossG.item(),
                    "loss_adv": L_adv.item(),
                    "loss_attr": L_attr.item(),
                    "loss_rec": L_rec.item(),
                }
            )

        if iteration % 5000 == 0:
            torch.save(G.state_dict(), f"./saved_models_{args.run_name}/G_latest.pth")
            torch.save(D.state_dict(), f"./saved_models_{args.run_name}/D_latest.pth")

            torch.save(
                G.state_dict(),
                f"./current_models_{args.run_name}/G_"
                + str(epoch)
                + "_"
                + f"{iteration:06}"
                + ".pth",
            )
            torch.save(
                D.state_dict(),
                f"./current_models_{args.run_name}/D_"
                + str(epoch)
                + "_"
                + f"{iteration:06}"
                + ".pth",
            )

        if (iteration % 250 == 0) and (args.use_wandb):
            ### Посмотрим как выглядит свап на трех конкретных фотках, чтобы проследить динамику
            G.eval()

            res1 = get_faceswap(
                "examples/images/training//source1.png",
                "examples/images/training//target1.png",
                G,
                netArc,
                device,
            )
            res2 = get_faceswap(
                "examples/images/training//source2.png",
                "examples/images/training//target2.png",
                G,
                netArc,
                device,
            )
            res3 = get_faceswap(
                "examples/images/training//source3.png",
                "examples/images/training//target3.png",
                G,
                netArc,
                device,
            )

            res4 = get_faceswap(
                "examples/images/training//source4.png",
                "examples/images/training//target4.png",
                G,
                netArc,
                device,
            )
            res5 = get_faceswap(
                "examples/images/training//source5.png",
                "examples/images/training//target5.png",
                G,
                netArc,
                device,
            )
            res6 = get_faceswap(
                "examples/images/training//source6.png",
                "examples/images/training//target6.png",
                G,
                netArc,
                device,
            )

            output1 = np.concatenate((res1, res2, res3), axis=0)
            output2 = np.concatenate((res4, res5, res6), axis=0)

            output = np.concatenate((output1, output2), axis=1)

            wandb.log(
                {
                    "our_images": wandb.Image(
                        output, caption=f"{epoch:03}" + "_" + f"{iteration:06}"
                    )
                }
            )

            G.train()


def train(args, device):
    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch

    # initializing main models
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    D = MultiscaleDiscriminator(
        input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d
    ).to(device)

    if args.eye_detector_loss:
        model_ft = models.FAN(4, "False", "False", 98)
        checkpoint = torch.load("./AdaptiveWingLoss/AWL_detector/WFLW_4HG.pth")
        if "state_dict" not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint["state_dict"]
            model_weights = model_ft.state_dict()
            pretrained_weights = {
                k: v for k, v in pretrained_weights.items() if k in model_weights
            }
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)
        model_ft = model_ft.to(device)
        model_ft.eval()
    else:
        model_ft = None

    if args.pretrained:
        try:
            G.load_state_dict(
                torch.load(args.G_path, map_location=torch.device("cpu")), strict=False
            )
            D.load_state_dict(
                torch.load(args.D_path, map_location=torch.device("cpu")), strict=False
            )
            print("Loaded pretrained weights for G and D")
        except FileNotFoundError as e:
            print(
                "Not found pretrained weights. Continue without any pretrained weights."
            )

    if args.vgg:
        dataset = FaceEmbedVGG2(
            args.dataset_path,
            same_prob=args.same_person,
            same_identity=args.same_identity,
        )
    else:
        dataset = FaceEmbed([args.dataset_path], same_prob=args.same_person)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    # Будем считать аккумулированный adv loss, чтобы обучать дискриминатор только когда он ниже порога, если discr_force=True
    loss_adv_accumulated = 20.0

    for epoch in range(0, max_epoch):
        train_one_epoch(
            G,
            D,
            opt_G,
            opt_D,
            scheduler_G,
            scheduler_D,
            netArc,
            model_ft,
            args,
            dataloader,
            device,
            epoch,
            loss_adv_accumulated,
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("cuda is not available. using cpu. check if it's ok")

    print("Starting traing")
    train(args, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument(
        "--dataset_path",
        default="/VggFace2-crop/",
        help="Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False",
    )
    parser.add_argument(
        "--G_path",
        default="./saved_models/G.pth",
        help="Path to pretrained weights for G. Only used if pretrained=True",
    )
    parser.add_argument(
        "--D_path",
        default="./saved_models/D.pth",
        help="Path to pretrained weights for D. Only used if pretrained=True",
    )
    parser.add_argument(
        "--vgg",
        default=True,
        type=bool,
        help="When using VGG2 dataset (or any other dataset with several photos for one identity)",
    )
    # weights for loss
    parser.add_argument(
        "--weight_adv", default=1, type=float, help="Adversarial Loss weight"
    )
    parser.add_argument(
        "--weight_attr", default=10, type=float, help="Attributes weight"
    )
    parser.add_argument(
        "--weight_id", default=20, type=float, help="Identity Loss weight"
    )
    parser.add_argument(
        "--weight_rec", default=10, type=float, help="Reconstruction Loss weight"
    )
    parser.add_argument(
        "--weight_eyes", default=0.0, type=float, help="Eyes Loss weight"
    )
    # training params you may want to change

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
    parser.add_argument(
        "--same_person",
        default=0.2,
        type=float,
        help="Probability of using same person identity during training",
    )
    parser.add_argument(
        "--same_identity",
        default=True,
        type=bool,
        help="Using simswap approach, when source_id = target_id. Only possible with vgg=True",
    )
    parser.add_argument(
        "--diff_eq_same",
        default=False,
        type=bool,
        help="Don't use info about where is defferent identities",
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        type=bool,
        help="If using the pretrained weights for training or not",
    )
    parser.add_argument(
        "--discr_force",
        default=False,
        type=bool,
        help="If True Discriminator would not train when adversarial loss is high",
    )
    parser.add_argument(
        "--scheduler",
        default=False,
        type=bool,
        help="If True decreasing LR is used for learning of generator and discriminator",
    )
    parser.add_argument("--scheduler_step", default=5000, type=int)
    parser.add_argument(
        "--scheduler_gamma",
        default=0.2,
        type=float,
        help="It is value, which shows how many times to decrease LR",
    )
    parser.add_argument(
        "--eye_detector_loss",
        default=False,
        type=bool,
        help="If True eye loss with using AdaptiveWingLoss detector is applied to generator",
    )
    # info about this run
    parser.add_argument(
        "--use_wandb",
        default=False,
        type=bool,
        help="Use wandb to track your experiments or not",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        type=str,
        help="Name of this run. Used to create folders where to save the weights.",
    )
    parser.add_argument("--wandb_project", default="your-project-name", type=str)
    parser.add_argument("--wandb_entity", default="your-login", type=str)
    # training params you probably don't want to change
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr_G", default=4e-4, type=float)
    parser.add_argument("--lr_D", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=2000, type=int)
    parser.add_argument("--show_step", default=500, type=int)
    parser.add_argument("--save_epoch", default=1, type=int)
    parser.add_argument("--optim_level", default="O2", type=str)

    args = parser.parse_args()

    if args.vgg == False and args.same_identity == True:
        raise ValueError(
            "Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True"
        )

    if args.use_wandb == True:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            settings=wandb.Settings(start_method="fork"),
        )

        config = wandb.config
        config.dataset_path = args.dataset_path
        config.weight_adv = args.weight_adv
        config.weight_attr = args.weight_attr
        config.weight_id = args.weight_id
        config.weight_rec = args.weight_rec
        config.weight_eyes = args.weight_eyes
        config.same_person = args.same_person
        config.Vgg2Face = args.vgg
        config.same_identity = args.same_identity
        config.diff_eq_same = args.diff_eq_same
        config.discr_force = args.discr_force
        config.scheduler = args.scheduler
        config.scheduler_step = args.scheduler_step
        config.scheduler_gamma = args.scheduler_gamma
        config.eye_detector_loss = args.eye_detector_loss
        config.pretrained = args.pretrained
        config.run_name = args.run_name
        config.G_path = args.G_path
        config.D_path = args.D_path
        config.batch_size = args.batch_size
        config.lr_G = args.lr_G
        config.lr_D = args.lr_D
    elif not os.path.exists("./images"):
        os.mkdir("./images")

    # Создаем папки, чтобы было куда сохранять последние веса моделей, а также веса с каждой эпохи
    if not os.path.exists(f"./saved_models_{args.run_name}"):
        os.mkdir(f"./saved_models_{args.run_name}")
        os.mkdir(f"./current_models_{args.run_name}")

    main(args)
