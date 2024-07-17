from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.optim import Optimizer
from torchvision.models import resnet101

from ..loss import AEILoss, EyeLoss, GANLoss
from .AADGenerator import ADDGenerator
from .MultiLevelAttributesEncoder import MultilevelAttributesEncoder
from .MultiScaleDiscriminator import MultiscaleDiscriminator


class Ghost(pl.LightningModule):
    def __init__(
        self,
        arcface_ckpt_path: str,
        arcface_vector_size: int = 256,
        learning_rate_E_G: float = 0.0004,
        learning_rate_D: float = 0.0004,
        eye_penalty_weight: float = 0.0,
        input_nc: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.learning_rate_E_G = learning_rate_E_G
        self.learning_rate_D = learning_rate_D

        self.G = ADDGenerator(arcface_vector_size)
        self.E = MultilevelAttributesEncoder()
        self.D = MultiscaleDiscriminator(input_nc=input_nc)

        self.Z = resnet101(num_classes=256)
        self.Z.load_state_dict(torch.load(arcface_ckpt_path, map_location="cpu"))

        self.G.train()
        self.E.train()
        self.D.train()
        self.Z.eval()

        self.gan_loss = GANLoss()
        self.aei_loss = AEILoss()
        self.eye_loss = EyeLoss()

    def embed_face(self, face: torch.Tensor) -> torch.Tensor:
        return self.Z(F.interpolate(face, size=112, mode="bilinear"))

    def forward(
        self, target_img: torch.Tensor, source_img: torch.Tensor
    ) -> torch.Tensor:
        # Embed source image
        z_id = self.embed_face(source_img)
        z_id = F.normalize(z_id)
        z_id = z_id.detach()

        # Encode target image
        feature_map = self.E(target_img)

        # Generate output image
        output = self.G(z_id, feature_map)

        # Embed output image
        output_z_id = self.embed_face(output)
        output_z_id = F.normalize(output_z_id)

        # Encode output image
        output_feature_map = self.E(output)
        return output, z_id, output_z_id, feature_map, output_feature_map

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        target_img, source_img, same = batch

        optimizer_G, optimizer_D = self.optimizers()

        # Train Generator
        self.toggle_optimizer(optimizer_G)
        optimizer_G.zero_grad()

        output, z_id, output_z_id, feature_map, output_feature_map = self(
            target_img, source_img
        )

        self.generated_img = output

        output_multi_scale_val = self.D(output)
        loss_GAN = self.gan_loss(output_multi_scale_val, True, for_discriminator=False)
        loss_E_G, loss_att, loss_id, loss_rec = self.aei_loss(
            target_img,
            output,
            feature_map,
            output_feature_map,
            z_id,
            output_z_id,
            same,
        )

        loss_G = loss_E_G + loss_GAN

        self.manual_backward(loss_G)
        optimizer_G.step()
        self.untoggle_optimizer(optimizer_G)

        self.logger.experiment.add_scalar("Loss G", loss_G.item(), self.global_step)
        self.logger.experiment.add_scalar(
            "Attribute Loss", loss_att.item(), self.global_step
        )
        self.logger.experiment.add_scalar("ID Loss", loss_id.item(), self.global_step)
        self.logger.experiment.add_scalar(
            "Reconstruction Loss", loss_rec.item(), self.global_step
        )
        self.logger.experiment.add_scalar("GAN Loss", loss_GAN.item(), self.global_step)

        # Train Discriminator
        self.toggle_optimizer(optimizer_D)
        optimizer_D.zero_grad()

        multi_scale_val = self.D(target_img)
        output_multi_scale_val = self.D(self.generated_img.detach())

        loss_D_fake = self.gan_loss(multi_scale_val, True)
        loss_D_real = self.gan_loss(output_multi_scale_val, False)

        loss_D = loss_D_fake + loss_D_real

        self.logger.experiment.add_scalar("Loss D", loss_D.item(), self.global_step)

        self.manual_backward(loss_D)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)

        return loss_G

    def validation_step(self, batch, batch_idx):
        target_img, source_img, same = batch

        output, z_id, output_z_id, feature_map, output_feature_map = self(
            target_img, source_img
        )

        self.generated_img = output

        output_multi_scale_val = self.D(output)
        loss_GAN = self.gan_loss(output_multi_scale_val, True, for_discriminator=False)
        loss_E_G, _, _, _ = self.aei_loss(
            target_img, output, feature_map, output_feature_map, z_id, output_z_id, same
        )
        loss_G = loss_E_G + loss_GAN
        return {
            "loss": loss_G,
            "target": target_img[0].cpu(),
            "source": source_img[0].cpu(),
            "output": output[0].cpu(),
        }

    def validation_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        validation_image = []
        for x in outputs:
            validation_image = validation_image + [
                x["target"],
                x["source"],
                x["output"],
            ]
        validation_image = torchvision.utils.make_grid(validation_image, nrow=3)

        self.logger.experiment.add_scalar(
            "Validation Loss", loss.item(), self.global_step
        )
        self.logger.experiment.add_image(
            "Validation Image", validation_image, self.global_step
        )

        return {
            "loss": loss,
            "image": validation_image,
        }

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Any]]:
        optimizer_G = torch.optim.Adam(
            list(self.G.parameters()) + list(self.E.parameters()),
            lr=self.learning_rate_E_G,
            betas=(0, 0.999),
        )
        optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=self.learning_rate_D, betas=(0, 0.999)
        )
        return [optimizer_G, optimizer_D], []
