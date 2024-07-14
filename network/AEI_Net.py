import torch
import torch.nn as nn

from .aad_generator import AADGenerator
from .attribute_encoder import MLAttrEncoder, MLAttrEncoderResnet


class AEI_Net(nn.Module):
    def __init__(self, backbone, num_blocks=2, c_id=256):
        super(AEI_Net, self).__init__()
        if backbone in ["unet", "linknet"]:
            self.encoder = MLAttrEncoder(backbone)
        elif backbone == "resnet":
            self.encoder = MLAttrEncoderResnet()
        self.generator = AADGenerator(backbone, c_id, num_blocks)

    def forward(self, Xt: torch.Tensor, z_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network
        
        Parameters:
        ----------
            Xt: torch.Tensor
                Input image tensor of shape (B, C, H, W)
            z_id: torch.Tensor
                Identity vector from ArcFace model of shape (B, 512)

        """
        attributes = self.get_attr(Xt)
        Y = self.generator(attributes, z_id)
        return Y, attributes

    def get_attr(self, X: torch.Tensor) -> torch.Tensor:
        """Get the attributes of the input image"""
        return self.encoder(X)
