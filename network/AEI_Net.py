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

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        return self.encoder(X)
