from .aad_res_block import AADResBlock

import torch.nn.functional as F
import torch.nn as nn
import torch

from ...utils import weight_init

class AADGenerator(nn.Module):
    """
    Adaptive Attentional Denormalization Generator

    This generator try to generate new face from the given source identity and target attribute features.
    """
    def __init__(self, backbone, c_id=256, num_blocks: int = 2) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AADResBlock(1024, 1024, 1024, c_id, num_blocks)
        if backbone == "linknet":
            self.AADBlk2 = AADResBlock(1024, 1024, 1024, c_id, num_blocks)
            self.AADBlk3 = AADResBlock(1024, 1024, 512, c_id, num_blocks)
            self.AADBlk4 = AADResBlock(1024, 512, 256, c_id, num_blocks)
            self.AADBlk5 = AADResBlock(512, 256, 128, c_id, num_blocks)
            self.AADBlk6 = AADResBlock(256, 128, 64, c_id, num_blocks)
            self.AADBlk7 = AADResBlock(128, 64, 32, c_id, num_blocks)
            self.AADBlk8 = AADResBlock(64, 3, 32, c_id, num_blocks)
        else:
            self.AADBlk2 = AADResBlock(1024, 1024, 2048, c_id, num_blocks)
            self.AADBlk3 = AADResBlock(1024, 1024, 1024, c_id, num_blocks)
            self.AADBlk4 = AADResBlock(1024, 512, 512, c_id, num_blocks)
            self.AADBlk5 = AADResBlock(512, 256, 256, c_id, num_blocks)
            self.AADBlk6 = AADResBlock(256, 128, 128, c_id, num_blocks)
            self.AADBlk7 = AADResBlock(128, 64, 64, c_id, num_blocks)
            self.AADBlk8 = AADResBlock(64, 3, 64, c_id, num_blocks)

        self.apply(weight_init)

    def forward(self, z_attr, z_id):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        m2 = F.interpolate(
            self.AADBlk1(m, z_attr[0], z_id),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m3 = F.interpolate(
            self.AADBlk2(m2, z_attr[1], z_id),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m4 = F.interpolate(
            self.AADBlk3(m3, z_attr[2], z_id),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m5 = F.interpolate(
            self.AADBlk4(m4, z_attr[3], z_id),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m6 = F.interpolate(
            self.AADBlk5(m5, z_attr[4], z_id),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m7 = F.interpolate(
            self.AADBlk6(m6, z_attr[5], z_id),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m8 = F.interpolate(
            self.AADBlk7(m7, z_attr[6], z_id),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        y = self.AADBlk8(m8, z_attr[7], z_id)
        return torch.tanh(y)
