from torch import nn
import torch
import torch.nn.functional as F

from ..utils import weight_init


def conv4x4(in_c, out_c, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip, backbone):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        if backbone == "linknet":
            return x + skip
        else:
            return torch.cat((x, skip), dim=1)


class MLAttrEncoder(nn.Module):
    """
    Multi-level Attribute Encoder
    """
    def __init__(self, backbone):
        super(MLAttrEncoder, self).__init__()
        self.backbone = backbone
        self.conv1 = conv4x4(3, 32)
        self.conv2 = conv4x4(32, 64)
        self.conv3 = conv4x4(64, 128)
        self.conv4 = conv4x4(128, 256)
        self.conv5 = conv4x4(256, 512)
        self.conv6 = conv4x4(512, 1024)
        self.conv7 = conv4x4(1024, 1024)

        if backbone == "unet":
            self.deconv1 = deconv4x4(1024, 1024)
            self.deconv2 = deconv4x4(2048, 512)
            self.deconv3 = deconv4x4(1024, 256)
            self.deconv4 = deconv4x4(512, 128)
            self.deconv5 = deconv4x4(256, 64)
            self.deconv6 = deconv4x4(128, 32)
        elif backbone == "linknet":
            self.deconv1 = deconv4x4(1024, 1024)
            self.deconv2 = deconv4x4(1024, 512)
            self.deconv3 = deconv4x4(512, 256)
            self.deconv4 = deconv4x4(256, 128)
            self.deconv5 = deconv4x4(128, 64)
            self.deconv6 = deconv4x4(64, 32)
        self.apply(weight_init)

    def forward(self, Xt):
        feat1 = self.conv1(Xt)
        # 32x128x128
        feat2 = self.conv2(feat1)
        # 64x64x64
        feat3 = self.conv3(feat2)
        # 128x32x32
        feat4 = self.conv4(feat3)
        # 256x16xx16
        feat5 = self.conv5(feat4)
        # 512x8x8
        feat6 = self.conv6(feat5)
        # 1024x4x4
        z_attr1 = self.conv7(feat6)
        # 1024x2x2

        z_attr2 = self.deconv1(z_attr1, feat6, self.backbone)
        z_attr3 = self.deconv2(z_attr2, feat5, self.backbone)
        z_attr4 = self.deconv3(z_attr3, feat4, self.backbone)
        z_attr5 = self.deconv4(z_attr4, feat3, self.backbone)
        z_attr6 = self.deconv5(z_attr5, feat2, self.backbone)
        z_attr7 = self.deconv6(z_attr6, feat1, self.backbone)
        z_attr8 = F.interpolate(
            z_attr7, scale_factor=2, mode="bilinear", align_corners=True
        )
        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8