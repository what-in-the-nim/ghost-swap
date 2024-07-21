import torch
import torch.nn as nn


class MultiLevelAttributesEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_channel = [3, 32, 64, 128, 256, 512, 1024, 1024]
        self.encoder = nn.ModuleDict(
            {
                f"layer_{i}": nn.Sequential(
                    nn.Conv2d(
                        self.encoder_channel[i],
                        self.encoder_channel[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.encoder_channel[i + 1]),
                    nn.LeakyReLU(0.1),
                )
                for i in range(7)
            }
        )

        self.decoder_inchannel = [1024, 2048, 1024, 512, 256, 128]
        self.decoder_outchannel = [1024, 512, 256, 128, 64, 32]
        self.decoder = nn.ModuleDict(
            {
                f"layer_{i}": nn.Sequential(
                    nn.ConvTranspose2d(
                        self.decoder_inchannel[i],
                        self.decoder_outchannel[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.decoder_outchannel[i]),
                    nn.LeakyReLU(0.1),
                )
                for i in range(6)
            }
        )

        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        arr_x = []
        for i in range(7):
            x = self.encoder[f"layer_{i}"](x)
            arr_x.append(x)

        arr_y = []
        arr_y.append(arr_x[6])
        y = arr_x[6]
        for i in range(6):
            y = self.decoder[f"layer_{i}"](y)
            y = torch.cat((y, arr_x[5 - i]), 1)
            arr_y.append(y)

        arr_y.append(self.Upsample(y))

        return arr_y
