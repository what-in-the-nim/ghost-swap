import math

import torch
import torch.nn as nn


class MultiscaleDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        num_layers: int = 3,
        num_discriminators: int = 3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid: bool = False,
        return_intermediate_feature: bool = False,
    ) -> None:
        super().__init__()
        self.num_discriminators = num_discriminators
        self.n_layers = num_layers
        self.return_intermediate_feature = return_intermediate_feature

        for discriminator_idx in range(num_discriminators):
            netD = NLayerDiscriminator(
                input_nc,
                ndf,
                num_layers,
                norm_layer,
                use_sigmoid,
                return_intermediate_feature,
            )
            if return_intermediate_feature:
                for layer_idx in range(num_layers + 2):
                    layer_name = f"scale{discriminator_idx}_layer{layer_idx}"
                    netD_layer_name = f"model{layer_idx}"
                    # Add the intermediate layer to the model
                    self.add_module(layer_name, getattr(netD, netD_layer_name))
            else:
                model_name = f"layer{discriminator_idx}"
                self.add_module(model_name, netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def singleD_forward(self, model, input):
        if self.return_intermediate_feature:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        result = []
        input_downsampled = input
        for i in range(self.num_discriminators):
            if self.return_intermediate_feature:
                model = [
                    getattr(
                        self,
                        "scale"
                        + str(self.num_discriminators - 1 - i)
                        + "_layer"
                        + str(j),
                    )
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, "layer" + str(self.num_discriminators - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_discriminators - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_sigmoid: bool = False,
        return_intermediate_feature: bool = False,
    ):
        super().__init__()
        self.return_intermediate_feature = return_intermediate_feature
        self.n_layers = n_layers

        kernel_width = 4
        padding_width = int(math.ceil((kernel_width - 1) / 2))
        sequence = [
            [
                nn.Conv2d(
                    input_nc,
                    ndf,
                    kernel_size=kernel_width,
                    stride=2,
                    padding=padding_width,
                ),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(
                        nf_prev,
                        nf,
                        kernel_size=kernel_width,
                        stride=2,
                        padding=padding_width,
                    ),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(
                    nf_prev,
                    nf,
                    kernel_size=kernel_width,
                    stride=1,
                    padding=padding_width,
                ),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [
            [
                nn.Conv2d(
                    nf, 1, kernel_size=kernel_width, stride=1, padding=padding_width
                )
            ]
        ]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if return_intermediate_feature:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.return_intermediate_feature:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
