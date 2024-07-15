import torch
import torch.nn as nn


class AADLayer(nn.Module):
    def __init__(self, c_x, attr_c, c_id):
        super(AADLayer, self).__init__()
        self.attr_c = attr_c
        self.c_id = c_id
        self.c_x = c_x

        self.conv1 = nn.Conv2d(
            attr_c, c_x, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.conv2 = nn.Conv2d(
            attr_c, c_x, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc1 = nn.Linear(c_id, c_x)
        self.fc2 = nn.Linear(c_id, c_x)
        self.norm = nn.InstanceNorm2d(c_x, affine=False)

        self.conv_h = nn.Conv2d(c_x, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, h_in, z_attr, z_id):
        # h_in cxnxn
        # zid 256x1x1
        # zattr cxnxn
        h = self.norm(h_in)
        gamma_attr = self.conv1(z_attr)
        beta_attr = self.conv2(z_attr)

        gamma_id = self.fc1(z_id)
        beta_id = self.fc2(z_id)
        A = gamma_attr * h + beta_attr
        gamma_id = gamma_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        beta_id = beta_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        I = gamma_id * h + beta_id

        M = torch.sigmoid(self.conv_h(h))

        out = (torch.ones_like(M).to(M.device) - M) * A + M * I
        return out


class AddBlocksSequential(nn.Sequential):
    def forward(self, *inputs):
        h, z_attr, z_id = inputs
        for i, module in enumerate(self._modules.values()):
            if i % 3 == 0 and i > 0:
                inputs = (inputs, z_attr, z_id)
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class AADResBlock(nn.Module):
    def __init__(self, cin, cout, c_attr, c_id, num_blocks) -> None:
        super().__init__()
        self.cin = cin
        self.cout = cout

        add_blocks = []
        for i in range(num_blocks):
            out = cin if i < (num_blocks - 1) else cout
            add_blocks.extend(
                [
                    AADLayer(cin, c_attr, c_id),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cin, out, kernel_size=3, stride=1, padding=1, bias=False),
                ]
            )
        self.add_blocks = AddBlocksSequential(*add_blocks)

        if cin != cout:
            last_add_block = [
                AADLayer(cin, c_attr, c_id),
                nn.ReLU(inplace=True),
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            ]
            self.last_add_block = AddBlocksSequential(*last_add_block)

    def forward(
        self, hidden_state: torch.Tensor, z_attr: torch.Tensor, z_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the AADResBlock

        Parameters:
        ----------
        hidden_state: torch.Tensor
            The hidden state of the previous block.
        z_attr: torch.Tensor
            The attribute latent vector from the parallel attribute encoder layer.
        z_id: torch.Tensor
            The identity latent vector from ArcFace model of shape (B, 512)

        Returns:
        -------
        x: torch.Tensor
            The output of the AADResBlock
        """
        x = self.add_blocks(hidden_state, z_attr, z_id)
        if self.cin != self.cout:
            hidden_state = self.last_add_block(hidden_state, z_attr, z_id)
        x = x + hidden_state
        return x
