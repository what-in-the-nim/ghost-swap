import torch
import torch.nn as nn


class AddCoordsTh(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).cuda()
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).cuda()
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).cuda()
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).cuda()
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        if self.with_boundary and type(heatmap) != type(None):
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(
                boundary_channel > 0.05, xx_channel, zero_tensor
            )
            yy_boundary_channel = torch.where(
                boundary_channel > 0.05, yy_channel, zero_tensor
            )
        if self.with_boundary and type(heatmap) != type(None):
            xx_boundary_channel = xx_boundary_channel.cuda()
            yy_boundary_channel = yy_boundary_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and type(heatmap) != type(None):
            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)
        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""

    def __init__(
        self,
        x_dim,
        y_dim,
        with_r,
        with_boundary,
        in_channels,
        first_one=False,
        *args,
        **kwargs
    ):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(
            x_dim=x_dim, y_dim=y_dim, with_r=with_r, with_boundary=with_boundary
        )
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel
