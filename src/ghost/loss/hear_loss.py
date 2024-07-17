import torch
from torch import nn


class HEAR_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def rec_loss(
        self, X: torch.Tensor, Y: torch.Tensor, same: torch.Tensor
    ) -> torch.Tensor:
        same = same.unsqueeze(-1).unsqueeze(-1)
        same = same.expand(X.shape)
        X = torch.mul(X, same)
        Y = torch.mul(Y, same)
        return 0.5 * self.l2(X, Y)

    def id_loss(self, z_id_X: torch.Tensor, z_id_Y: torch.Tensor) -> torch.Tensor:
        z_id_size = z_id_X.size(1)
        inner_product = torch.bmm(
            z_id_X.view(-1, 1, z_id_size), z_id_Y.view(-1, z_id_size, 1)
        ).squeeze()
        return self.l1(torch.ones_like(inner_product), inner_product)

    def chg_loss(self, Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.l1(Y_hat, Y)

    def forward(
        self,
        Y_hat: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        z_id_X: torch.Tensor,
        z_id_Y: torch.Tensor,
        same: torch.Tensor,
    ) -> torch.Tensor:
        rec_loss = self.rec_loss(X, Y, same)
        id_loss = self.id_loss(z_id_X, z_id_Y)
        chg_loss = self.chg_loss(Y_hat, Y)
        return rec_loss + id_loss + chg_loss, rec_loss, id_loss, chg_loss
