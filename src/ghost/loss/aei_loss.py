import torch
from torch import nn


class AEILoss(nn.Module):
    def __init__(
        self,
        att_weight: float | int = 10,
        id_weight: float | int = 5,
        rec_weight: float | int = 10,
    ) -> None:
        super().__init__()

        self.att_weight = att_weight
        self.id_weight = id_weight
        self.rec_weight = rec_weight

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def att_loss(
        self, z_att_X: list[torch.Tensor], z_att_Y: list[torch.Tensor]
    ) -> torch.Tensor:
        loss = 0
        for i in range(8):
            loss += self.l2(z_att_X[i], z_att_Y[i])
        return 0.5 * loss

    def id_loss(self, z_id_X: torch.Tensor, z_id_Y: torch.Tensor) -> torch.Tensor:
        inner_product = torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze()
        return self.l1(torch.ones_like(inner_product), inner_product)

    def rec_loss(
        self, X: torch.Tensor, Y: torch.Tensor, same: torch.Tensor
    ) -> torch.Tensor:
        same = same.unsqueeze(-1).unsqueeze(-1)
        same = same.expand(X.shape)
        X = torch.mul(X, same)
        Y = torch.mul(Y, same)
        return 0.5 * self.l2(X, Y)

    def forward(self, X, Y, z_att_X, z_att_Y, z_id_X, z_id_Y, same):

        att_loss = self.att_loss(z_att_X, z_att_Y)
        id_loss = self.id_loss(z_id_X, z_id_Y)
        rec_loss = self.rec_loss(X, Y, same)

        return (
            self.att_weight * att_loss
            + self.id_weight * id_loss
            + self.rec_weight * rec_loss,
            att_loss,
            id_loss,
            rec_loss,
        )
