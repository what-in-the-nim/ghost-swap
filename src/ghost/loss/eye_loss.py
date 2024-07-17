import math
import os.path as op
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..network import FAN

FILE_DIR = op.dirname(__file__)
CHECKPOINT_PATH = op.join(FILE_DIR, "../../../weights/WFLW_4HG.pth")


class EyeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FAN(4, "False", "False", 98)

        # Load the pretrained model
        if not op.exists(CHECKPOINT_PATH):
            raise FileNotFoundError("WFLW_4HG.pth not found in the weights directory")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.mean = torch.tensor([[[0.5]], [[0.5]], [[0.5]]])
        self.std = torch.tensor([[[0.5]], [[0.5]], [[0.5]]])

    def eye_detect(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = (self.std * inputs) + self.mean

        outputs, _ = self.model(inputs)
        pred_heatmap = outputs[-1][:, :-1, :, :]
        pred_landmarks, _ = self.get_preds_fromhm(pred_heatmap)
        landmarks = pred_landmarks * 4.0
        eyes = torch.cat((landmarks[:, 96, :], landmarks[:, 97, :]), 1)
        return eyes, pred_heatmap[:, 96, :, :], pred_heatmap[:, 97, :, :]

    def get_preds_fromhm(
        self, heatmap: torch.Tensor, center=None, scale=None, rot=None
    ):
        max, idx = torch.max(
            heatmap.view(
                heatmap.size(0), heatmap.size(1), heatmap.size(2) * heatmap.size(3)
            ),
            2,
        )
        idx += 1
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
        preds[..., 0].apply_(lambda x: (x - 1) % heatmap.size(3) + 1)
        preds[..., 1].add_(-1).div_(heatmap.size(2)).floor_().add_(1)

        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = heatmap[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.FloatTensor(
                        [
                            hm_[pY, pX + 1] - hm_[pY, pX - 1],
                            hm_[pY + 1, pX] - hm_[pY - 1, pX],
                        ]
                    )
                    preds[i, j].add_(diff.sign_().mul_(0.25))

        preds.add_(-0.5)

        preds_orig = torch.zeros(preds.size())
        if center is not None and scale is not None:
            for i in range(heatmap.size(0)):
                for j in range(heatmap.size(1)):
                    preds_orig[i, j] = self.transform(
                        preds[i, j], center, scale, heatmap.size(2), rot, True
                    )

        return preds, preds_orig

    def transform(self, point, center, scale, resolution, rotation=0, invert=False):
        _pt = np.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = np.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if rotation != 0:
            rotation = -rotation
            r = np.eye(3)
            ang = rotation * math.pi / 180.0
            s = math.sin(ang)
            c = math.cos(ang)
            r[0][0] = c
            r[0][1] = -s
            r[1][0] = s
            r[1][1] = c

            t_ = np.eye(3)
            t_[0][2] = -resolution / 2.0
            t_[1][2] = -resolution / 2.0
            t_inv = torch.eye(3)
            t_inv[0][2] = resolution / 2.0
            t_inv[1][2] = resolution / 2.0
            t = reduce(np.matmul, [t_inv, r, t_, t])

        if invert:
            t = np.linalg.inv(t)
        new_point = (np.matmul(t, _pt))[0:2]

        return new_point.astype(int)

    def forward(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
    ) -> torch.Tensor:
        _, source_heatmap_left, source_heatmap_right = self.eye_detect(source_image)
        _, target_heatmap_left, target_heatmap_right = self.eye_detect(target_image)

        left_eye_loss = F.mse_loss(source_heatmap_left, target_heatmap_left)
        right_eye_loss = F.mse_loss(source_heatmap_right, target_heatmap_right)
        eye_loss = left_eye_loss + right_eye_loss
        return eye_loss
