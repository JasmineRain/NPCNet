import torch.nn as nn
import torch.nn.functional as F
import torch
from models.RendPoint import sampling_features


__all__ = ["RendLoss"]


class RendLoss(nn.Module):
    def __init__(self):
        super(RendLoss, self).__init__()

    def forward(self, output, mask):
        # weight = torch.tensor([1., 3., 2.]).to(mask.device)
        pred = F.interpolate(output['coarse'], mask.shape[-2:], mode="bilinear", align_corners=False)
        gt_points = sampling_features(mask, output['points'], mode='bilinear', align_corners=False).argmax(dim=1)
        mask = mask.argmax(dim=1)
        rend = output['rend']
        seg_loss = F.cross_entropy(pred, mask)
        point_loss = F.cross_entropy(rend, gt_points)

        # print("\n")
        # print(gt_points.shape, (gt_points == 0).sum(), (gt_points == 1).sum(), (gt_points == 2).sum())
        # print(rend.shape, (rend.argmax(dim=1) == 0).sum(), (rend.argmax(dim=1) == 1).sum(), (rend.argmax(dim=1) == 2).sum())
        # print("\n")
        loss = seg_loss + point_loss

        return seg_loss, point_loss
