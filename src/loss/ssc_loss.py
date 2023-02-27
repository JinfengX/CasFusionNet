from functools import partial

import torch
from torch import nn

from .ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from pointnet2_ops.pointnet2_utils import gather_operation, furthest_point_sample
from .focal_loss import FocalLoss

chamfer_dist = chamfer_3DDist()


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points)
    )
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def chamfer_sqrt(p1, p2):
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2, idx1


def chamfer(p1, p2):
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2), idx1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def seg_loss(pred, gt, idx, gamma, weight):
    gt_label = gt[:, :, 3].reshape(-1, gt.shape[1]).gather(1, idx.long())
    pred_label = pred[:, :, 3:]
    ls_seg = FocalLoss(gamma, alpha=weight)(pred_label, gt_label)
    return ls_seg, pred_label, gt_label


class Loss(nn.Module):
    def __init__(self, seg_weight, sqrt=True):
        super(Loss, self).__init__()

        self.cd_ls = chamfer_sqrt if sqrt else chamfer
        self.seg_ls = partial(seg_loss, weight=seg_weight)

    def forward(self, pcds_pred, gt, epoch):
        """loss function
        Args
            pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
        """

        gamma = max(min(5 * (epoch / 200), max(0, 20)), min(0, 20))
        cd_ls, seg_ls = [], []
        cd_i, seg_i, pred_label, gt_label = None, None, None, None
        for i, p_i in enumerate(pcds_pred):
            gt_i = (
                gt
                if i == 0 or i == len(pcds_pred) - 1
                else fps_subsample(gt, p_i.shape[1])
            )
            cd_i, idx_i = self.cd_ls(
                p_i[:, :, :3].contiguous(), gt_i[:, :, :3].contiguous()
            )

            seg_i, pred_label, gt_label = self.seg_ls(p_i, gt_i, idx_i, gamma)
            if i > 0:
                cd_ls.append(cd_i)
                seg_ls.append(seg_i)
        loss_cmp, loss_seg = sum(cd_ls) * 1e3, sum(seg_ls) * 1e2
        loss_all = loss_cmp + loss_seg

        return {
            "sum_loss": loss_all,
            "last_cd": cd_i,
            "last_seg": seg_i,
            "pred_seg": pred_label,
            "gt_seg": gt_label,
        }
