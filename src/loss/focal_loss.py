import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable
from typing import Iterable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, Iterable):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred, target):
        # weight_dmc = torch.zeros(self.cls)
        # for c in range(0, self.cls):
        #     weight_dmc[c] = torch.sum(target == c)
        # idx_nz = (weight_dmc != 0)
        # weight_dmc[idx_nz] = 1.0 / weight_dmc[idx_nz]
        # weight_dmc = F.normalize(weight_dmc, p=1, dim=0) * 10
        # self.alpha = torch.mul(self.alpha, weight_dmc)

        # if pred.dim() > 2:
        #     pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
        #     pred = pred.transpose(1, 2)  # N,C,H*W => N,H*W,C
        #     pred = pred.contiguous().view(-1, pred.size(2))  # N,H*W,C => N*H*W,C
        pred = pred.contiguous().view(-1, pred.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).long()

        logpt = nn.functional.log_softmax(pred, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
