#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   emd_loss.py
@time    :   2019/09/06 13:03:54
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import torch
import torch.nn as nn

from configs import TORCH_BIN_DIM

#%%
class EMDLoss(nn.Module):
    """
    normalized Earth Mover’s Distance
        EMD(p, p_hat) = (
            1 / N * \sum_{k = 1}^{N} | CDF_{p}(k) - CDF_{p_hat}(k) |^{2}
        )^{1 / 2}
    """

    def __init__(self, exponent=2):
        """
        arguments:
            exponent (int)
        """
        super(EMDLoss, self).__init__()

        self._exponent = exponent

    def forward(self, predictions, ground_truths):
        """
        """
        pred_cdf = self._cdf(predictions)
        gt_cdf = self._cdf(ground_truths)

        loss = torch.pow(torch.mean(
            torch.pow(torch.abs(gt_cdf - pred_cdf), exponent=self._exponent),
            dim=TORCH_BIN_DIM
        ), exponent= 1 / self._exponent)

        return torch.mean(loss)

    def _cdf(self, input):
        """
        Cumulative Distribution Function
        """
        return torch.cumsum(input=input, dim=TORCH_BIN_DIM)


#%%
if __name__ == "__main__":

    size = (5, 10)
    preds = torch.softmax(torch.randn(*size), dim=TORCH_BIN_DIM)
    gts = torch.softmax(torch.ones(*size), dim=TORCH_BIN_DIM)

    criterion = EMDLoss()

    print(criterion(preds, gts))

#%%
