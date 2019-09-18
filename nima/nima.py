#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   nima.py
@time    :   2019/09/06 11:52:35
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import torch
import torch.nn as nn

#%%
class NIMA(nn.Module):
    """
    NIMA: Neural Image Assessment
        Hossein Talebi, Peyman Milanfar
        TIP (2018)
        https://ieeexplore.ieee.org/document/8352823

        - mima: baseline cnn + fully connected layer + softmax
        - baseline cnn: vgg16, inception-v2, mobile-net
    """

    def __init__(self, baseline_model_dict, out_features=10):
        super(NIMA, self).__init__()

        if baseline_model_dict["arch"] == "vgg":
            self._baseline_model = baseline_model_dict["model"]
            in_features = \
                self._baseline_model.classifier[-1].in_features
            del(self._baseline_model.classifier[-1])
            self._rating_distribution = nn.Linear(
                in_features=in_features, out_features=out_features
            )
        else:
            raise AssertionError(
                "ERROR: {} is not available!".format(baseline_model_dict["arch"])
            )

    def forward(self, x):
        """
        arguments:
            x (4-dim tensor) - images in a batch
                shape (batch_size, channels, height, width)

        return:
            softmax (2-dim tensor)
        """
        x = self._baseline_model(x)
        x = self._rating_distribution(x)
        x = torch.softmax(input=x, dim=1)

        return x


#%%
if __name__ == "__main__":

    from nima import vgg16_bn

    vgg16_net = vgg16_bn(
        pretrained_path="e:/src/jupyter/pytorch/models/vgg16_bn-6c64b313.pth"
    )
    vgg16_net_dict = {
        "arch": "vgg",
        "model": vgg16_net,
    }

    nima_net = NIMA(baseline_model_dict=vgg16_net_dict)
    print(nima_net)


#%%
