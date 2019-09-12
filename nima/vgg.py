#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   vgg.py
@time    :   2019/09/05 15:05:10
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import os
import torch
import torch.nn as nn


#%%
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


#%%
class VGG(nn.Module):
    """
    VGG - Very Deep Convolutional Networks For Large-Scale Image Recognition
    """
    def __init__(self, features, num_classes=1000, is_init_weights=True):
        """
        """
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # linear units
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        if is_init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        """
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)


def _make_layers(cfg, is_batch_norm=False):
    """
    """
    layers = []
    in_channels = 3

    for v in cfg:

        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v,
                                    kernel_size=3, padding=1))
            if is_batch_norm:
                layers.append(nn.BatchNorm2d(num_features=v))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            in_channels=v

    return nn.Sequential(*layers)


#%%
def _vgg(arch, cfg, is_batch_norm, pretrained_path=None, **kwargs):
    """"""

    if os.path.isfile(pretrained_path):
        is_pretrained = True
    else:
        is_pretrained = False

    if is_pretrained:
        kwargs["is_init_weights"] = False

    model = VGG(_make_layers(cfg=cfgs[cfg], is_batch_norm=is_batch_norm),
                **kwargs)

    if is_pretrained:
        model.load_state_dict(torch.load(pretrained_path))

    return model


#%%
def vgg11(pretrained_path, **kwargs):
    """
    VGG 11-layer model (configuration "A")
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg11", cfg="A", is_batch_norm=False,
                pretrained_path=pretrained_path, **kwargs)


def vgg11_bn(pretrained_path, **kwargs):
    """
    VGG 11-layer model (configuration "A") with batch normalization
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg11_bn", cfg="A", is_batch_norm=True,
                pretrained_path=pretrained_path, **kwargs)


def vgg13(pretrained_path, **kwargs):
    """
    VGG 13-layer model (configuration "B")
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg13", cfg="B", is_batch_norm=False,
                pretrained_path=pretrained_path, **kwargs)


def vgg13_bn(pretrained_path, **kwargs):
    """
    VGG 13-layer model (configuration "B") with batch normalization
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg13_bn", cfg="B", is_batch_norm=True,
                pretrained_path=pretrained_path, **kwargs)


def vgg16(pretrained_path, **kwargs):
    """
    VGG 16-layer model (configuration "D")
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg16", cfg="D", is_batch_norm=False,
                pretrained_path=pretrained_path, **kwargs)


def vgg16_bn(pretrained_path, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg16_bn", cfg="D", is_batch_norm=True,
                pretrained_path=pretrained_path, **kwargs)


def vgg19(pretrained_path, **kwargs):
    """
    VGG 19-layer model (configuration "E")
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg19", cfg="E", is_batch_norm=False,
                pretrained_path=pretrained_path, **kwargs)


def vgg19_bn(pretrained_path, **kwargs):
    """
    VGG 19-layer model (configuration "E") with batch normalization
    Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf>

    arguments:
        pretrained (string) - if not none, returns a model pre-trained
            on ImageNet
    """
    return _vgg(arch="vgg19_bn", cfg="E", is_batch_norm=True,
                pretrained_path=pretrained_path, **kwargs)


#%%
if __name__ == "__main__":

    model = vgg16_bn(
        pretrained_path="e:/src/jupyter/pytorch/models/vgg16_bn-6c64b313.pth"
    )
    print(model)


#%%
