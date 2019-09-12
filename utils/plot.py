#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   plot.py
@time    :   2019/09/10 14:36:50
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""


#%%
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torchvision.utils import make_grid

#%%
def plot_sample(sample, idx_fig=1, figsize=(6, 8)):
    """
    """
    fig = plt.figure(num=idx_fig, figsize=figsize)
    fig.clf()

    # image
    ax = fig.add_subplot(2, 1, 1, frameon=False)
    ax.imshow(sample["image"])
    ax.axis("off")
    # ratings
    ax = fig.add_subplot(2, 1, 2, frameon=True)
    sample["ground_truth"].plot.bar(ax=ax)

    plt.show()


def plot_image_ratings(image, ratings, idx_fig=1, figsize=(6, 8)):
    """
    """
    fig = plt.figure(num=idx_fig, figsize=figsize)
    fig.clf()

    # image
    ax = fig.add_subplot(2, 1, 1, frameon=False)
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose(1, 2, 0)
    ax.imshow(image)
    ax.axis("off")
    # ratings
    ax = fig.add_subplot(2, 1, 2, frameon=True)
    x = list(range(1, ratings.shape[0] + 1))
    ax.bar(x, ratings)
    ax.set_xticks(x)

    plt.show()


def plot_batch(batch_images, batch_ground_truths, figsize=(8, 6)):
    """
    """
    nrow = 8

    image = make_grid(batch_images, nrow=nrow).numpy().transpose(1, 2, 0)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([1, 1, 1])
    image = std * image + mean

    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")

    if batch_ground_truths is not None:
        ground_truths = \
            make_grid(batch_ground_truths).numpy()[0]

        fig = plt.figure(figsize=figsize)
        batch_size = ground_truths.shape[0]
        ncol = math.ceil(batch_size / nrow)
        for idx in range(batch_size):
            ax = fig.add_subplot(ncol, nrow, idx + 1)
            ax.bar(list(range(1, 11)), ground_truths[idx])
            ax.set_yticks([])

    plt.show()

