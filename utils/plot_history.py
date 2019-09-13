#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   plot_history.py
@time    :   2019/09/04 09:43:59
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import matplotlib.pyplot as plt

def plot_history(history, filename=None):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121)
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_loss"], label="val loss")
    ax = fig.add_subplot(122)
    ax.semilogy(history["epoch"], history["train_loss"], label="train loss")
    ax.semilogy(history["epoch"], history["val_loss"], label="val loss")

    if filename is not None:
        plt.savefig(filename)

    plt.show()