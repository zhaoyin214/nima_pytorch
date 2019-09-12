#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   split.py
@time    :   2019/09/12 14:40:42
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
from sklearn.model_selection import train_test_split


#%%
def train_val_test_split(dataset, val_ratio, test_ratio):
    """
    splitting a dataset into training, validation and test sets
    """
    index_name = dataset.index.name
    train_val_sets, test_set = train_test_split(
        dataset, test_size=test_ratio, shuffle=True
    )
    train_set, val_set = train_test_split(
        train_val_sets, test_size=val_ratio, shuffle=True
    )

    train_set.reset_index(drop=True, inplace=True)
    train_set.index.name = index_name
    val_set.reset_index(drop=True, inplace=True)
    val_set.index.name = index_name
    test_set.reset_index(drop=True, inplace=True)
    test_set.index.name = index_name

    return train_set, val_set, test_set


#%%
