#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   ava.py
@time    :   2019/09/07 16:04:36
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import pandas as pd
import os
from skimage.io import imread, imsave
from PIL import ImageFile

from utils import download_image, resize_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

#%%
class AVADataset(object):
    """
    AVA dataset
    """

    # global attributes
    _label_index = "index"
    _label_key_id = "id"
    _label_key_image_id = "image_id"
    _label_key_ratings = ["rating_" + str(score) for score in range(1, 11)]
    _label_key_tags = ["tag_1", "tag_2"]
    _label_key_challenge_id = "challenge_id"
    _image_large_side = 640

    def __init__(self, image_root=None, label_filepath=None):
        """
        Constructor
        """

        if image_root is not None:
            assert os.path.isdir(image_root), \
                "ERROR: {} does not exist!".format(image_root)
            self._image_root = image_root
        else:
            self._image_root = None

        if label_filepath is not None:
            assert os.path.isfile(label_filepath), \
                "ERROR: {} does not exist!".format(label_filepath)
            self._read_label(filepath=label_filepath)
        else:
            self._labels = None

    def __len__(self):
        """
        number of samples
        """
        return len(self._labels)

    def __getitem__(self, index):
        """
        index
        """
        image_id = str(self._labels.loc[index, self._label_key_image_id])
        image_filepath = os.path.join(self._image_root, image_id + ".jpg")
        image = imread(fname=image_filepath)

        ratings = self._labels.loc[index, self._label_key_ratings]

        sample = {"image": image, "ground_truth": ratings}

        return sample

    def _read_label(self, filepath):

        if os.path.splitext(filepath)[-1] == ".txt":
            columns = [self._label_key_id, self._label_key_image_id] + \
                self._label_key_ratings + self._label_key_tags + \
                [self._label_key_challenge_id]

            self._labels = pd.read_csv(
                filepath_or_buffer=filepath, delimiter=" ", header=None
            )
            self._labels.columns=columns
            self._labels.set_index(keys=[self._label_key_id], inplace=True)
        elif os.path.splitext(filepath)[-1] == ".csv":
            self._labels = pd.read_csv(
                filepath_or_buffer=filepath, header=0, index_col=0
            )
        # self._labels[self._label_key_image_id] = \
        #     self._labels[self._label_key_image_id].astype(str)

    def _verification(self):
        assert isinstance(self._labels, pd.DataFrame), \
            "ERROR: labels is not available!"
        assert os.path.isdir(self._image_root), \
            "ERROR: {} is not available!".format(self._image_root)

    def data_cleaning(self, start_idx=1,
                      abnormal_filepath=None,
                      label_cleaned_filepath=None):
        """
        index
        modes: ["check", "download", "truncated"]
        """
        abnormals = pd.DataFrame(
            columns=[self._label_key_id, self._label_key_image_id]
        )

        self._verification()

        # traversing samples
        for index in range(start_idx, len(self._labels) + 1):

            image_id = str(self._labels.loc[index, self._label_key_image_id])
            image_filepath = os.path.join(self._image_root, image_id + ".jpg")
            print("index: {}, image id: {}".format(index, image_id))

            # abnormals
            if not os.path.isfile(image_filepath):

                print("!" * 20)
                print("abnormal: index: {}, image id: {}".format(index, image_id))
                print("!" * 20)
                # record abnormal samples
                abnormals = abnormals.append({
                    self._label_key_id: index,
                    self._label_key_image_id: image_id
                }, ignore_index=True)
                # remove abnormal samples
                self._labels.drop(labels=[index], axis="index", inplace=True)

        abnormals.set_index(keys=self._label_key_id, inplace=True)
        self._labels.reset_index(drop=True, inplace=True)
        self._labels.index.name = self._label_key_id

        # def _download(image_id):

        #     print("image {} is downloading...".format(image_id))
        #     image_root = os.path.join(self._image_root, "append")
        #     download_image(image_id=image_id, image_root=image_root)
        #     image_filepath = os.path.join(image_root, image_id)
        #     image = imread(fname=image_filepath + "_.jpg")
        #     image = resize_image(
        #         image=image, large_side=self._image_large_side
        #     )
        #     imsave(fname=image_filepath + ".jpg", arr=image)
        #     # image_filepath = os.path.join(self._image_root, image_id + ".jpg")
        #     # imsave(fname=image_filepath, arr=image)

        # # traversing samples
        # for index in range(start_idx, len(self._labels) + 1):

        #     image_id = str(
        #         self._labels.loc[index, self._label_key_image_id].values[0]
        #     )
        #     image_filepath = os.path.join(self._image_root, image_id + ".jpg")
        #     print("index: {}, image id: {}".format(index, image_id))

        #     if os.path.isfile(image_filepath):
        #         # try:
        #         #     warnings.warn(imread(fname=image_filepath))
        #         # except:
        #         #     _download(image_id=image_id)
        #         pass
        #     else:
        #         try:
        #             _download(image_id=image_id)
        #         except:
        #             abnormals = abnormals.append({
        #                 self._label_key_id: index,
        #                 self._label_key_image_id: image_id
        #             }, ignore_index=True)

        if abnormal_filepath is not None:
            abnormals.to_csv(abnormal_filepath)

        if label_cleaned_filepath is not None:
            self._labels.to_csv(label_cleaned_filepath)

        return abnormals

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, filepath):
        if os.path.isfile(filepath):
            self._read_label(filepath=filepath)
        else:
            raise AssertionError(
                "ERROR: {} is not available!".format(filepath)
            )

    @property
    def image_root(self):
        return self._image_root

    @image_root.setter
    def image_root(self, image_root):
        if os.path.isdir(image_root):
            self._image_root = image_root
        else:
            raise AssertionError(
                "ERROR: {} is not available!".format(image_root)
            )

#%%
if __name__ == "__main__":

    from configs import AVA_LABEL_FILEPATH, AVA_IMAGE_ROOT, \
        AVA_ABNORMAL, AVA_LABEL_CLEANED_FILEPATH, \
        TEST_RATIO, VAL_RATIO, \
        AVA_TRAIN_LABEL_FILEPATH, AVA_VAL_LABEL_FILEPATH, \
        AVA_TEST_LABEL_FILEPATH
    from utils import plot_sample
    from dataset import train_val_test_split

    ava_dataset = AVADataset()

    # # only for first run, data cleaning
    # ava_dataset.labels = AVA_LABEL_FILEPATH
    # ava_dataset.image_root = AVA_IMAGE_ROOT

    # # data cleaning
    # abnormals = ava_dataset.data_cleaning(
    #     start_idx=1, abnormal_filepath=AVA_ABNORMAL,
    #     label_cleaned_filepath=AVA_LABEL_CLEANED_FILEPATH
    # )

    ava_dataset.labels = AVA_LABEL_CLEANED_FILEPATH
    ava_dataset.image_root = AVA_IMAGE_ROOT

    print("-" * 20)
    print("AVA dataset labels:")
    print("-" * 20)
    print(ava_dataset.labels.sample(10))
    print(ava_dataset.labels.describe())

    # samples
    for idx in range(len(ava_dataset)):

        print(idx)
        sample = ava_dataset[idx]
        plot_sample(sample=sample)

        if idx == 3:
            break

    # training, validation and test sets
    labels_train, labels_val, labels_test = train_val_test_split(
        dataset=ava_dataset.labels, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
    )
    labels_train.to_csv(AVA_TRAIN_LABEL_FILEPATH)
    labels_val.to_csv(AVA_VAL_LABEL_FILEPATH)
    labels_test.to_csv(AVA_TEST_LABEL_FILEPATH)

#%%
