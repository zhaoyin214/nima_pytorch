#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   data_generator.py
@time    :   2019/09/11 19:04:54
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""


#%%
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


#%%
class NIMADataset(Dataset):
    """

    """
    def __init__(self, dataset, transforms=None, is_target_tensor=True):
        super(NIMADataset, self).__init__()

        self._dataset = dataset
        self._transforms = transforms
        self._is_target_tensor = is_target_tensor

    def __len__(self):
        """
        len()
        """
        return len(self._dataset)

    def __getitem__(self, index):
        """
        index
        """

        sample = self._dataset[index]
        image = sample["image"]
        ground_truth = sample["ground_truth"].values
        ground_truth = ground_truth / np.sum(ground_truth)

        if self._transforms is not None:
            image = Image.fromarray(image).convert("RGB")
            image = self._transforms(image)

        if self._is_target_tensor:
            ground_truth = torch.from_numpy(ground_truth)

        return image, ground_truth


#%%
if __name__ == "__main__":

    from torchvision import transforms
    from torch.utils.data import DataLoader

    from dataset import AVADataset
    from configs import AVA_LABEL_CLEANED_FILEPATH, AVA_IMAGE_ROOT
    from utils import plot_image_ratings, plot_batch

    ava_dataset = AVADataset(image_root=AVA_IMAGE_ROOT,
                             label_filepath=AVA_LABEL_CLEANED_FILEPATH)

    print(ava_dataset.labels.describe())

    transforms = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))
    ])

    nima_dataset = NIMADataset(
        dataset=ava_dataset, transforms=transforms,
        is_target_tensor=True
    )
    index = np.random.choice(range(len(nima_dataset)))
    image, ratings = nima_dataset[index]

    print(index)
    print(type(image))
    plot_image_ratings(image, ratings)

    nima_dataloader = DataLoader(dataset=nima_dataset,
                                 shuffle=True,
                                 batch_size=4,
                                 num_workers=0)

    batch_images, batch_ground_truths = iter(nima_dataloader).next()
    print(batch_images.shape)
    print(batch_ground_truths.shape)
    plot_batch(batch_images, batch_ground_truths)

#%%
