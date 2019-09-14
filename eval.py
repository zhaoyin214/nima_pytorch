#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   eval.py
@time    :   2019/09/13 16:45:48
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import torch
import time
from tqdm import tqdm

from configs import TEST_PHASE_LIST, TRAIN, TEST, \
    MODEL_SAVE_PATH, TORCH_FLOAT

#%%
def eval(model, dataloaders, criterion, device):
    """
    """
    epoch_loss = {}
    dataloader_sizes = {
        phase: len(dataloaders[phase]) for phase in TEST_PHASE_LIST
    }
    pbar_write = {
        phase: dataloader_sizes[phase] // 10 for phase in TEST_PHASE_LIST
    }

    model.to(device)
    model.eval()

    since = time.time()

    # test
    for phase in TEST_PHASE_LIST:
        running_loss = 0

        # progress bar
        pbar = tqdm(total=dataloader_sizes[phase],
                    desc=phase,
                    ascii=True)

        # iterate over samples
        for idx_batch, batch_samples in enumerate(dataloaders[phase]):

            # instances
            batch_images, batch_ground_truths = batch_samples
            batch_images = batch_images.type(TORCH_FLOAT).to(device)
            batch_ground_truths = batch_ground_truths.type(TORCH_FLOAT).to(device)

            # forward
            with torch.set_grad_enabled(mode=(phase == TRAIN)):

                batch_predicitons = model(batch_images).type(TORCH_FLOAT)
                loss = criterion(batch_predicitons, batch_ground_truths)

            # statistics
            running_loss += loss.item()

            pbar.update(1)
            if not (idx_batch % pbar_write[phase]):
                pbar.write("batch {}, loss: {:.4f}".format(
                    (idx_batch + 1), loss.item()))

        epoch_loss[phase] = running_loss / dataloader_sizes[phase]
        pbar.close()

        print("-" * 20)
        print("{} loss: {:.4f}".format(phase, epoch_loss[phase]))

    time_elapsed = time.time() - since
    print("training complete in {:.0f}m, {:.0f}s" \
        .format(time_elapsed // 60, time_elapsed % 60))

    return epoch_loss

#%%
if __name__ == "__main__":

    import os
    import pickle
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from configs import DEVICE, NUM_EPOCHS, \
        IMAGE_SIZE, \
        DATASET_CONFIG, DATA_LOADER_CONFIG, \
        MODEL_LOAD_PATH, EVAL_LOSS_PATH

    from dataset import AVADataset, NIMADataset
    from utils import plot_batch
    from nima import NIMA, EMDLoss, vgg16_bn

    # -- dataset --
    # ava dataset
    ava_datasets = {
        phase: AVADataset(
            image_root=DATASET_CONFIG["image_root"],
            label_filepath=DATASET_CONFIG[phase + "_label_filepath"]
        ) for phase in TEST_PHASE_LIST
    }

    # transforms
    transforms = {
        TEST: transforms.Compose(transforms=[
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))
        ])
    }

    nima_datasets = {
        phase: NIMADataset(
            dataset=ava_datasets[phase],
            transforms=transforms[phase],
            is_target_tensor=True
        ) for phase in TEST_PHASE_LIST
    }

    nima_dataloaders = {
        phase: DataLoader(
            dataset=nima_datasets[phase],
            batch_size=DATA_LOADER_CONFIG[phase + "_batch_size"],
            shuffle=DATA_LOADER_CONFIG[phase + "_shuffle"],
            num_workers=DATA_LOADER_CONFIG[phase + "_num_workers"]
        ) for phase in TEST_PHASE_LIST
    }

    batch_images, batch_ground_truths = iter(nima_dataloaders[TEST]).next()
    print(batch_images.shape)
    print(batch_ground_truths.shape)
    plot_batch(batch_images, batch_ground_truths)

    # -- nima network --
    vgg16_net = vgg16_bn(
        pretrained_path="e:/src/jupyter/pytorch/models/vgg16_bn-6c64b313.pth",
        is_init_weights=False
    )
    vgg16_net_dict = {
        "arch": "vgg",
        "model": vgg16_net,
    }
    nima_net = NIMA(baseline_model_dict=vgg16_net_dict)
    assert os.path.isfile(MODEL_LOAD_PATH), \
        "ERROR: model weights does not exist!"
    nima_net.load_state_dict(torch.load(MODEL_LOAD_PATH))
    print(nima_net)

    # -- loss --
    criterion = EMDLoss()

    # -- evaluation --
    loss = eval(
        model=nima_net, dataloaders=nima_dataloaders,
        criterion=criterion, device=DEVICE,
    )

    with open(EVAL_LOSS_PATH, "wb") as fw:
        pickle.dump(loss, fw)

#%%
