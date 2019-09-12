#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   train.py
@time    :   2019/09/12 12:23:38
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import copy
import torch
import time
import sys
from tqdm import tqdm

from configs import TRAINING_PHASE_LIST, TRAIN, VALIDATION, \
    MODEL_SAVE_PATH

#%%
def train(model, dataloaders, criterion, optimizer, scheduler, device,
          num_epochs=25,
          early_stopping_patience=None,
          reduce_lr_on_plateau=None):
    """
    """
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    dataloader_sizes = {
        phase: len(dataloaders[phase]) for phase in TRAINING_PHASE_LIST
    }
    pbar_write = {
        phase: dataloader_sizes[phase] // 10 for phase in TRAINING_PHASE_LIST
    }

    if early_stopping_patience is not None:
        early_stopping_counter = 0

    if reduce_lr_on_plateau is not None:
        plateau_counter = 0

    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max
    model.to(device)

    since = time.time()

    # train
    for epoch in range(1, num_epochs + 1):

        print("-" * 20)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 20)

        # each epoch has a training and validation phase
        for phase in TRAINING_PHASE_LIST:

            if phase == TRAIN:
                model.train()  # set model to training mode
                scheduler.step()
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0

            # progress bar
            pbar = tqdm(total=dataloader_sizes[phase],
                        desc=phase,
                        ascii=True)

            # iterate over samples
            for idx_batch, batch_samples in enumerate(dataloaders[phase]):

                # instances
                batch_images, batch_ground_truths = batch_samples
                batch_images = batch_images.to(device)
                batch_ground_truths = batch_ground_truths.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(mode=(phase == TRAIN)):

                    batch_predicitons = model(batch_images)
                    loss = criterion(batch_predicitons, batch_ground_truths)

                    # backward + optimize, only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

                pbar.update(1)
                if not (idx_batch % pbar_write[phase]):
                    pbar.write("batch {}, loss: {:.4f}".format(
                        (idx_batch + 1), loss.item()))

            epoch_loss = running_loss / dataloader_sizes[phase]
            pbar.close()

            print("-" * 20)
            print("{} loss: {:.4f}".format(phase, epoch_loss))

            # history tracking
            if phase == TRAIN:
                history["epoch"].append(epoch)
                history["train_loss"].append(epoch_loss)
            else:
                history["val_loss"].append(epoch_loss)

            if phase == VALIDATION:

                # early stopping
                if early_stopping_patience is not None:

                    if epoch_loss >= best_loss:
                        early_stopping_counter += 1
                    else:
                        early_stopping_counter = 0

                    if early_stopping_counter >= early_stopping_patience:
                        print("early stopping ...")
                        # load best model weights
                        model.load_state_dict(best_model_weights)
                        return model, history

                # reduce lr on plateau
                if reduce_lr_on_plateau is not None:

                    if epoch_loss >= best_loss:
                        plateau_counter += 1
                    else:
                        plateau_counter = 0

                    if plateau_counter >= reduce_lr_on_plateau["patience"]:
                        plateau_counter = 0
                        print("error plateau, reducing the learning rate ...")
                        for params in optimizer.param_groups:
                            params["lr"] *= reduce_lr_on_plateau["factor"]
                        # print last lr
                        print("learning rate: {}".format(params["lr"]))

                # best save according to val_loss
                if epoch_loss < best_loss:
                    print("best save ...")
                    best_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    model_path = MODEL_SAVE_PATH.format(
                        history["epoch"][epoch],
                        history["train_loss"][epoch],
                        history["val_loss"][epoch]
                    )
                    torch.save(model.state_dict(), model_path)
                    print(model_path)

        print("\n\n")

    time_elapsed = time.time() - since
    print("training complete in {:.0f}m, {:.0f}s" \
        .format(time_elapsed // 60, time_elapsed % 60))
    print("best val loss: {.4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_weights)

    return model, history

#%%
if __name__ == "__main__":

    import os
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from configs import DEVICE, NUM_EPOCHS, \
        TRAIN_RESIZE, IMAGE_SIZE, \
        DATASET_CONFIG, DATA_LOADER_CONFIG, \
        LEARNING_RATE, WEIGHT_DECAY, BETAS, WEIGHT_DECAY, \
        EARLY_STOPPING_PATIENCE,  REDUCE_LR_ON_PLATEAU, \
        LR_SCHEDULE_STEP_SIZE, LR_SCHEDULE_GAMMA, \
        MODEL_LOAD_PATH, HISTORY_PATH, HISTORY_PLOT_PATH

    from dataset import AVADataset, NIMADataset
    from utils import plot_batch
    from nima import NIMA, EMDLoss, vgg16_bn

    # -- dataset --
    # ava dataset
    ava_datasets = {
        phase: AVADataset(
            image_root=DATASET_CONFIG["image_root"],
            label_filepath=DATASET_CONFIG[phase + "_label_filepath"]
        ) for phase in TRAINING_PHASE_LIST
    }

    # transforms
    transforms = {
        TRAIN: transforms.Compose(transforms=[
            transforms.Resize(TRAIN_RESIZE),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))
        ]),
        VALIDATION: transforms.Compose(transforms=[
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
        ) for phase in TRAINING_PHASE_LIST
    }

    nima_dataloaders = {
        phase: DataLoader(
            dataset=nima_datasets[phase],
            batch_size=DATA_LOADER_CONFIG[phase + "_batch_size"],
            shuffle=DATA_LOADER_CONFIG[phase + "_shuffle"],
            num_workers=DATA_LOADER_CONFIG[phase + "_num_workers"]
        ) for phase in TRAINING_PHASE_LIST
    }

    batch_images, batch_ground_truths = iter(nima_dataloaders[TRAIN]).next()
    print(batch_images.shape)
    print(batch_ground_truths.shape)
    plot_batch(batch_images, batch_ground_truths)

    # -- nima network --
    vgg16_net = vgg16_bn(
        pretrained_path="e:/src/jupyter/pytorch/models/vgg16_bn-6c64b313.pth"
    )
    vgg16_net_dict = {
        "arch": "vgg",
        "model": vgg16_net,
    }
    nima_net = NIMA(baseline_model_dict=vgg16_net_dict)
    if os.path.isfile(MODEL_LOAD_PATH):
        faceboxes_net.load_state_dict(torch.load(MODEL_LOAD_PATH))
    print(nima_net)

    # optimizer
    optimizer = torch.optim.Adam(params=nima_net.parameters(),
                                 lr=LEARNING_RATE,
                                 betas=BETAS,
                                 weight_decay=WEIGHT_DECAY)
    # scheduler
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=LR_SCHEDULE_STEP_SIZE,
        gamma=LR_SCHEDULE_GAMMA
    )

    # -- loss --
    criterion = EMDLoss()

    # -- train --
    nima_net, history = train(
        model=nima_net, dataloaders=nima_dataloaders,
        criterion=criterion, optimizer=optimizer,
        scheduler=exp_lr_scheduler, device=DEVICE,
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        reduce_lr_on_plateau=REDUCE_LR_ON_PLATEAU
    )

    with open(HISTORY_PATH, "wb") as fw:
        pickle.dump(history, fw)

    plot_history(history, HISTORY_PLOT_PATH)

#%%
