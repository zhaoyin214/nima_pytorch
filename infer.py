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
import pandas as pd
import torch
import time
from tqdm import tqdm

from configs import TEST_PHASE_LIST, TRAIN, TEST, \
    MODEL_SAVE_PATH, TORCH_FLOAT
from dataset import AVADataset

#%%
def infer(model, image, transforms, device="cpu"):
    """
    """

    model.to(device)
    model.eval()

    since = time.time()

    # inference
    predictions = pd.DataFrame(columns=AVADataset._label_key_ratings)

    # instances
    image = transforms(image).type(TORCH_FLOAT).to(device)

    # forward
    with torch.set_grad_enabled(mode=(phase == TRAIN)):

        batch_predicitions = model(batch_images).type(TORCH_FLOAT)

        batch_predicitions = pd.DataFrame(
            data=batch_predicitions.cpu().numpy(),
            columns=AVADataset._label_key_ratings
        )
        prediction = prediction.append(
            batch_predicitions, ignore_index=True
        )

    predictions.index.name = AVADataset._label_key_id

    time_elapsed = time.time() - since
    print("training complete in {:.0f}m, {:.0f}s" \
        .format(time_elapsed // 60, time_elapsed % 60))

    return prediction

#%%
if __name__ == "__main__":

    import os
    import pickle
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from configs import DEVICE, NUM_EPOCHS, \
        IMAGE_SIZE, \
        DATASET_CONFIG, DATA_LOADER_CONFIG, \
        MODEL_LOAD_PATH, EVAL_LOSS_PATH, EVAL_PRED_PATH

    from dataset import AVADataset, NIMADataset
    from utils import plot_batch, plot_prediction, ave_rating
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

    # -- evaluation --
    predictions, loss = eval(
        model=nima_net, dataloaders=nima_dataloaders,
        criterion=criterion, device=DEVICE,
    )
    for phase in TEST_PHASE_LIST:
        predictions[phase].insert(
            loc=0, column=AVADataset._label_key_image_id,
            value=ava_datasets[phase].labels[AVADataset._label_key_image_id]
        )

    with open(EVAL_PRED_PATH, "wb") as fw:
        pickle.dump(predictions, fw)
    with open(EVAL_LOSS_PATH, "wb") as fw:
        pickle.dump(loss, fw)

    image = ava_datasets[TEST][0]["image"]
    ground_truth = ava_datasets[TEST][0]["ground_truth"]
    prediction = predictions[TEST].loc[0, AVADataset._label_key_ratings]
    plot_prediction(
        image=image, ground_truth=ground_truth, prediction=prediction
    )

    print("predicted average score: {}".format(ave_rating(prediction)))
    print("ground truth score: {}".format(ave_rating(ground_truth)))

