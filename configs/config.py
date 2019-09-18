import os
import torch

# -- gpu -- #
# IS_GPU = False
IS_GPU = True
if IS_GPU:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")

# -- dataset -- #
AVA_DATASET_ROOT = "e:/src/jupyter/datasets/AVA"
# image directory
AVA_IMAGE_ROOT = os.path.join(AVA_DATASET_ROOT, "images")
# labels
AVA_LABEL_FILEPATH = os.path.join(AVA_DATASET_ROOT, "AVA.txt")
AVA_ABNORMAL = os.path.join(AVA_DATASET_ROOT, "abnormals.csv")
AVA_LABEL_CLEANED_FILEPATH = os.path.join(AVA_DATASET_ROOT, "AVA.csv")
# image size
TRAIN_RESIZE = [256, 256]
IMAGE_SIZE = [224, 224]
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# train, val, test split
TEST_RATIO = 0.33
VAL_RATIO = 0.33
AVA_TRAIN_LABEL_FILEPATH = "./data/ava_train.csv"
AVA_VAL_LABEL_FILEPATH = "./data/ava_val.csv"
AVA_TEST_LABEL_FILEPATH = "./data/ava_test.csv"

# -- batch -- #
NUM_EPOCHS = 1024
TRAINING_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 32
NUM_WORKERS = 4

# -- data generator -- #
# dataset
DATASET_CONFIG = {
    "dataset": "AVA",
    "image_root": AVA_IMAGE_ROOT,
    "train_label_filepath": AVA_TRAIN_LABEL_FILEPATH,
    "val_label_filepath": AVA_VAL_LABEL_FILEPATH,
    "test_label_filepath": AVA_TEST_LABEL_FILEPATH,
}
# data loader
DATA_LOADER_CONFIG = {
    "train_batch_size": TRAINING_BATCH_SIZE,
    "train_shuffle": True,
    "train_num_workers": NUM_WORKERS,
    "val_batch_size": VAL_BATCH_SIZE,
    "val_shuffle": False,
    "val_num_workers": NUM_WORKERS,
    "test_batch_size": TEST_BATCH_SIZE,
    "test_shuffle": False,
    "test_num_workers": NUM_WORKERS,
}

# -- training --#
MODEL_VGG16_BN_PATH = "./data/vgg16_bn-6c64b313.pth"
MODEL_SAVE_PATH = "./output/best_model-epoch_{}-train_loss_{:.4f}-val_loss_{:.4f}.pth"
MODEL_LOAD_PATH = "./data/best_model.pth"
HISTORY_PATH = "./output/history.pickle"
HISTORY_PLOT_PATH = "./output/history.png"
# -- evaluation --
EVAL_PRED_PATH = "./output/test_predictions.pickle"
EVAL_LOSS_PATH = "./output/test_loss.pickle"

# -- optimizer -- #
BASELINE_LEARNING_RATE = 3e-7
LEARNING_RATE = 3e-6
MOMENTUM = 0.5
BETAS = [0.9, 0.999]
WEIGHT_DECAY = 9
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_ON_PLATEAU = {"patience": 5, "factor": 0.2}
LR_SCHEDULE_STEP_SIZE = 10
LR_SCHEDULE_GAMMA = 0.5