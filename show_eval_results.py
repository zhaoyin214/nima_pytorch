#%%
import numpy as np
import pickle

from configs import EVAL_PRED_PATH, TEST, DATASET_CONFIG
from dataset import AVADataset
from utils import ave_rating, plot_prediction

with open(EVAL_PRED_PATH, "rb") as fr:
    predictions = pickle.load(fr)[TEST]

# ava dataset
ava_dataset = AVADataset(
    image_root=DATASET_CONFIG["image_root"],
    label_filepath=DATASET_CONFIG[TEST + "_label_filepath"]
)

#%%
# instance
index = np.random.randint(low=0, high=len(ava_dataset))
image = ava_dataset[index]["image"]
ground_truth = ava_dataset[index]["ground_truth"]
image_id = predictions.loc[index, AVADataset._label_key_image_id]
prediction = predictions.loc[index, AVADataset._label_key_ratings]

plot_prediction(
    image=image, ground_truth=ground_truth, prediction=prediction
)

print("image id: {}".format(predictions.loc[index, "image_id"]))
print("predicted average score: {}".format(ave_rating(prediction)))
print("ground truth score: {}".format(ave_rating(ground_truth)))

#%%
