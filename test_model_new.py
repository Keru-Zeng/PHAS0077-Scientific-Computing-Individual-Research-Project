# import useful libraries
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from keras.models import load_model
from model_functions import *
from premodel_functions import *

# set parameter
batch_size = 10
img_width = 128
img_height = 128
img_channel = 3
test_path = "./test_set/"
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
n_num = num(dirname="tifs_unanalysed/")

# build PLA test set
test_ids = next(os.walk(test_path))[1]

# To ensure that len of X_test is equal to len of sizes_test
X_test = np.zeros((n_num, img_height, img_width, img_channel), dtype=np.uint8)
sizes_test = []
sys.stdout.flush()
n = 0
for id_ in tqdm(test_ids, total=len(test_ids)):
    path = test_path + id_ + "/images/"
    dir_3 = os.listdir(path)
    k = len(dir_3)
    for i in range(0, k):
        img = imread(path + dir_3[i])[:, :, :img_channel]
        # append sizes of figures i.e. 256*256 pixel: [256,256]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode="constant", preserve_range=True)
        X_test[n] = img
        n += 1

# load the model and use it to make predicted results
model = load_model("model_new.h5", custom_objects={"accuracy": "accuracy"})
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_test_thresh = (preds_test > 0.8).astype(np.uint8)

# count the number of items in prediction
ID = 0
ID_dir = {}
for id_ in test_ids:
    dirs = os.listdir(test_path + id_ + "/images/")
    k = len(dirs)
    for i in range(0, k):
        ID_dir[ID] = dirs[i]
        ID += 1

# show the result of threshold prediction
for i in range(0, len(X_test)):
    plt.figure(figsize=(12, 12))
    plt.imshow(preds_test_thresh[i])
    plt.axis("off")
    plt.tight_layout()
    figure_save_path = "pred_set/"
    if not os.path.exists(figure_save_path):  # if directory not exist, create it
        os.makedirs(figure_save_path)
    plt.savefig(
        os.path.join(figure_save_path, "{}.png".format(ID_dir[i])),
        bbox_inches="tight",
        pad_inches=-0.1,
    )
    plt.close()

# show the cell segmentation in threshold prediction
path = "./pred_set/"
dirs = os.listdir(path)
kk = len(dirs)
for i in range(0, kk):
    name = path + str(dirs[i])
    sep_count_cells(filename=name)
    figure_save_path = "sep_set/"
    if not os.path.exists(figure_save_path):  # if directory not exist, create it
        os.makedirs(figure_save_path)
    plt.savefig(
        os.path.join(figure_save_path, "{}.png".format(dirs[i])),
        bbox_inches="tight",
        pad_inches=-0.1,
    )
    plt.close()
