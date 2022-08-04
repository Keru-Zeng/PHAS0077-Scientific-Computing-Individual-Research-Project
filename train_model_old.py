# These codes are modified based on Abhinav Sagar, "Nucleus Segmentation using U-Net",
# accessed from https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4

# import useful libraries
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_functions import *
from premodel_functions import *

# set parameter
batch_size = 10
img_width = 128
img_height = 128
img_channel = 3
train_path = "./stage1_train/"
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

# load all files' name into training directory
train_ids = next(os.walk(train_path))[1]

# build X and Y train set
X_train = np.zeros((len(train_ids), img_height, img_width, img_channel), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)
# output on the same line with time interval
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = train_path + id_
    img = imread(path + "/images/" + id_ + ".png")[:, :, :img_channel]
    # make pictures from 256*256 to 128*128 to increase computational speed
    img = resize(img, (img_height, img_width), mode="constant", preserve_range=True)
    X_train[n] = img
    mask1 = np.zeros((img_height, img_width, 1), dtype=np.bool)
    for masks in next(os.walk(path + "/masks/"))[2]:
        mask = imread(path + "/masks/" + masks)
        # Compress the picture to form one picture
        mask = np.expand_dims(
            resize(mask, (img_height, img_width), mode="constant", preserve_range=True),
            axis=-1,
        )
        mask1 = np.maximum(mask, mask1)
    Y_train[n] = mask1

# data augmentation and generate train and validation set
x, y, x_val, y_val = data_aug(X_train, Y_train)
train_generator = zip(x, y)
vali_generator = zip(x_val, y_val)

# build model
model = trainU_net(img_height, img_width, img_channel)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# fit the model
# Stop training when a monitored metric has stopped improving.
earlystopper = EarlyStopping(patience=2, verbose=1)
# save a model or weights (in a checkpoint file) at some interval,
# so the model or weights can be loaded later to continue the training from the state saved.
checkpointer = ModelCheckpoint("model_old.h5", verbose=1, save_best_only=True)
# Use fit_generator(), since there are data argumentations (default batch size is 32)
results = model.fit_generator(
    train_generator,
    validation_data=vali_generator,
    validation_steps=10,
    steps_per_epoch=100,
    epochs=20,
    callbacks=[earlystopper, checkpointer],
)
results.history

# save loss and accuracy
mse = np.array((results.history["loss"]))
val_mse = np.array((results.history["val_loss"]))
# np.save('loss.npy', mse)
# np.save('val_loss.npy', val_mse)
acc = np.array((results.history["accuracy"]))
val_acc = np.array((results.history["val_accuracy"]))
# np.save('acc.npy', acc)
# np.save('val_acc.npy', val_acc)

# summarize history and plot the relevant data
plt.figure(figsize=(12, 12))
plt.subplot(211)
plt.plot(results.history["accuracy"])
plt.plot(results.history["val_accuracy"])
plt.title("model accuracy", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.xlabel("epoch", fontsize=14)
plt.legend(["train_set", "vali_set"], loc="upper right", fontsize=14)
plt.subplot(212)
plt.plot(results.history["loss"])
plt.plot(results.history["val_loss"])
plt.title("model loss", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.xlabel("epoch", fontsize=14)
plt.legend(["train_set", "vali_set"], loc="upper right", fontsize=14)
plt.show()
plt.savefig("summary_old.png")
