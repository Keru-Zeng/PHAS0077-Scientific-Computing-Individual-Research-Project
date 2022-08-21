# import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from keras.preprocessing import image
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.models import Model

# count the number of pictures
def num(dirname=""):
    """
    count the number of the PLA data in the directory

    Parameters
    ----------
    dirname: string
        The dir of the files

    Returns
    ----------
    The total number of the images in the whole directory
    """

    if dirname != "":  # if dirname is not empty, use given dir
        dirname = dirname
    else:  # if dirname is empty
        dirname = "./" + dirname
    dirs = os.listdir(dirname)  # read all sub-folders' names in dirname folder
    n_files = len(dirs)
    n = 0
    for i in range(0, n_files):
        if dirs[i] != ".DS_Store":
            dataname = dirname + dirs[i] + "/"
            data = dataname + "tifs/"
            dir_1 = os.listdir(data)  # read all PLA pictures' names
            k = len(dir_1)
            for j in range(0, k):
                if dir_1[j] != ".DS_Store":
                    n += 1
    return n


# cell segmentation
# modified based on https://pyimagesearch.com/2015/11/02/watershed-opencv/
def sep_count_cells(filename=""):
    """
    separate the cells that overlapping with each other by using watershed and Otsu method and then plot them.

    Parameters
    ----------
    filename: string
        The name of the files

    """
    paths = filename
    image = cv2.imread(paths)
    # perform pyramid mean shift filtering
    shifted = cv2.pyrMeanShiftFiltering(image, 20, 55)
    # convert image to grayscale
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    # apply OTSU thresholding
    thresh = cv2.threshold(gray, 0, 254, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # calculate the binary pixel's Euclidean distance to the nearest zero pixel.
    d = ndimage.distance_transform_edt(thresh)
    # find peaks
    localMax = peak_local_max(d, indices=False, min_distance=50, labels=thresh)
    # perform connected component analysis on the local peaks
    marker = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    # appy Watershed method
    label = watershed(-d, marker, mask=thresh)
    plt.figure(figsize=(10, 10))
    # give different colors with label 0 and 1
    plt.imshow(label, cmap=plt.cm.nipy_spectral)
    plt.axis("off")


# do data augmentation
# modified based on Abhinav Sagar, "Nucleus Segmentation using U-Net",
# accessed from https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4
def data_aug(X_train, Y_train):
    """
    build data augmentation to avoid the overfitting of the model
    and also to increase the ability of model to generalize.

    Parameters
    ----------
    X_train: numpy array
        The training set of data
    Y_train: numpy array
        The binary classifier set of data, contains {0,1}
    Returns
    ----------
    The augmented data of x, y, x validation and y validation data sets
    """
    # Creating the training Image and Mask generator (zoom in range[0.8,1.2])
    image_datagen = image.ImageDataGenerator(
        shear_range=0.5,
        rotation_range=50,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="reflect",
    )
    mask_datagen = image.ImageDataGenerator(
        shear_range=0.5,
        rotation_range=50,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="reflect",
    )

    # Keep the same seed for image and mask generators so they fit together
    image_datagen.fit(X_train[: int(X_train.shape[0] * 0.9)], augment=True, seed=42)
    mask_datagen.fit(Y_train[: int(Y_train.shape[0] * 0.9)], augment=True, seed=42)

    # flow method generate batch of augmented data
    x = image_datagen.flow(
        X_train[: int(X_train.shape[0] * 0.9)], batch_size=10, shuffle=True, seed=42
    )
    y = mask_datagen.flow(
        Y_train[: int(Y_train.shape[0] * 0.9)], batch_size=10, shuffle=True, seed=42
    )

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()

    image_datagen_val.fit(X_train[int(X_train.shape[0] * 0.9) :], augment=True, seed=42)
    mask_datagen_val.fit(Y_train[int(Y_train.shape[0] * 0.9) :], augment=True, seed=42)

    x_val = image_datagen_val.flow(
        X_train[int(X_train.shape[0] * 0.9) :], batch_size=10, shuffle=True, seed=42
    )
    y_val = mask_datagen_val.flow(
        Y_train[int(Y_train.shape[0] * 0.9) :], batch_size=10, shuffle=True, seed=42
    )

    return x, y, x_val, y_val


# build U-net model
# modified based on Abhinav Sagar, "Nucleus Segmentation using U-Net",
# accessed from https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4
def trainU_net(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """
    build the U-net model

    Parameters
    ----------
    IMG_HEIGHT: number
        The input image height of parameter
    IMG_WIDTH: number
        The input image width of parameter
    IMG_CHANNELS: number
        The input image channels of parameter
    Returns
    ----------
    The build-in U-net model
    """

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    models = Model(inputs=[inputs], outputs=[outputs])

    return models
