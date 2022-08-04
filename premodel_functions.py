# import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
import pims

# show the 4D PLA and load data
def stackloader(filename, dir_in="", plot=True):
    """
    load and seperate 4D PLA pictures into 20 2D figures and then plot them if needed

    Parameters
    ----------
    filename: string
        The name of the files
    dir_in: string
        Specific directory of the files, defaults to empty
    plot: boolean
        True to plot the figures, False only to load the data rather than plot the figures

    Returns
    -------
    There are 4 returns: value of nuclei and nps as well as number of rows and columns
    """
    # load data
    if dir_in == "":
        data = pims.TiffStack(
            "./" + filename
        )  # if the tif in the same folder as the code does.
    else:
        data = pims.TiffStack(
            dir_in + filename
        )  # if the dir is not empty, use the dir to open the tif
    nuclei = np.array(data[1::2])  # start from 1 with step 2, i.e.1,3,5,7,9...
    nps = np.array(data[::2])  # start from 0 with step size 2, i.e. 0,2,4,6,8...
    z_size, y_size, x_size = np.shape(nuclei)  # shape the nuclei data (a square matrix)
    # print(np.shape(data)) #(40,512,521),512 entries in x y axes,with 20 in Z and 2 channel,total 40
    nrows = np.int(np.ceil(np.sqrt(z_size)))
    ncols = np.int(z_size // nrows + 1)
    if plot == True:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        for n in range(z_size):
            i = n // ncols
            j = n % ncols
            axes[i, j].imshow(
                nuclei[n], interpolation="nearest", cmap="gray"
            )  # total 20 necleis,total 20 pictures
        for (
            ax
        ) in (
            axes.ravel()
        ):  # returns contiguous flattened array(ravel() return an array to 1D)
            if not (len(ax.images)):
                fig.delaxes(ax)  # remove the Axes ax from its figure.
        fig.tight_layout()
        plt.show()
        plt.close()  # reduce memory after function.

    return nuclei, nps, nrows, ncols


# Convert PLA data into 2D picture and plot it
def plotpic(filename, dir_in="", plot=True):
    """
    convert 4D PLA images into 2D figures and then plot them

    Parameters
    ----------
    filename: string
        The name of the files
    dir_in: string
        Specific directory of the files, defaults to empty
    plot: boolean
        True to plot the figures
    """

    if dir_in == "":
        data = "./" + filename  # if PLA data in the same folder as the code does.
    else:
        data = (
            dir_in + filename
        )  # if directory not empty, use directory to open the picture
    IM = io.imread(data)[10]
    IM_MAX = np.max(IM, axis=0)
    IM_MAX = resize(IM_MAX, (512, 512), mode="constant", preserve_range=True)
    if plot == True:
        plt.imshow(IM_MAX, cmap="gray")
        plt.axis("off")


# In folders, change all PLA into 2D pictures and save them.
def filefolder(dirname="", plot=True):
    """
    load, plot and save 4D PLA pictures into 2D figures in given folders

    Parameters
    ----------
    dirname: string
        General directory of the file, defaults to empty
    plot: boolean
        True to plot and save the figures in a given folder
    """

    if dirname != "":  # if dirname is not empty, use given dir
        dirname = dirname
    else:  # if dirname is empty
        dirname = "./" + dirname
    dirs = os.listdir(dirname)  # read all sub-folders' names in dirname folder
    n_files = len(dirs)
    for i in range(0, n_files):
        if dirs[i] != ".DS_Store":
            dataname = dirname + dirs[i] + "/"
            data = dataname + "tifs/"
            dir_1 = os.listdir(data)  # read all PLA pictures' names
            k = len(dir_1)
            for j in range(0, k):
                if dir_1[j] != ".DS_Store":  # check whether it is tifs folders
                    name = data + dir_1[j]
                    IM = io.imread(name)[0]
                    IM_MAX = np.max(IM, axis=0)
                    IM_MAX = resize(
                        IM_MAX, (512, 512), mode="constant", preserve_range=True
                    )
                    if plot == True:
                        plt.imshow(IM_MAX, cmap="gray")
                        plt.axis("off")
                        figure_save_path = (
                            "test_set/" + dirs[i] + "/images/"
                        )  # dir to save pictures
                        if not os.path.exists(
                            figure_save_path
                        ):  # if directory not exist, create it
                            os.makedirs(figure_save_path)
                        plt.savefig(
                            os.path.join(figure_save_path, "{}".format(dir_1[j])),
                            bbox_inches="tight",
                            pad_inches=-0.1,
                        )
                        plt.close()  # close figures, so the plot will not display


# binarize the images in train set
def improve_reso(k):
    """
    In order to do a binary classification with train set data, convert all values in images above 0 to 1.
    So, picture only have two values {0,1}, which have more accurate resolution.

    Parameters
    ----------
    k: number
        input a iterate number
    """
    path = "./train_set/"
    dirs = os.listdir(path)  # walk through all files in path
    n = len(dirs)
    if k < n:
        N = k
    else:
        N = n
    for i in range(0, N):  # walk through specified number of files
        name = path + dirs[i] + "/masks/"
        dirss = os.listdir(name)
        nn = len(dirss)
        for j in range(0, nn):
            names = name + dirss[j]
            mask = io.imread(names)  # read images
            # need to binarize the images, convert all values above 0 to 1 to assign a pixel value of 1 for class
            masked = np.where(mask > 0, 1, mask)
            plt.imshow(masked, cmap="gray")
            plt.axis("off")
            figure_save_path = "train_set/" + dirs[i] + "/masks/"
            plt.savefig(
                os.path.join(figure_save_path, "{}".format(dirss[j])),
                bbox_inches="tight",
                pad_inches=-0.1,
            )
            plt.close()  # reduce memory
