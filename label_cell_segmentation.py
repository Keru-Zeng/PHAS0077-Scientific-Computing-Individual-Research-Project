# modified based on https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html
# import useful libraries
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.morphology import label
import matplotlib.patches as mpatches
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from functions import *

# load test path and test items
test_path = "./test_set/"
test_ids = next(os.walk(test_path))[1]

# traditional method to label the cell segmentation
for id_ in tqdm(test_ids, total=len(test_ids)):
    path = test_path + id_ + "/images/"
    dir_3 = os.listdir(path)
    k = len(dir_3)
    for i in range(0, k):
        image = imread(path + dir_3[i])[:, :, :3]  # load the pictures
        image = rgb2gray(image)  # convert to greyscale after loading
        thresh = threshold_otsu(image)  # perform automatic thresholding
        bw = closing(image > thresh, square(3))  # convert 0，1 to bool
        # remove artifacts connected to image border
        cleared = clear_border(bw)
        # label image regions
        label_image = label(cleared)
        # make the background transparent
        image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 150:
                # draw rectangle around segment cells
                minr, minl, maxr, maxl = region.bbox
                rect = mpatches.Rectangle(
                    (minl, minr),
                    maxl - minl,
                    maxr - minr,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
                ax.add_patch(rect)
        ax.set_axis_off()
        plt.tight_layout()
        figure_save_path = "label_set/" + id_
        if not os.path.exists(figure_save_path):  # if directory not exist, create it
            os.makedirs(figure_save_path)
        plt.savefig(
            os.path.join(figure_save_path, "{}".format(dir_3[i])),
            bbox_inches="tight",
            pad_inches=-0.1,
        )
        plt.close()  # reduce memory
