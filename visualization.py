import os
import glob
import cv2

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap


OUTPUT_DIR = os.path.join(os.getcwd(), "output")
IMGS_DIR = "./images"
MASKS_DIR = "./masks"
DATA_ROOT ="/home/arnisha/Projects/landcover-segmentation"
IMG_PATHS = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
MASK_PATHS = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

IMAGE_SIZE = 512

labels_cmap = ListedColormap(['yellow', 'red', 'green', 'blue',"pink"])


def visualize_dataset(num_samples = 8, seed = 42,
                     w = 10, h = 10, nrows = 4, ncols = 4, save_title = None,
                     pad = 0.8, indices = None):
    """
    A function to visualize the images of the dataset along with their
    corresponding masks.
    """
    data_list = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
    if indices == None:
        np.random.seed(seed)
        indices = np.random.randint(low = 0, high = len(data_list),
                                   size = num_samples)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (h,w), nrows = num_samples//2,
                           ncols = 4)
    for i, idx in enumerate(indices):
        r,rem = divmod(i,2)
        img = cv2.imread(data_list[idx])/255
        mask_pt = data_list[indices[i]].split(".jpg")[0] + "_m.png"
        mask = cv2.imread(mask_pt)[:,:,1]
        ax[r,2*rem].imshow(img)
        ax[r,2*rem].set_title("Sample"+str(i+1))
        ax[r,2*rem+1].imshow(mask, cmap = labels_cmap, interpolation = None,
                            vmin = -0.5, vmax = 4.5)
        ax[r,2*rem+1].set_title("Mask" + str(i+1))
    #plt.suptitle("Samples of 512 x 512 images", fontsize = 10)
    plt.tight_layout(pad = 0.8)
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()
    

def visualize_tif(tif_name = None, save_title = None,
                  h = 12, w = 12, index = None):
    """
    A function to visualize the complete tif images.
    """
    if index is not None:
        img = cv2.imread(IMG_PATHS[index])/255
        mask = cv2.imread(MASK_PATHS[index])
    elif tif_name is not None:
        img = cv2.imread(os.path.join(IMGS_DIR, tif_name))/255
        mask = cv2.imread(os.path.join(MASKS_DIR, tif_name))
    labels = mask[:,:,0]
    del mask
    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (w,h),
                           nrows = 1, ncols =2)
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title("RGB Image")

    ax[1].imshow(labels, cmap=labels_cmap, interpolation = None,
                vmin = -0.5, vmax = 4.5)
    ax[1].axis("off")
    ax[1].set_title("Mask")
    plt.tight_layout(pad = 0.8)
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()

visualize_dataset(num_samples = 8, w = 12, h = 12, pad = 1.4,
                 save_title = "Visualize_dataset", indices = [0,1,17,20,29,5,6,7])
