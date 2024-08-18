import os
import glob
import cv2

import matplotlib.colors



import numpy as np
import shutil
import time
from pathlib import Path


OUTPUT_DIR = os.path.join(os.getcwd(), "output")
IMGS_DIR = "./images"
MASKS_DIR = "./masks"
DATA_ROOT = "/home/arnisha/Projects/landcover-segmentation"
IMG_PATHS = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
MASK_PATHS = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

IMAGE_SIZE = 512

def split_images(TARGET_SIZE = IMAGE_SIZE):
    """
    A function to split the aerial images into squared images of
    size equal to TARGET_SIZE. Stores the new images into
    a directory named output, located in working directory.
    """
    tic = time.time()
    print(f"Splitting the images...\n")
    img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
    mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

    img_paths.sort()
    mask_paths.sort()
    
    if Path(OUTPUT_DIR).exists() and Path(OUTPUT_DIR).is_dir():
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], TARGET_SIZE):
            for x in range(0, img.shape[1], TARGET_SIZE):
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

                if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                    out_img_path = os.path.join(OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k))
                    cv2.imwrite(out_mask_path, mask_tile)

                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
    mins,sec = divmod(time.time()-tic,60)
    print(f"Execution completed in {mins} minutes and {sec:.2f} seconds.")



IMAGE_SIZE = 512
split_images(TARGET_SIZE = IMAGE_SIZE)

