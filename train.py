import torch
from torch.utils.data import DataLoader
from utils import SegmentationDataset
import numpy as np
import albumentations as A
from utils import training_loop
#from utils import UNet
from unet import UNet
import segmentation_models_pytorch as smp


device = "cuda" if torch.cuda.is_available() else "cpu"
target_names = np.array(["background", "building", "woodland", "water", "road"])

# Loss function - Mean IoU loss
loss_fn = smp.losses.JaccardLoss(mode = "multiclass",
                                classes = 5).to(device)

# Hyperparameters
batch_size = 8
epochs = 80
lr = 5e-5

# Configuring the set of transformations
transforms = A.Compose([
    A.OneOf([
        A.HueSaturationValue(40,40,30,p=1),
        A.RandomBrightnessContrast(p=1,brightness_limit = 0.2,
                                  contrast_limit = 0.5)], p = 0.5),
    A.OneOf([
        A.RandomRotate90(p=1),
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop(min_max_height=(248,512),height=512,width=512, p =1)
    ], p = 0.5)])

# Preparing datasets and DataLoaders
train_set = SegmentationDataset(mode = "train", transforms = transforms,
                               ratio = 0.6)
val_set = SegmentationDataset(mode = "val", ratio = 0.7)

train_dloader = DataLoader(train_set, batch_size = batch_size,
                           shuffle = True, num_workers = 2)
val_dloader = DataLoader(val_set, batch_size=batch_size, num_workers = 2)

model = UNet(in_channels = 3, out_channels = 5).to(device)

#Train
training_loop(model, train_dloader, val_dloader, epochs, lr, loss_fn, mod_epochs =1,
             regularization = "L2", reg_lambda = 1e-6, early_stopping = True,
             patience = 4, verbose = True, model_title = "Vanilla UNet", save = True,
             stopping_criterion = "loss")
