import os
import glob
import cv2

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset


import numpy as np
import shutil
import random
import time
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
IMGS_DIR = "./images"
MASKS_DIR = "./masks"
DATA_ROOT = "./"
IMG_PATHS = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
MASK_PATHS = glob.glob(os.path.join(MASKS_DIR, "*.tif"))
import torchmetrics
from torch.optim import Adam
from matplotlib.colors import ListedColormap

IMAGE_SIZE = 512
labels_cmap = ListedColormap(['yellow', 'red', 'green', 'blue',"pink"])

class SegmentationDataset(Dataset):
    """
    The main class that handles the dataset. Reads the images from
    OUTPUT_DIR, handles the data augmentation transformations and converts
    the numpy images to tensors.
    """
    def __init__(self, mode = "train", ratio = None, transforms = None, seed = 42):
        self.mode = mode
        self.transforms = transforms
        self.output_dir = OUTPUT_DIR
        self.data_root = DATA_ROOT
        
        if mode in ["train", "test", "val"]:
            with open(os.path.join(self.data_root, self.mode + ".txt")) as f:
                self.img_names = f.read().splitlines()
                if ratio is not None:
                    print(f"Using the {100*ratio:.2f}% of the initial {mode} set --> {int(ratio*len(self.img_names))}|{len(self.img_names)}")
                    np.random.seed(seed)
                    self.indices = np.random.randint(low = 0, high = len(self.img_names),
                                             size = int(ratio*len(self.img_names)))
                else:
                    print(f"Using the whole {mode} set --> {len(self.img_names)}")
                    self.indices = list(range(len(self.img_names)))
        else:
            raise ValueError(f"mode should be either train, val or test ... not {self.mode}.")
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        if self.transforms is None:
            img = np.transpose(cv2.imread(os.path.join(self.output_dir, self.img_names[self.indices[item]] + ".jpg")),(2,0,1))
            mask = cv2.imread(os.path.join(self.output_dir, self.img_names[self.indices[item]] + "_m.png"))
            label = mask[:,:,1]
        else:
            img = cv2.imread(os.path.join(self.output_dir, self.img_names[self.indices[item]] + ".jpg"))
            mask = cv2.imread(os.path.join(self.output_dir, self.img_names[self.indices[item]] + "_m.png"))
            label = mask[:,:,1]
            transformed = self.transforms(image = img, mask = label)
            img = np.transpose(transformed["image"], (2,0,1))
            label = transformed["mask"]
        del mask
        return torch.tensor(img, dtype = torch.float32)/255, torch.tensor(label, dtype = torch.int64)

 

#################### Early Stopping ######################

# Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
##########################################################
##########################################################

# Training loop


def training_loop(model, train_loader, val_loader, epochs,
                  lr, loss_fn, regularization=None,
                  reg_lambda=None, mod_epochs=5, early_stopping = False,
                  patience = None, verbose = None, model_title = None, save = None,
                 stopping_criterion = "loss"):
    if stopping_criterion not in ["loss","IoU"]:
        raise ValueError(f"stopping criterion should be either 'loss' or 'IoU', not {stopping_criterion}.")
    print("Training of " + model_title + " starts!")
    print(f"Using {stopping_criterion} as stopping criterion.\n")
    tic = time.time()
    
    optim = Adam(model.parameters(), lr=lr)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    if stopping_criterion == "IoU":
        jaccard = torchmetrics.JaccardIndex(num_classes = 5).to(device)
        iou_loss_list = []

    train_loss_list = []
    val_loss_list = []
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    counter_epochs = 0

    if early_stopping:
        ear_stopping = EarlyStopping(patience= patience, verbose=verbose)

    for epoch in range(epochs):
        counter_epochs+=1
        model.train()
        train_loss, val_loss = 0.0, 0.0
        if stopping_criterion == "IoU": iou_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc = "Training Progress")
        for train_batch in train_loader_tqdm:
            X, y = train_batch[0].to(device), train_batch[1].to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            train_loss += loss.item()
            if stopping_criterion == "IoU":
                iou_loss += 1- float(jaccard(preds, y).cpu())

            # Regulirization
            if regularization == 'L2':
                l_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + reg_lambda * l_norm
            elif regularization == 'L1':
                l_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + reg_lambda * l_norm

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                X, y = val_batch[0].to(device), val_batch[1].to(device)
                preds = model(X)
                if stopping_criterion == "loss":
                    val_loss += loss_fn(preds, y).item()
                else:
                    # Calculate the 1-IoU as validation loss
                    val_loss += 1-float(jaccard(preds,y).cpu())
        train_loss /= num_train_batches
        val_loss /= num_val_batches
        if stopping_criterion == "IoU":
            iou_loss /= num_train_batches
            iou_loss_list.append(iou_loss)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if (epoch + 1) % mod_epochs == 0:
            if stopping_criterion == "loss":
                print(
                    f"Epoch: {epoch + 1}/{epochs}{5 * ' '}Training Loss: {train_loss:.4f}{5 * ' '}Validation Loss: {val_loss:.4f}")
            else:
                print(
                    f"Epoch: {epoch + 1}/{epochs}{5 * ' '}Training IoU Loss: {iou_loss:.4f}{5 * ' '}Validation IoU Loss: {val_loss:.4f}{5*' '}Training Loss: {train_loss:.4f}")

        if early_stopping:
            ear_stopping(val_loss, model)
            if ear_stopping.early_stop:
                print("Early stopping")
                break
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(8, 8))
    if stopping_criterion == "loss":
        ax.plot(range(1, counter_epochs + 1), train_loss_list, label='Train Loss',
               color = "#808080", linewidth = 2.5)
    else:
        ax.plot(range(1, counter_epochs + 1), iou_loss_list, label='Train IoU Loss',
               color = "#808080", linewidth = 2.5)
    ax.plot(range(1, counter_epochs + 1), val_loss_list, label='Val Loss',
            color = "#36454F", linewidth = 2.5)
    ax.set_title(model_title, fontsize = 15)
    ax.set_ylabel("Loss", fontsize = 13)
    ax.set_xlabel("Epochs", fontsize = 13)
    plt.legend()
    if save is not None:
        plt.savefig(model_title + ".png")
    plt.show()

    if early_stopping:
        model.load_state_dict(torch.load("checkpoint.pt"))
    total_time = time.time() - tic
    mins, secs = divmod(total_time, 60)
    if mins < 60:
        print(f"\n Training completed in {mins} m {secs:.2f} s.")
    else:
        hours, mins = divmod(mins, 60)
        print(f"\n Training completed in {hours} h {mins} m {secs:.2f} s.")
        
# Test loop
        
def segmentation_test_loop(model, test_loader, device = "cpu"):
    stat_scores = torchmetrics.StatScores(task = "multiclass",reduce = "macro", num_classes = 5, average = None,
                            mdmc_reduce = "global").to(device)
    acc = torchmetrics.Accuracy(task = "multiclass",num_classes = 5, average = "micro",
                   mdmc_average = "global").to(device)
    jaccard = torchmetrics.JaccardIndex(task = "multiclass", num_classes = 5).to(device)
    
    model.eval()

    class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4:0}
    num_samples = {0:0, 1:0, 2:0, 3:0, 4:0}

    for X,y in test_loader:
        X = X.to(device)
        #y = val_sample[1].cpu().numpy().flatten()
        y = y.to(device)
        #targets_list = np.concatenate((targets_list, y))

        with torch.no_grad():
            logits = F.softmax(model(X), dim =1)
            aggr = torch.max(logits, dim = 1)
            #preds = aggr[1].cpu().numpy().flatten()
            preds = aggr[1]
            probs = aggr[0]
            for label in class_probs.keys():
                class_probs[label]+= probs[preds == label].flatten().sum()
                num_samples[label]+= preds[preds == label].flatten().size(dim = 0)
            #predictions_list = np.concatenate((predictions_list, preds))
            stat_scores.update(preds, y)
            acc.update(preds,y)
            jaccard.update(preds, y)
    for label in class_probs.keys():
        class_probs[label] /= num_samples[label]
    return stat_scores.compute(), acc.compute(), jaccard.compute(), class_probs

# Custom Implementation of FocalLoss

class FocalLoss(nn.Module):
    def __init__(self, weight = None, gamma = 2, reduction = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim = 1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1-prob)**self.gamma)*log_prob,
            targets,
            weight = self.weight,
            reduction = self.reduction
        )
    
    
def class_report(classes, scores, acc, jaccard, class_probs):
    print(f"{10*' '}precision{10*' '}recall{10*' '}f1-score{10*' '}support\n")
    acc = float(acc.cpu())
    jaccard = float(jaccard.cpu())
    for i,target in enumerate(classes):
        precision = float((scores[i,0]/(scores[i,0]+scores[i,1])).cpu())
        recall = float((scores[i,0]/(scores[i,0]+scores[i,3])).cpu())
        #TP, FP, TN, FN, total = scores[i].cpu().numpy()
        #precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        #recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2*precision*recall)/(precision+recall)
        print(f"{target}{10*' '}{precision:.2f}{10*' '}{recall:.2f}{10*' '}{f1:.2f}{10*' '}{scores[i,4]}")

    print(f"\n- Total accuracy:{acc:.4f}\n")
    print(f"- Mean IoU: {jaccard:.4f}\n")
    print("- Class probs")
    for idx in class_probs.keys():
        print(f"{classes[idx]}:{class_probs[idx].cpu():.3f}")


def visualize_preds(model, train_set, title ="", num_samples = 4, seed = 42,
                    w = 10, h = 10, save_title = None, indices = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    if indices == None:
        indices = np.random.randint(low = 0, high = len(train_set),
                                    size = num_samples)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (w,h),
                           nrows = num_samples, ncols = 3)
    model.eval()
    for i,idx in enumerate(indices):
        X,y = train_set[idx]
        X_dash = X[None,:,:,:].to(device)
        preds = torch.argmax(model(X_dash), dim = 1)
        preds = torch.squeeze(preds).detach().cpu().numpy()

        ax[i,0].imshow(np.transpose(X.cpu(), (2,1,0)))
        ax[i,0].set_title("True Image")
        ax[i,0].axis("off")
        ax[i,1].imshow(y, cmap = labels_cmap, interpolation = None,
                      vmin = -0.5, vmax = 4.5)
        ax[i,1].set_title("Labels")
        ax[i,1].axis("off")
        ax[i,2].imshow(preds, cmap = labels_cmap, interpolation = None,
                      vmin = -0.5, vmax = 4.5)
        ax[i,2].set_title("Predictions")
        ax[i,2].axis("off")
    #fig.suptitle(title, fontsize = 20)
    plt.tight_layout()
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()