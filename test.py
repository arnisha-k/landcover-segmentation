from utils import segmentation_test_loop
from utils import class_report
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import SegmentationDataset
from unet import UNet
from utils import visualize_preds


device = "cuda" if torch.cuda.is_available() else "cpu"
test_set = SegmentationDataset(mode = "test")

test_dloader = DataLoader(test_set, batch_size = 8, num_workers = 2)
target_names = np.array(["background", "building", "woodland", "water", "road"])
model = UNet(in_channels = 3, out_channels = 5).to(device)

#Load saved checkpoint
checkpoint_path = './checkpoint/checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
#print(checkpoint.keys())
model.load_state_dict(checkpoint)

scores, acc, jaccard, class_probs = segmentation_test_loop(model = model, test_loader = test_dloader,
                                        device = device)

class_report(target_names, scores, acc, jaccard, class_probs)

#Visualization

visualize_preds(model, test_set, save_title = "output", h = 12, w = 12, indices = [957,961,1476,1578])