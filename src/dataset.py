### IMPORTED LIBRARIES ########
#General libraries:
import numpy as np
import scipy as sp
import pandas as pd
import random
from PIL import Image
import os
import time
import cv2 

#For Neural nets:
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler


class MicroStructDatasetTruss(Dataset):
    def __init__(self,  root_dir, ed, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.ed = ed
                
    def __len__(self):
        return self.ed#len(data_params_all)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = "sample_" + str(idx)+".png"
        img_path = self.root_dir + "/" + img_name
        image = Image.open(img_path).convert('RGB')
               
        if self.transform:
            image = self.transform(image)    

        sample = image
        return sample