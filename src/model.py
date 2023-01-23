# model.py
import os
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import KFold
import random
from time import time
import IPython
import copy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from warmup_scheduler import GradualWarmupScheduler

# from src.train import train_model
# from utils.imageprocess import image_transformer, image_processor
# from utils.EarlyStopping import EarlyStopping
# from utils.dataloader import CustomDataLoader
# from utils.radams import RAdam
# from utils.call_model import CallModel

from tqdm import tqdm
import logging


## 2
#from torchvision import models
#resnet18_pretrained = models.resnet18(pretrained=True)

##3
from torchvision.models import resnet18
from efficientnet_pytorch import EfficientNet

"""
Here every model to be used for pretraining/training is defined.
"""
class PreResnet18(nn.Module):
    def __init__(self):
        super(PreResnet18, self).__init__()
        
        base_model = resnet18()
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 10),
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

    
class PreEfficientnetB0(nn.Module):
    def __init__(self):
        super(PreEfficientnetB0, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PreEfficientnetB1(nn.Module):
    def __init__(self):
        super(PreEfficientnetB1, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=10)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PreEfficientnetB2(nn.Module):
    def __init__(self):
        super(PreEfficientnetB2, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=10)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out
