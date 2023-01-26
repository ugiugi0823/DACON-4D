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

import torchvision.transforms.functional as TF
# from src.train import train_model
# from utils.imageprocess import image_transformer, image_processor
# from utils.EarlyStopping import EarlyStopping
# from utils.dataloader import CustomDataLoader
# from utils.radams import RAdam
# from utils.call_model import CallModel

from tqdm import tqdm
import logging

# imageprocess.py

class RotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        if isinstance(self.angles, list):
            angle = random.choice(self.angles)
        else:
            angle = self.angles
        return TF.rotate(x, angle)


def image_transformer(input_image=None, train=True):
    """
    Using torchvision.transforms, make PIL image to tensor image
    with normalizing and flipping augmentations
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    if train:
        transformer = transforms.Compose([ 
            transforms.Resize((224, 224)),       
            RotateTransform([0, 0, 0, -90, 90, 180]),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            #transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    transformed_image = transformer(input_image)
    
    return transformed_image



def tta_transformer(input_image, angle):
    """
    Test Time Augmentation for creating final test labels.
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transformer = transforms.Compose([       
        transforms.Resize((224, 224)), 
        RotateTransform(angle),
        transforms.ToTensor(),
        normalize,
    ])

    transformed_image = transformer(input_image)
    
    return transformed_image
