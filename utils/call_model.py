# call_model.py
import os
import torch
import logging

from torch import nn
from src.model import *


# from torchvision.models import resnet50, ResNet50_Weights, resnet18,  resnet152
# from efficientnet_pytorch import EfficientNet




class CallModel():
    def __init__(self, model_type=None, pretrained=True, logger=None, path='/content/drive/MyDrive/pretrained_model'):
        
        # MODEL TYPE
        if model_type == 'resnet50':
            base_model = PreResnet50()
            weight_path = os.path.join(path, 'resnet50-IMAGENET1K_V2.pth')
              
        elif model_type == 'efficientnetb7':
            base_model = PreEfficientnetB7()
            weight_path = os.path.join(path, 'efficientnet-b7.pth')
            
        elif model_type == 'efficientnetb1':
            base_model = PreEfficientnetB1()
            weight_path = os.path.join(path, 'efficientnet-b1.pth')
            
        elif model_type == 'efficientnetb2':
            base_model = PreEfficientnetB2()
            weight_path = os.path.join(path, 'efficientnet-b2.pth')
            
        else:
            raise Exception(f"No such model type: {model_type}")
        
        
        # LOAD PRETRAINED WEIGHTS
        if pretrained:
            logger.info(f"Using pretrained model. Loading weights from {weight_path}")
            base_model = CallModel._load_weights(base_model, weight_path)
            
            # b5 model
            nn.init.xavier_normal_(base_model.block[0].fc.weight)
           
        else:
            logger.info(f"Not using pretrained model.")
        
        self.model = base_model
            
    def model_return(self):
        return self.model
        
    @staticmethod
    def _load_weights(model, path):
        model.load_state_dict(torch.load(path), strict=False)
        return model
