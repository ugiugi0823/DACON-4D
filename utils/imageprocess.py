# imageprocess.py

import random
from torchvision import transforms
import torchvision.transforms.functional as TF




def image_transformer(input_image=None, train=True):
    """
    Using torchvision.transforms, make PIL image to tensor image
    with normalizing and flipping augmentations
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if train:
        transformer = transforms.Compose([
            transforms.Resize((224, 224)),       
            #RotateTransform([0, 0, 0, -90, 90, 180]),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            #transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            normalize,                        
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            normalize,
        ])

    transformed_image = transformer(input_image)
    
    return transformed_image
