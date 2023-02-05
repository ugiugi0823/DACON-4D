# inference.py

import os
import cv2
import copy
import torch

import pandas as pd
import argparse
import ttach as tta
import albumentations as A

from tqdm import tqdm
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from glob import glob
from datetime import datetime
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet












class PreResnet50(nn.Module):
    def __init__(self):
        super(PreResnet50, self).__init__()
        
        base_model = resnet50()
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 10),
        )
        
        #nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_df, transforms):        
        self.image_folder = image_folder   
        self.label_df = label_df
        self.transforms = transforms

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):        
        image_fn = self.image_folder +\
            str(self.label_df.iloc[index,0]).zfill(5) + '.jpg'
                                              
        image = cv2.imread(image_fn)        
        # image = image.reshape([256, 256, 3])

        label = self.label_df.iloc[index,1:].values.astype('float')

        if self.transforms is not None:            
            image = self.transforms(image=image)['image']

        return image, label

base_transforms = {
    'test' : A.Compose([   
            A.Resize(224,224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2(),
        ]),
}

tta_transforms = tta.Compose(
     [
         tta.Rotate90(angles=[0, 90, 180, 270]),
     ]
)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./dataset/test/")
    parser.add_argument('--sub_path', type=str, default="./dataset/sample_submission.csv")
    parser.add_argument('--label_path', type=str, default="./dataset/test_.csv")
    parser.add_argument('--weight_path', type=str, default='/content/DACON-4D/ckpt/model_1')
    parser.add_argument('--out_path', type=str, default='/content/')

    parser.add_argument('--model', type=str, default='resnet50')    
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--device', type=str, default=device)

    args = parser.parse_args()

    assert os.path.isdir(args.image_path), 'wrong path'
    assert os.path.isfile(args.label_path), 'wrong path'
    assert os.path.isdir(args.weight_path), 'wrong path' 
    assert os.path.isdir(args.out_path), 'wrong path'

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    
    weights = glob(os.path.join(args.weight_path, '*.pth'))

    test_df = pd.read_csv(args.label_path)


    test_set = DatasetMNIST(
        image_folder=args.image_path,
        label_df=test_df,
        transforms=base_transforms['test']
    )

    submission_df = pd.read_csv(args.sub_path)



    for weight in weights:   
        model = PreResnet50()
        model.load_state_dict(torch.load(weight, map_location=args.device))
        print('=' * 50)
        print('[info msg] weight {} is loaded'.format(weight))

        test_data_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size = args.batch_size,
                shuffle = False,
            )

        model.to(args.device)
        model.eval()
        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

        batch_size = args.batch_size
        batch_index = 0

        print('=' * 50)
        print('[info msg] inference start')

        for i, (images, _) in enumerate(tqdm(test_data_loader)):
            images = images.to(args.device)
            # outputs = model(images).detach().cpu().numpy().squeeze() # not tta            
            outputs = tta_model(images).detach().cpu().numpy().squeeze() # soft
            #outputs = (outputs > 0.5).astype(int) # hard vote
            batch_index = i * batch_size
            submission_df.iloc[batch_index:batch_index+batch_size, 1:] += outputs
    
    submission_df.iloc[:,1:] = (submission_df.iloc[:,1:] / len(weights) > 0.35).astype(int)
    
    SAVE_FN = os.path.join(args.out_path, datetime.now().strftime("%m%d%H%M") + '_ensemble_submission.csv')

    submission_df.to_csv(
        SAVE_FN,
        index=False
        )

    print('=' * 50)
    print('[info msg] submission fils is saved to {}'.format(SAVE_FN))


if __name__ == '__main__':
    main()
