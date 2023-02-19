# main.py

import os
import copy
import torch
import logging
import random

import argparse
import pandas as pd
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import KFold
from torch import nn


from src.train import train_model
from utils.imageprocess import image_transformer
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader
from utils.call_model import CallModel





# -----------------------------
#  train/val splitting method
# -----------------------------
def split_index(total_index, val_ratio):
    tot_len = len(total_index)
    train_len = int(tot_len*(1-val_ratio))

    train_sampled = random.sample(total_index, train_len)
    val_sampled = [i for i in total_index if i not in train_sampled]
    logger.info(f"Trainset length: {len(train_sampled)}, Valset length: {len(val_sampled)}")
    
    return train_sampled, val_sampled


def split_kfold(k, train_len=65988):    
    kfold = KFold(n_splits=k, shuffle=True)
    splitted = kfold.split(range(train_len))
    return splitted



def split_dataset(args):
    if args.fold_k == 1:
        train_index_set, val_index_set = split_index(range(65988), args.val_ratio)
        train_index_set, val_index_set = [train_index_set], [val_index_set]
        
    elif args.fold_k > 1:
        splitted = split_kfold(args.fold_k, train_len=65988)
        train_index_set, val_index_set = [], []
        
        for train_fold, val_fold in splitted:
            train_index_set.append(train_fold)
            val_index_set.append(val_fold)
        
        logger.info(f"Trainset length: {len(train_index_set[0])}, Valset length: {len(val_index_set[0])}")
    
    return train_index_set, val_index_set
    





# -----------------------------------
#   Setting seeds for reproduction
# -----------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


# ----------------------
#    Loading Dataset
# ----------------------
def load_dataset(mode='train', **kwargs):

    img_dir = kwargs['img_dir']
    label_dir = kwargs['label_dir']
    device = kwargs['device']

    if mode=='train':
        
        train_index = kwargs['train_index']
        val_index = kwargs['val_index']
        batch_size = kwargs['batch_size']
        
        train_set = CustomDataLoader(
            img_dir=img_dir,
            label_dir=label_dir,
            train=True,
            row_index=train_index,
            device=device)

        val_set = CustomDataLoader(
            img_dir=img_dir,
            label_dir=label_dir,
            train=False,
            row_index=val_index,
            device=device)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)

        return train_loader, val_loader




def load_trained_weight(model_input=None, model_index=0, model_type='early', fold_k=1, trained_weight_path='/content/DACON-4D/ckpt'):
    '''
    Load trained weights to your model.
    '''
    assert model_index > 0

    model_name = f'early_stopped_fold{fold_k}.pth' if model_type == 'early' else f'model_ckpt_fold{fold_k}_{model_type}.pth'
    ckpt_path = os.path.join(trained_weight_path, f'model_{model_index}', model_name)

    trained_model = model_input
    trained_model.load_state_dict(torch.load(ckpt_path))
    trained_model.eval()

    return trained_model
    






if __name__ == "__main__":
    
    # ARGUMENTS PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_index", type=int, default=0, help='My model index. Integer type, and should be greater than 0')
    parser.add_argument("--base_dir", type=str, default="/content/DACON-4D", help='Base PATH of your work')
    parser.add_argument("--mode", type=str, default="train", help='[train]')
    parser.add_argument("--data_type", type=str, default="denoised", help='[original | denoised]: default=denoised')
    parser.add_argument("--ckpt_path", type=str, default="/content/DACON-4D/ckpt", help='PATH to weights of ckpts.')
    parser.add_argument("--base_model", type=str, default="resnet152", help="[plain_resnet50, custom_resnet50, plain_efficientnetb4]")
    parser.add_argument("--pretrained", dest='pretrained', action='store_true', help='Default is false, so specify this argument to use pretrained model')
    parser.add_argument("--pretrained_weights_dir", type=str, default="/content/pretrained_model", help='PATH to weights of pretrained model')
    parser.add_argument("--cuda", dest='cuda', action='store_false', help='Whether to use CUDA: defuault is True, so specify this argument not to use CUDA')
    parser.add_argument("--device_index", type=int, default=0, help='Cuda device to use. Used for multiple gpu environment')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for train-loader for training phase')
    parser.add_argument("--val_ratio", type=float, default=0.2, help='Ratio for validation set: default=0.1')
    parser.add_argument("--epochs", type=int, default=100, help='Epochs for training: default=100')
    parser.add_argument("--learning_rate", type=float, default=0.0001, help='Learning rate for training: default=0.0029')
    parser.add_argument("--patience", type=int, default=8, help='Patience of the earlystopper: default=10')
    parser.add_argument("--verbose", type=int, default=100, help='Between batch range to print train accuracy: default=100')
    parser.add_argument("--threshold", type=float, default=0.0, help='Threshold used for predicting 0/1')
    parser.add_argument("--seed", type=int, default=777, help='Seed used for reproduction')
    parser.add_argument("--fold_k", type=int, default=5, help='Number of fold for k-fold split. If k=1, standard train/val splitting is done.')
    args = parser.parse_args()
    
    
    # ASSERT CONDITIONS
    assert (args.model_index > 0) and (args.mode in ['train'])
    
    # ------------------
    #   logger setting
    # ------------------
    LOG_PATH = os.path.join(args.base_dir, 'logs')
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s : %(message)s", 
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(LOG_PATH, f"log_model_{args.model_index}.txt")),
                            logging.StreamHandler()
                        ])
    logger.info("START")
    
    
    # ------------------------
    #   GLOBAL CUDA SETTING
    # ------------------------
    global_cuda = args.cuda and torch.cuda.is_available()
    if global_cuda:
        global_device = torch.device(f'cuda:{args.device_index}')
    else:
        global_device = torch.device('cpu')

    logger.info(f"Global Device: {global_device}")
    logger.info(f'Parsed Args: {args}')
    
    
    # ------------------
    #    Seed Setting
    # ------------------
    set_seed(args.seed)
    
    
    # -----------------------
    #      SET DIRECTORY
    # -----------------------
    base_dir = args.base_dir
    
    if args.data_type == 'original':
        data_to_use = ['train_new']
    elif args.data_type == 'denoised':
        data_to_use = ['denoised_trainset_weak']
    else:
        raise Exception(f"Data Type Choice Error. No {args.data_type}")

    data_path_train = os.path.join('dataset', data_to_use[0]) 
    logger.info(f'Data used: train: {data_path_train}')
    
    img_dir_train = os.path.join(base_dir, data_path_train)
    label_dir_train = os.path.join(base_dir, 'dataset/train_new.csv')
    
    ckpt_folder_path = os.path.join(args.ckpt_path, f'model_{args.model_index}')
    
    
    # -------------------
    #   TRAIN/VAL SPLIT
    # -------------------
    train_index_set, val_index_set = split_dataset(args)


    # ----------------
    #    Call model
    # ----------------
    base_model_type = args.base_model
    base_model = CallModel(model_type=base_model_type,
                           pretrained=args.pretrained,
                           logger=logger,
                           path=args.pretrained_weights_dir).model_return()
    
    model = base_model.to(global_device)
    
    
    # --------------------
    #        TRAIN
    # --------------------
    if args.mode == 'train':
        
        #  MAKE FOLDER for saving CHECKPOINTS
        # if folder already exists, assert. Else, make folder.
        assert not os.path.exists(ckpt_folder_path), "Model checkpoint folder already exists."
        os.makedirs(ckpt_folder_path)
        
        for k in range(args.fold_k):
            model_to_train = copy.deepcopy(model)
            
            logger.info(f"Training on Fold ({k+1}/{args.fold_k})")
            # Load trainset/valset
            train_index = train_index_set[k]
            val_index = val_index_set[k]
            
            train_loader, val_loader = load_dataset(mode='train',
                                                    batch_size=args.batch_size,
                                                    img_dir=img_dir_train,
                                                    label_dir=label_dir_train,
                                                    train_index=train_index,
                                                    val_index=val_index,
                                                    device=global_device)
            
            # Train model
            train_model(model_to_train, k, ckpt_folder_path, args, logger, train_loader, val_loader)
    
  
