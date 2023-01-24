# train.py
import torch.optim as optim
import torch
import os
from time import time

from torch import nn
from warmup_scheduler import GradualWarmupScheduler

from utils.EarlyStopping import EarlyStopping


def train_model(input_model, fold_k, model_save_path, args, logger, *loaders):
    
    fold_k = fold_k+1
    
    model = input_model
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    logger = logger
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, fold_k=fold_k, path=model_save_path)

    # ----------------------
    #  Loss function / Opt
    # ----------------------
    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4,momentum=0.9)
    
    # -----------------
    #   amp wrapping
    # -----------------
    scaler = torch.cuda.amp.GradScaler()
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #   optimizer=optimizer,
    #   mode='min',
    #   patience=2,
    #   factor=0.5,
    #   verbose=True
    #   )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=[
                                                            int(args.epochs*0.3),
                                                            int(args.epochs*0.4),
                                                            int(args.epochs*0.6)
                                                        ],
                                                        gamma=0.7)

    logger.info(f"""
---------------------------------------------------------------------------
    TRAINING INFO
        Loss function : {loss_function}
        Optimizer     : {optimizer}
        LR_Scheduler  : {lr_scheduler}
---------------------------------------------------------------------------""")
    
    # ----------------
    #     WARM-UP
    # ----------------
    warmup_epochs = int(args.epochs * 0.15)
    lr_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=lr_scheduler)
    
    train_tot_num = train_loader.dataset.__len__()
    val_tot_num = val_loader.dataset.__len__()
    
    train_corrects_sum = 0
    train_loss_sum = 0.0
    
    val_corrects_sum = 0
    val_loss_sum = 0.0
    
    logger.info(f'Training begins... Epochs = {epochs}')
    
    for epoch in range(epochs):
        time_start = time()

        if epoch <= warmup_epochs:
            lr_warmup.step()
        
        logger.info(f"""
===========================================================================
    PHASE INFO
        Current fold  : Fold ({fold_k})
        Current phase : {epoch+1}th epoch
        Learning Rate : {optimizer.param_groups[0]['lr']:.6f}
---------------------------------------------------------------------------""")
        
        torch.cuda.empty_cache()
        model.train()
        
        train_tmp_num = 0
        train_tmp_corrects_sum = 0
        train_tmp_loss_sum = 0.0
        for idx, (train_X, train_Y) in enumerate(train_loader):
            
            #IPython.embed(); exit(1)
            train_tmp_num += len(train_Y)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                train_pred = model(train_X)
                train_loss = loss_function(train_pred, train_Y)
            
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_pred_label = train_pred > args.threshold
            train_corrects = (train_pred_label == train_Y).sum()
            train_corrects_sum += train_corrects.item()
            
            train_loss_sum += train_loss.item()
            
            train_tmp_corrects_sum += train_corrects.item()
            train_tmp_loss_sum += train_loss.item()
            
            # Check between batches
            verbose = args.verbose
            
            if (idx+1) % verbose == 0:
                print(f"-- ({str((idx+1)).zfill(4)} / {str(len(train_loader)).zfill(4)}) Train Loss: {train_tmp_loss_sum/train_tmp_num:.6f} | Train Acc: {train_tmp_corrects_sum/(train_tmp_num*10)*100:.4f}%")
                
                # initialization
                train_tmp_num = 0
                train_tmp_corrects_sum = 0
                train_tmp_loss_sum = 0.0
            
        
        with torch.no_grad():
            
            for idx, (val_X, val_Y) in enumerate(val_loader):
                
                with torch.cuda.amp.autocast():
                    val_pred = model(val_X)
                    val_loss = loss_function(val_pred, val_Y)
                
                val_pred_label = val_pred > args.threshold
                val_corrects = (val_pred_label == val_Y).sum()
                val_corrects_sum += val_corrects.item()
                
                val_loss_sum += val_loss.item()
                
        train_acc = train_corrects_sum/(train_tot_num*10)*100
        train_loss = train_loss_sum/train_tot_num
        
        val_acc = val_corrects_sum/(val_tot_num*10)*100
        val_loss = val_loss_sum/val_tot_num
        
        time_end = time()
        time_len_m, time_len_s = divmod(time_end - time_start, 60)
        
        logger.info(f"""
---------------------------------------------------------------------------
    SUMMARY
        Finished phase  : {epoch+1}th epoch
        Time taken      : {time_len_m:.0f}m {time_len_s:.2f}s
        Training Loss   : {train_loss:.6f}  |  Training Acc   : {train_acc:.4f}%
        Validation Loss : {val_loss:.6f}  |  Validation Acc : {val_acc:.4f}%
===========================================================================\n""")
        
        # -------------------
        #    EARLY STOPPER
        # -------------------
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping condition met --- TRAINING STOPPED")
            logger.info(f"Best score: {early_stopping.best_score}")
            break
        
        # INITIALIZATION
        train_corrects_sum = 0
        val_corrects_sum = 0
        
        train_loss_sum = 0.0
        val_loss_sum = 0.0
  
        lr_scheduler.step()

    return model
