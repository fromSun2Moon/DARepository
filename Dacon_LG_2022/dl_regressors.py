#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random
import os
import argparse
import numpy as np
from itertools import combinations
from abc import ABC, abstractmethod
import sys
import pickle
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from torch.optim import * # optimizer, scheduler
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from model import *
from utils import *


criterion_dict = {"l1":nn.L1Loss(reduction="mean"), \
                "mse":nn.MSELoss(reduction="mean"), "nrmse": NRMSELoss()}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer():
    def __init__(self, model, optimizer, train_loader, val_loader, criterion, test_loader=None, \
                                    scheduler=None, device='cuda', parallel=False):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.device = device
        self.parallel = parallel
        self.save_dir = save_dir
        # Loss Function
        self.criterion = criterion

    def train(self, valid_score):

        self.model.to(self.device)
        self.model.train()
        train_loss = []
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            _x = self.model(x)


            loss = 1.2 * self.criterion(_x[:,:8], y[:,:8]) + self.criterion(_x[:,8:], y[:,8:])
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            
        # if self.scheduler is not None and valid_score is not None:
        #     self.scheduler.step(valid_score)

            if self.scheduler is not None:
                self.scheduler.step()
            
        return train_loss
    
    def saver(self, save_dir):
        if parallel:
            torch.save(self.model.module.state_dict(), save_dir, _use_new_zipfile_serialization=False)
        torch.save(self.model.state_dict(), save_dir, _use_new_zipfile_serialization=False)

    
    def evaluation_score(self):
        preds = []
        gts = []
        self.model.eval()
        with torch.no_grad():
            for x,y in val_loader:
                out = self.model(x.to(self.device))
                preds.append(out)
                gts.append(y)
            preds = torch.cat(preds).cpu().numpy()
            gts = torch.cat(gts).cpu().numpy()
            score = self.lg_nrmse(preds, gts)
        return score

    def evaluation(self):
        preds = []
        gts = []
        self.model.eval()
        val_losses=[]
        with torch.no_grad():
            for x,y in val_loader:
                y = y.to(self.device)
                out = self.model(x.to(self.device))
                val_loss =  1.2 * self.criterion(out[:,:8], y[:,:8]) + self.criterion(out[:,8:], y[:,8:])
                val_losses.append(val_loss)
        return val_losses

    def test_and_save(self, submit, save_name):
        preds=[]
        with torch.no_grad():
            for x in test_loader:
                preds.append(self.model(x.to(self.device)))
            preds = torch.cat(preds).cpu().numpy()
            
            for idx, col in enumerate(submit.columns):
                if col=='ID':
                    continue
                submit[col] = preds[:,idx-1]
            submit.to_csv(save_name, index=False)
    
    def lg_nrmse(self, gt, preds):
        # Y Feature별 NRMSE 총합
        # Y_01 ~ Y_08 까지 20% 가중치 부여
        all_nrmse = []
        for idx in range(14): # ignore 'ID'
            rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
            nrmse = rmse/np.mean(np.abs(gt[:,idx]))
            all_nrmse.append(nrmse)
        score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
        return score


if __name__ == "__main__":

    ################
    # Parameter settings
    ################
    EPOCHS = 200
    LR = 7e-03
    BS = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = 'mse'
    parallel = False
    v1 = f"4depth-linear-schedule-valloss-weight"
    version = f'skipmodel_{BS}_{LR}_{v1}_{crit}.pth'
    save_dir = 'result'

    print_msg(version)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cache_dir = save_dir + version


    ################
    # Data settings
    ################

    # valid ratio 조정 
    thr = 0.9
    seed = 1006
    seed_everything(seed) # Seed 고정

    train_df = pd.read_csv('./train.csv')
    
    train_set = train_df.sample(frac=thr,random_state=seed)

    val_set = train_df.iloc[list(set(train_df.index) - set(train_set.index))]
    train_x = train_set.filter(regex='X') # Input : X Featrue
    train_y = train_set.filter(regex='Y') # Output : Y Feature
    
    val_x = val_set.filter(regex='X') 
    val_y = val_set.filter(regex='Y') 

    test_x = pd.read_csv('./test.csv').drop(columns=['ID'])
    # val_x = test_x
    # val_df = pd.read_csv('./auto_csv_result/ensems_rfc_lgb_ex_1.927172971621052.csv')
    # val_y = val_df.drop(columns="ID")

    # test data format
    submit = pd.read_csv('./sample_submission.csv')

    train_dataset = MyDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)

    val_dataset = MyDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)

    test_dataset = MyDataset(test_x, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    model = ShortSkipConnection()
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]


    t_total = len(train_loader) * EPOCHS

    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=LR)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    # optimizer = Adam(params = model.parameters(), lr = LR)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, \
                            # threshold_mode='abs', min_lr=1e-8, verbose=True)
    
    trainer = Trainer(model, optimizer, train_loader, val_loader, criterion_dict[crit], test_loader,scheduler, device)

    stop_count = 0
    best_score = 1e09
    file_name = ''.join(version.split(".")[:-1]) +".csv"
    valid_score = None #initialize

    for ind, epoch in enumerate(range(EPOCHS)):
        train_losses = trainer.train(valid_score)
        train_loss_avg = np.mean(train_losses)
        #valid_score = trainer.evaluation_score()
        valid_score = torch.FloatTensor(trainer.evaluation()).mean()

        print(f'Epoch : [{epoch}] Train loss : [{np.mean(train_losses)}] |Val score : [{valid_score}])')

        if best_score > valid_score:
            best_score = valid_score
            trainer.saver(cache_dir)
            print_msg(f"Update score and save caches... | best score : {best_score}")
            # file_name = file_name + f"{str(valid_score)[:6]}"+".csv"
            trainer.test_and_save(submit, save_dir+f"{file_name}")
            stop_count = 0 
        
        else:
            stop_count +=1
            if epoch > (EPOCHS//2) and stop_count >=10:
                print("End Loop for early stopping")
                break








