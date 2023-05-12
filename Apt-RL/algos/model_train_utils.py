import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
from utils import core
from utils.replay_buffer import ReplayBuffer, EvalBuffer
from utils import models
from utils import helper
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import time


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CreateTorchDataset(Dataset):
    def __init__(self, custom_dataset, 
                in_features, out_features):
        self.custom_dataset = custom_dataset
        self.in_features = in_features
        self.out_features = out_features

    def __len__(self):
        return len(self.custom_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.custom_dataset[idx, :-self.out_features]
        label = self.custom_dataset[idx, -self.out_features:]
        return sample, label


def train_one_epoch(train_loader, optimizer, 
                    model, loss_func, train_loss, 
                    device):
    running_loss = 0.0
    for batch, label in train_loader:
        optimizer.zero_grad()
        batch, label = batch.to(device).float(), label.to(device).float()
        pred = model(batch)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss.append(running_loss/len(train_loader))
    return train_loss, model

def train_model(train_data, model, 
                in_features, 
                out_features, 
                optimizer,
                batch_size=64, 
                lr=1e-3, save=False):
    # create torch dataset
    train_dataset = CreateTorchDataset(train_data, in_features=in_features, 
                                    out_features=out_features)
    
    # create data loader
    train_loader = DataLoader(dataset=train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True)

    # device
    device = torch.device("cpu")
    
    # loss function
    loss_func = nn.MSELoss()

    # learning rate
    learning_rate = lr

    model.to(device)

    # early stopping 
    early_stopper = EarlyStopper(patience=10, min_delta=0)
  
    # initialize loss 
    train_loss, eval_loss = [],  []

    while True:
        # training loop
        train_loss, model = train_one_epoch(train_loader, optimizer, 
                        model, loss_func, train_loss, device)

        latest_validation_loss = train_loss[-1]

        if early_stopper.early_stop(latest_validation_loss): 
            break

    if save:
        torch.save(model.state_dict(), 'data/{}/{}.pth'.format(test_name, model_name))
    return model

def update_models(env, replay_buffer, lr=1e-3):
    obs = torch.tensor(replay_buffer.obs_buf[:replay_buffer.size])
    act = torch.tensor(replay_buffer.act_buf[:replay_buffer.size])
    obs2 = torch.tensor(replay_buffer.obs2_buf[:replay_buffer.size])
    rew = torch.tensor(replay_buffer.rew_buf[:replay_buffer.size]).unsqueeze(1)

    dyn_train_data = torch.cat((obs, act, obs2), dim=1)
    rew_train_data = torch.cat((obs, act, rew), dim=1)
   
    # environment specs: obs, act dimensions    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    
    # get feature dimensions for model
    in_features = obs_dim[0] + act_dim
    out_dyn_features = obs_dim[0]
    out_rew_features = 1

    # models with pretrained weights: pretrained for small steps for better warm up
    # dynamics model
    dyn_model = models.DynamicsModel(in_features=in_features, out_features=out_dyn_features)
    # dyn_model.load_state_dict(torch.load(dyn_path))
    # reward model
    rew_model = models.RewardModel(in_features=in_features, out_features=out_rew_features)


    # Set up optimizers for dynamics and reward model
    dyn_optimizer = torch.optim.Adam(dyn_model.parameters(), lr=lr)
    rew_optimizer = torch.optim.Adam(rew_model.parameters(), lr=lr)


    # train dynamics model
    dyn_model = train_model(train_data=dyn_train_data, 
                model=dyn_model, 
                in_features=in_features, 
                out_features=out_dyn_features, 
                optimizer=dyn_optimizer,
                batch_size=64, 
                lr=1e-3)    
    
    # eval dynamics model
    rew_model = train_model(train_data=rew_train_data, 
                model=rew_model, 
                in_features=in_features, 
                out_features=out_rew_features,
                optimizer=rew_optimizer, 
                batch_size=64, 
                lr=1e-3)
    return dyn_model, rew_model
