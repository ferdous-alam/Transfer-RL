import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import gym
import time
from tqdm import tqdm
import json
from utils.utils import get_similarity, get_loader
from utils.utils import get_dataset, CreateTorchDataset
from torch.utils.data import Dataset, DataLoader
from utils.models import PendulumDynamicsModel, PendulumRewardModel, PendulumDynamicsModelSuccessor, CartPoleModel
from utils.plot_utils import plot_task_similarity_score
from utils.utils import train_model, eval_model


def pendulum_trainer(dpath, ds_type, mname):
    # create dataset
    custom_dataset = get_dataset(dpath, ds_type=ds_type)
    pendulum_data = CreateTorchDataset(custom_dataset, state_dim=3, 
                                        action_dim=1, ds_type=ds_type)
    if ds_type == "dynamics":
        model = PendulumDynamicsModel(num_features=4)  # numebr of inputer features
    elif ds_type == "reward":
        model = PendulumRewardModel(num_features=4)  # numebr of inputer features
    else:
        raise Exception("Please choose dynamics or reward as ds_type")

    lr = 0.00001
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    model_name = mname
    batch_size = 32
    train_model(pendulum_data, model, lr,
                    num_epochs, optimizer,
                    loss, batch_size, ds_type,
                    model_name=model_name)
                    
    print('completed!')



def cartpole_trainer(dpath, ds_type, mname):
    
    # cartpole environment specific information
    state_dim = 4   # state dimension
    action_dim = 1  # action dimension
    in_features = state_dim + action_dim   # number of input features
    # number of output features depend on dataset type
    if ds_type == 'reward':
        out_features = 1
    elif ds_type == "dynamics":
        out_features = state_dim
    else:
        raise Exception('Please use reward or dynamics dataset type (ds_type)')

    # create dataset
    custom_dataset = get_dataset(dpath, ds_type=ds_type)
    cartpole_data = CreateTorchDataset(custom_dataset, state_dim=state_dim, 
                                        action_dim=action_dim, ds_type=ds_type)
    # instantiate model class
    model = CartPoleModel(in_features=in_features, 
                        out_features=out_features)
    # training hyper-parameters
    lr = 0.00001
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    model_name = mname
    batch_size = 32   # fixed batch size for all experiments
    train_model(cartpole_data, model, lr,
                    num_epochs, optimizer,
                    loss, batch_size, ds_type,
                    model_name=model_name)
                    
    print('completed!')


def halfcheetah_trainer(dpath, ds_type, mname):
    
    # cartpole environment specific information
    state_dim = 17   # state dimension
    action_dim = 6  # action dimension
    in_features = state_dim + action_dim   # number of input features
    # number of output features depend on dataset type
    if ds_type == 'reward':
        out_features = 1
    elif ds_type == "dynamics":
        out_features = state_dim
    else:
        raise Exception('Please use reward or dynamics dataset type (ds_type)')

    # create dataset
    custom_dataset = get_dataset(dpath, ds_type=ds_type)
    halfcheetah_data = CreateTorchDataset(custom_dataset, state_dim=state_dim, 
                                        action_dim=action_dim, ds_type=ds_type)
    # instantiate model class
    model = CartPoleModel(in_features=in_features, 
                        out_features=out_features)
    # training hyper-parameters
    lr = 0.00001
    num_epochs = 250
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    model_name = mname
    batch_size = 32   # fixed batch size for all experiments
    train_model(halfcheetah_data, model, lr,
                    num_epochs, optimizer,
                    loss, batch_size, ds_type,
                    model_name=model_name)
                    
    print('completed!')



if __name__ == "__main__":

    ds_type = 'dynamics'
    for k in range(1, 2, 1):
    # path to json dataset
        path = f'/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/Task_similarity/dataset/Halfcheetah/Halfcheetah_dataset_{k}_train.json'
        print(f'data loaded#{k}')
        halfcheetah_trainer(path, ds_type, f'halhcheetah_{k}')
        
    print(f'# # # training completed! # # # ')
        
