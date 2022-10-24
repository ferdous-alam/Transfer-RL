import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import gym
import time
from tqdm import tqdm
import json
from livelossplot import PlotLosses
from utils.utils import get_similarity, get_loader
from utils.utils import get_dataset, CreateTorchDataset
from torch.utils.data import Dataset, DataLoader
from utils.models import PendulumDynamicsModel, PendulumRewardModel, CartPoleModel, PendulumDynamicsModelSuccessor
from utils.plot_utils import plot_task_similarity_score
from utils.utils import train_model



def task_similarity_score():
    # path to json dataset
    
    ds_type = 'dynamics'

    means = [] 
    stds = [] 
    s_id = 0   # source task id
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

    
    for k in range(0, 6, 1):
        target_path = f'/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/Task_similarity/dataset/Cartpole/Cartpole_dataset_{k}_train.json'
        
        # load pre-trained models    
        trained_target = f'saved_data/cartpole_trained_model/cartpole_{k}.pth'
        trained_source = f'saved_data/cartpole_trained_model/cartpole_{s_id}.pth'
        
        source_model = CartPoleModel(in_features=in_features, 
                                    out_features=out_features)
        target_model = CartPoleModel(in_features=in_features, 
                                    out_features=out_features)
        
        data_loader = get_loader(target_path, state_dim=state_dim, 
                                ds_type='dynamics')
        
        mean_sim, std_sim = get_similarity(source_model, target_model, 
                                            trained_source, trained_target, 
                                            data_loader)
        means.append(mean_sim)
        stds.append(std_sim)

    plot_task_similarity_score(means, stds, fig_name='pred_cartpole_dynamics')


if __name__ == "__main__":
    task_similarity_score()
 






