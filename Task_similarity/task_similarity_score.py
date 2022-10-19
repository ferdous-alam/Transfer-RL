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
from utils.models import PendulumDynamicsModel, PendulumRewardModel, CartpoleDynamicsModel, PendulumDynamicsModelSuccessor
from utils.plot_utils import plot_task_similarity_score
from utils.utils import train_model



def task_similarity_score():
    # path to json dataset
    
    ds_type = 'reward'

    means = [] 
    stds = [] 
    s_id = 9
    for k in range(9, 14, 1):
        target_path = f'/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/Task_similarity/dataset/Pendulum/Pendulum_dataset_{k}_train.json'
        
        # load pre-trained models    
        trained_source = f'saved_data/pendulum_{s_id}.pth'
        trained_target = f'saved_data/pendulum_{k}.pth'
        if ds_type == "dynamics":
            source_model = PendulumDynamicsModel(num_features=4)
            target_model = PendulumDynamicsModel(num_features=4)
            data_loader = get_loader(target_path, state_dim=3, ds_type='dynamics')
        elif ds_type == "reward":
            source_model = PendulumRewardModel(num_features=4)
            target_model = PendulumRewardModel(num_features=4)
            data_loader = get_loader(target_path, state_dim=3, ds_type='reward')
        else: 
            raise Exception("Please choose reward or dynamics for ds_type")
        
        mean_sim, std_sim = get_similarity(source_model, target_model, 
                                            trained_source, trained_target, 
                                            data_loader)
        means.append(mean_sim)
        stds.append(std_sim)

    plot_task_similarity_score(means, stds, fig_name='latent_pendulum_reward')


if __name__ == "__main__":
    task_similarity_score()
 






