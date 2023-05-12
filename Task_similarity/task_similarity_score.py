import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from utils import core
from utils.replay_buffer import ReplayBuffer, EvalBuffer, ModelBuffer
from utils import models
from utils import helper
from utils.models import DynamicsModel, RewardModel
import itertools
import json
import codecs
import time
from pathlib import Path



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


def get_dataset(json_data_path, ds_type):
  json_data = json.load(open(json_data_path))
  source_data = json_data['data']
  source_state_action_data = []

  for d in source_data:
    o, a, o2, r, terminated = d
    if ds_type == 'reward':
        m = o + a + [r]   # state + action + reward
    elif ds_type == 'dynamics':
        m = o + a + o2   # state + action + next_state
    else:
        raise Exception("dataset must be reward or dynamics")

    source_state_action_data.append(m)

  dataset = np.array(source_state_action_data)
  data = torch.tensor(dataset)
  return data


def get_similarity(model_s, model_t, model_source_path, model_target_path, target_loader):
    model_s.load_state_dict(torch.load(model_source_path, map_location='cpu'))
    model_t.load_state_dict(torch.load(model_target_path, map_location='cpu'))
    model_s.eval()
    model_t.eval()
    
    mse = nn.MSELoss(reduction='none')
    sim = []
    for batch, label in target_loader:
        with torch.no_grad():
            pred_t = model_t(batch.float())
            pred_s = model_s(batch.float())   
        sim_val = mse(pred_s, pred_t)
        sim_val = torch.sum(sim_val, 1)
        sim.append(sim_val)
    
    res = torch.cat(sim)
    
    return res.detach().cpu().numpy()


def task_similarity_score(env_fn, trained_source, 
                        trained_target, target_data_path, 
                        ds_type, fname):
    # path to json dataset
    # initialize environment and make a copy
    env = env_fn()

    # environment specs: obs, act dimensions    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    
    # get feature dimensions for model
    in_features = obs_dim[0] + act_dim
    out_dyn_features = obs_dim[0]
    out_rew_features = 1
        
    # model 
    source_dyn_model = DynamicsModel(in_features=in_features, out_features=out_dyn_features)
    target_dyn_model = DynamicsModel(in_features=in_features, out_features=out_dyn_features)    
    source_rew_model = RewardModel(in_features=in_features, out_features=out_rew_features)
    target_rew_model = RewardModel(in_features=in_features, out_features=out_rew_features)
    

    target_data = get_dataset(target_data_path, ds_type=ds_type)

    # create torch dataset
    if ds_type == 'dynamics': 
        target_dataset = CreateTorchDataset(target_data, in_features=in_features, 
                                    out_features=out_dyn_features)
    elif ds_type == 'reward':
        target_dataset = CreateTorchDataset(target_data, in_features=in_features, 
                                    out_features=out_rew_features)
    else:
         raise Exception('check dataset type: dynamics or reward')

    # create data loader
    target_loader = DataLoader(dataset=target_dataset, 
                        batch_size=64, 
                        shuffle=False)
    
    print('calculating similarity: ')
    if ds_type == 'dynamics': 
        sim = get_similarity(source_dyn_model, target_dyn_model, 
                                            trained_source, trained_target, 
                                            target_loader)
    elif ds_type == 'reward':
        sim = get_similarity(source_rew_model, target_rew_model, 
                                            trained_source, trained_target, 
                                            target_loader)
    else:
         raise Exception('check dataset type: dynamics or reward')

     # save data
    np.save(f'/home/ghost-083/Research/1_Transfer_RL/Task_similarity/data/ant/{fname}.npy', sim)



if __name__ == "__main__":

    # path ---> 
    # d_path = '/home/ghost-083/Research/1_Transfer_RL/Task_similarity/env_mods/halfcheetah/assets/'
    # xml_path = d_path + 'half_cheetah_0.xml'

    # env ---> 
    # env_fn = lambda: gym.make("Pendulum-v1", g=10.0)
    
    # env_fn = lambda: gym.make('HalfCheetah-v3', xml_file=xml_path, 
    #                     forward_reward_weight=-2.0, 
    #                     ctrl_cost_weight=0.1, 
    #                     reset_noise_scale=0.1
    #                     )

    d_path = '/home/ghost-083/Research/1_Transfer_RL/Task_similarity/env_mods/Ant/assets/'
    xml_path = d_path + 'ant_0.xml'

    env_fn = lambda: gym.make('Ant-v3', xml_file=xml_path)
    env_id = 1
    s_path = '/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/data/'
    trained_source =  s_path + f'ant/task_sim_dataset/ant_xml0_dyn_model.pth'
    trained_target = s_path + f'ant/task_sim_dataset/ant_xml{env_id}_dyn_model.pth'
    target_data_path = s_path + f'ant/task_sim_dataset/ant_train_xml{env_id}.json'
    ds_type = 'dynamics'
    fname = f'ant_xml0_xml{env_id}'

    task_similarity_score(env_fn, trained_source, 
                        trained_target, target_data_path, 
                        ds_type, fname)
 






