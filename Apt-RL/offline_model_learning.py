import numpy as np
import gym
import torchEarlyStopss
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


def interact(env, steps=10000, max_ep_len=500, 
            dataset_name=None, save_data=True):
     # reset environment 
    o = env.reset()
    ep_len = 0
    data = []
    for t in tqdm(range(steps)):
        # exploration policy for data collection
        a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        single_data = [o.tolist(), a.tolist(),
                      o2.tolist(), r, d]
        data.append(single_data)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_len = env.reset(), 0

     # save as json file
    dataset = {'data': data, 
               'steps': steps}

    if save_data == True:
        json.dump(dataset, codecs.open('/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/dataset/{}.json'.format(dataset_name), 'w', encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4)


def create_dataset(env_fn, train_steps, 
                eval_steps, 
                train_dataset_name, 
                eval_dataset_name):
    # environment
    env = env_fn()
    
    # create dataset
    interact(env, steps=10000, dataset_name=train_dataset_name)
    interact(env, steps=2000, dataset_name=eval_dataset_name)
    


   
    # # save models
    # Path('data/{}'.format(test_name)).mkdir(parents=True, exist_ok=True)
    # dyn_path = 'data/{}/dyn_model.pth'.format(test_name)
    # rew_path = 'data/{}/rew_model.pth'.format(test_name)
    
    # torch.save(dyn.state_dict(), dyn_path) 
    # torch.save(rew.state_dict(), rew_path) 

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
    return train_loss


def eval_one_epoch(eval_loader, model, 
                loss_func, eval_loss, 
                device):
    running_loss = 0.0
    for batch, label in eval_loader:
        batch, label = batch.to(device).float(), label.to(device).float()
        with torch.no_grad():
            pred = model(batch)
        loss = loss_func(pred, label)
        running_loss += loss.item()
    eval_loss.append(running_loss/len(eval_loader))
    return eval_loss


def train_model(train_data, 
                eval_data, model, 
                in_features, 
                out_features, 
                batch_size=64, 
                lr=1e-3, 
                loss_func=nn.MSELoss(),
                optimizer=None,
                test_name=None,  
                model_name=None, 
                save=False):
    # create torch dataset
    train_dataset = CreateTorchDataset(train_data, in_features=in_features, 
                                    out_features=out_features)
    eval_dataset = CreateTorchDataset(eval_data, in_features=in_features, 
                                    out_features=out_features)
    
    # create data loader
    train_loader = DataLoader(dataset=train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True)
    eval_loader = DataLoader(dataset=eval_dataset, 
                        batch_size=batch_size, 
                        shuffle=False)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # learning rate
    learning_rate = lr

    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # early stopping 
    early_stopper = EarlyStopper(patience=10, min_delta=0)
  
    # initialize loss 
    train_loss, eval_loss = [],  []

    while True:
        # training loop
        train_loss = train_one_epoch(train_loader, optimizer, 
                        model, loss_func, train_loss, device)

        # early stopping
        eval_loss = eval_one_epoch(eval_loader, model, 
                                loss_func, eval_loss, device)
        latest_validation_loss = eval_loss[-1]

        if early_stopper.early_stop(latest_validation_loss): 
            break
    
    if save:
        torch.save(model.state_dict(), 'data/{}/{}.pth'.format(test_name, model_name))


def learn_models(env_fn, train_path, eval_path, test_name): 
    # initialize environment and make a copy
    env = env_fn()

    # environment specs: obs, act dimensions    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    
    # get feature dimensions for model
    in_features = obs_dim[0] + act_dim
    out_dyn_features = obs_dim[0]
    out_rew_features = 1

    # create dataset
    dyn_train_data = get_dataset(train_path, ds_type='dynamics')
    dyn_eval_data = get_dataset(eval_path, ds_type='dynamics')

    rew_train_data = get_dataset(train_path, ds_type='reward')    
    rew_eval_data = get_dataset(eval_path, ds_type='reward')    
   
    Path('data/{}'.format(test_name)).mkdir(parents=True, exist_ok=True)

    # train dynamics model
    train_model(train_data=dyn_train_data, 
                eval_data=dyn_eval_data, 
                model=dyn_model, 
                in_features=in_features, 
                out_features=out_dyn_features, 
                batch_size=64, 
                lr=1e-3, 
                loss_func=nn.MSELoss(), 
                test_name=test_name,
                model_name='dyn_model', 
                save=True)
    
    # eval dynamics model
    train_model(train_data=rew_train_data, 
                eval_data=rew_eval_data, 
                model=rew_model, 
                in_features=in_features, 
                out_features=out_rew_features, 
                batch_size=64, 
                lr=1e-3, 
                loss_func=nn.MSELoss(),
                test_name=test_name, 
                model_name='rew_model', 
                save=True)


if __name__ == "__main__":
    env_fn = lambda: gym.make("Pendulum-v1", g=10.0)
    # create_dataset(env_fn, train_steps=10000, 
    #                         eval_steps=5000, 
    #                         train_dataset_name='pendulum_train_g_10', 
    #                         eval_dataset_name='pendulum_eval_g_10')


    train_path = '/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/dataset/pendulum_train_g_10.json'
    eval_path = '/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/dataset/pendulum_eval_g_10.json'
    
    learn_models(env_fn, train_path, eval_path, 'pendulum_g_10')
