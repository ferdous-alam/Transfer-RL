import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gym
import json
from livelossplot import PlotLosses


# helper functions
def get_dataset(json_data, ds_type=None):
  source_data = json_data['data']
  source_state_action_data = []

  for trj in source_data:
      for d in trj:
          x = []
          s, a, s_next, r, terminated = d
          if ds_type == 'reward':
            m = s + a + [r]   # state + action + reward
          elif ds_type == 'dynamics':
            m = s + a + s_next   # state + action + next_state
          else:
            raise Exception("dataset must be reward or dynamics")

          source_state_action_data.append(m)

  dataset = np.array(source_state_action_data)
  data = torch.tensor(dataset)
  return data


class CreateTorchDataset(Dataset):
    def __init__(self, custom_dataset, 
                state_dim, action_dim, 
                ds_type=None):
        self.custom_dataset = custom_dataset
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ds_type = ds_type

    def __len__(self):
        return len(self.custom_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.ds_type == 'dynamics':
            sample = self.custom_dataset[idx, :-self.state_dim]
            label = self.custom_dataset[idx, -self.state_dim:]
        elif self.ds_type == 'reward':
            sample = self.custom_dataset[idx, :-1]
            label = self.custom_dataset[idx, -1]
        else:
            raise Exception("dataset must be reward or dynamics")
        return sample, label

      


def train_model(custom_dataset, model, lr,
                num_epochs, optimizer,
                loss_func, batch_size, 
                model_name=None):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  
  train_loader = DataLoader(dataset=custom_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

  # random_seed = 9029323
  learning_rate = lr

  # torch.manual_seed(random_seed)
  model.to(device)

  liveloss = PlotLosses()

  # training loop
  for epoch in range(num_epochs): 
      model.train() 
      for batch, label in train_loader:
          logs = {}
          batch, label = batch.to(device).float(), label.to(device).float()
          s_next_pred, latent = model(batch)
          loss = 5 * loss_func(s_next_pred, label)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      # liveloss plotting
      logs['loss'] = loss.item()
      liveloss.update(logs)
      liveloss.send()

  torch.save(model.state_dict(), 'saved_data/{}.pth'.format(model_name))



def get_similarity(model_source, model_target, target_loader):
  k = 0

  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  mse = nn.MSELoss(reduction='none')

  sim = []

  for batch, label in target_loader:
      pred_t, z_t = model_target(batch.float())
      pred_s, z_s = model_source(batch.float())
      sim_val = cos(z_s, z_t)
      # sim_val = mse(z_t, z_s)
      # sim.append(sim_val.sum(dim=1))
      sim.append(sim_val)

  sim = torch.cat(sim)
  mean_sim = sim.mean().item()
  return mean_sim
