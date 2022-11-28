import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gym
import json
from livelossplot import PlotLosses
from tqdm import tqdm


# helper functions
def get_dataset(json_data_path, ds_type):
  json_data = json.load(open(json_data_path))
  source_data = json_data['data']
  source_state_action_data = []

  for trj in source_data:
      for d in trj:
          x = []
          s, a, s_next, r, terminated, info = d
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



def get_loader(json_data_path, state_dim, ds_type='None'):

    custom_dataset = get_dataset(json_data_path, ds_type)
    cartpole_dynamics_dataset = CreateTorchDataset(custom_dataset, state_dim=state_dim, 
                                        action_dim=1, ds_type=ds_type)
                                    
    target_loader = DataLoader(dataset=cartpole_dynamics_dataset, 
                            batch_size=32, 
                            shuffle=True)

    return target_loader
    

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
                loss_func, batch_size, ds_type,
                model_name=None, viz_loss=False):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f'device: {device}')
  
  train_loader = DataLoader(dataset=custom_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

  # random_seed = 9029323
  learning_rate = lr

  # torch.manual_seed(random_seed)
  model.to(device)

  if viz_loss:
    liveloss = PlotLosses()

  # training loop
  pbar = tqdm(range(num_epochs))
  for epoch in pbar:
    running_loss = 0.0 
    model.train() 
    for batch, label in train_loader:
        if viz_loss:
           logs = {}
        optimizer.zero_grad()
        batch, label = batch.to(device).float(), label.to(device).float()
        pred, latent = model(batch)
        if ds_type == 'reward':
          label = label.unsqueeze(1)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        running_loss +=  loss.item()

    epoch_loss = running_loss / len(train_loader) 
    if viz_loss:       # liveloss plotting
        logs['loss'] = epoch_loss
        liveloss.update(logs)
        liveloss.send()
    pbar.set_description("loss: %s" % str(epoch_loss)) 
    
  torch.save(model.state_dict(), 'saved_data/{}.pth'.format(model_name))



def get_similarity(model_s, model_t, model_source_path, model_target_path, target_loader):
    model_s.load_state_dict(torch.load(model_source_path, map_location='cpu'))
    model_t.load_state_dict(torch.load(model_target_path, map_location='cpu'))
    model_s.eval()
    model_t.eval()
    
    k = 0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    mse = nn.MSELoss(reduction='none')

    sim = []

    for batch, label in target_loader:
        with torch.no_grad():
            pred_t, z_t = model_t(batch.float())
            pred_s, z_s = model_s(batch.float())
        # sim_val = cos(z_s, z_t)
   
        sim_val = mse(pred_s, pred_t)
        sim_val = torch.sum(sim_val, 1)
    
        sim.append(sim_val)
    
    res = torch.cat(sim)
    mean_sim = res.mean().item()
    std_sim = res.std().item()
    
    return mean_sim, std_sim


def eval_model(model, trained_model_path, data_loader):
    loss_func = nn.MSELoss()
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    loss_vals  = []
    with torch.no_grad():
        for batch, label in data_loader:
            pred, latent = model(batch.float())
            loss = loss_func(pred, label)
            loss_vals.append(loss.item())
    loss_mean = np.array(loss_vals).mean() 
    loss_std = np.array(loss_vals).std() 
    
    return loss_mean, loss_std