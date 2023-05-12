import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class DynamicsModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_sim = 256
        self.hidden_1 = nn.Linear(in_features, self.hidden_sim)
        self.hidden_2 = nn.Linear(self.hidden_sim, self.hidden_sim)
        self.hidden_3 = nn.Linear(self.hidden_sim, self.hidden_sim)
        self.hidden_4 = nn.Linear(self.hidden_sim, self.hidden_sim)
        self.linear = nn.Linear(self.hidden_sim, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.hidden_3(x)
        x = self.relu(x)
        x = self.hidden_4(x)
        x = self.relu(x)        
        x = self.linear(x)
        return x



class RewardModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_sim = 128
        self.hidden_1 = nn.Linear(in_features, 128)
        self.hidden_2 = nn.Linear(self.hidden_sim , self.hidden_sim )
        self.hidden_3 = nn.Linear(self.hidden_sim , self.hidden_sim )
        self.hidden_4 = nn.Linear(self.hidden_sim , self.hidden_sim )
        self.linear = nn.Linear(self.hidden_sim , out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.hidden_3(x)
        x = self.relu(x)
        x = self.hidden_4(x)
        x = self.relu(x)        
        x = self.linear(x)
        return x
