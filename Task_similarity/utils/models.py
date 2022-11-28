import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# Pendulum  
class PendulumRewardModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear_1 = nn.Linear(num_features, 32)
        self.linear_2 = nn.Linear(32, 16)
        self.linear_3 = nn.Linear(16, 32)
        self.linear_4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        z = self.relu(x)
        x = self.linear_3(z)
        x = self.relu(x)
        r = self.linear_4(x)
        return r, z


class PendulumDynamicsModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear_1 = nn.Linear(num_features, 32)
        self.linear_2 = nn.Linear(32, 16)
        self.linear_3 = nn.Linear(16, 32)
        self.linear_4 = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        z = self.relu(x)
        x = self.linear_3(z)
        x = self.relu(x)
        x = self.linear_4(x)
        return x, z
    
class PendulumDynamicsModelSuccessor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear_1 = nn.Linear(num_features, 32)
        self.linear_2 = nn.Linear(32, 32)
        self.linear_3 = nn.Linear(32, 16)
        self.linear_4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        phi = self.linear_3(x)
        x = self.linear_4(phi)
        return x, phi



class CartPoleModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, 32)
        self.linear_2 = nn.Linear(32, 16)
        self.linear_3 = nn.Linear(16, 32)
        self.linear_4 = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        z = self.relu(x)
        x = self.linear_3(z)
        x = self.relu(x)
        x = self.linear_4(x)
        return x, z