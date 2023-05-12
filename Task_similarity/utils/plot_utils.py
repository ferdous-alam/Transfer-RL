import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import helper

def plot_avg_ret(ret, steps, f_name=None):
    plt.figure(figsize=(10, 8))
    smooth_ret = helper.smooth(ret, window=10)
    plt.plot(steps, smooth_ret, lw=2.0)
    plt.xlabel(r'steps')
    plt.ylabel(r'average return')
    plt.savefig(f'figures/{f_name}.pdf')



