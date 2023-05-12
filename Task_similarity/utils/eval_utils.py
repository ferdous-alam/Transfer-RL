import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import core
from utils import models
from utils import helper
from pathlib import Path


def get_action(ac, o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                    deterministic)


def run_trained_policy(env_fn, max_ep_len=500, 
                       num_runs=1, load_path=None, 
                       render=False, f_name=None):

    # initialize environment and make a copy
    test_env = env_fn
    # initialize policy network
    actor_critic = core.MLPActorCritic
    # Create actor-critic module and target networks
    ac_kwargs = dict(hidden_sizes=[200]*4)
    ac = actor_critic(test_env.observation_space, test_env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(load_path))
    

    ep_reward_cache = []
    for n in range(num_runs):
        ret, ep_len = 0, 0
        reward_cache = []
        obs, done = test_env.reset(), False    
        while not done or (ep_len == max_ep_len):
            if render:
                test_env.render()
            act = get_action(ac, obs, True)
            obs2, reward, done, _ = test_env.step(act)
            ret += reward
            reward_cache.append(reward)
            obs = obs2
            if done:
                break
        ep_reward_cache.append(reward_cache)
        test_env.close()
    Path('data/{}'.format(f_name)).mkdir(parents=True, exist_ok=True)
    np.save(f'data/{f_name}/eval_policy_rewards.npy', ep_reward_cache)
