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
import time 
from PIL import Image
from colabgymrender.recorder import Recorder
from gym.wrappers.monitoring import video_recorder
import cv2

# for saving video unset LD PRELOAD before running the python file 


def get_action(ac, o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                    deterministic)


def run_trained_policy(env, max_ep_len=1000, 
                       num_runs=1, load_path=None, 
                       render=False, f_name=None, agent=None):

    # initialize environment and make a copy
    # test_env = Recorder(env, 'video/')
    actor_critic = core.MLPActorCritic
    # Create actor-critic module and target networks
    ac_kwargs = dict(hidden_sizes=[200]*4)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    if load_path is not None: 
        ac.load_state_dict(torch.load(load_path))

    out = cv2.VideoWriter(f"video/{f_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (500,500))
    
    ep_reward_cache = []
    t = 0
    for n in range(num_runs):
        ret, ep_len = 0, 0
        reward_cache = []
        obs, done = env.reset(), False    
        while not (done or (ep_len == max_ep_len)):
            if render:
                img = env.render(mode='rgb_array')
                out.write(img)
                # time.sleep(0.05)
                # if t == 5:
                im_sv = Image.fromarray(img)
                im_sv.save(f"video/{f_name}/{f_name}_{t}.png")
            if agent == 'random':
                act = env.action_space.sample()
            else:
                act = get_action(ac, obs, True)

            obs2, reward, done, _ = env.step(act)
            ret += reward
            ep_len += 1
            reward_cache.append(reward)
            obs = obs2
            if done:
                # im_sv = Image.fromarray(img)
                # im_sv.save(f"images/{f_name}.png")
                out = cv2.VideoWriter(f"video/{f_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (500,500))
                break
            t += 1
        ep_reward_cache.append(ret)
        env.close()
    out.release()
    print(f'total average return: {np.mean(ep_reward_cache)}')

    # save data
    if f_name is not None:
        Path('data/{}'.format(f_name)).mkdir(parents=True, exist_ok=True)
        np.save(f'data/{f_name}/eval_policy_rewards.npy', ep_reward_cache)
