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
from utils.replay_buffer import ReplayBuffer, EvalBuffer
from utils import models
from utils import helper
from algos import sac
from utils import eval_utils
import time
import argparse



def run_pendulum():
    """
    For Pendulum, tasks are created as the following: 
    Dynamics change:
        gravity is changed from 10.0 to 18.0 gradually to change dynamics  
    Reward change: 
        no reward change for this environment
    Total tasks: 5 different dynamics = 5 tasks
    """

    gravity = [10.0, 12.0, 14.0, 16.0, 18.0]
    # g = gravity[3]
    for g in gravity:
        start = time.time()
        env_fn = lambda: gym.make("Pendulum-v1", g=g)
        # vanilla sac 
        sac.sac(env_fn, total_steps=10000, lr=1e-3, learning_starts=2000, update_every=25, 
                replay_buffer_size=10000, steps_per_epoch=100, max_ep_len=200,
                batch_size=64, hidden_size=200, num_grad_updates=25, num_test_episodes=10, 
                test_name=f'pendulum_g_{g}_sac_adversarial')

        finish = time.time()
        print(f'================================================')
        print(f'Total time: {(finish - start)/60} minutes')
        print(f'================================================')


def run_half_cheetah(task_type):
    """
    For half-cheetah, tasks are created as the following: 
    Dynamics change:
        joint stiffness is increased gradually from Task 0 to Task 4, 
        where Task 0 has the default gym values for joint stiffness 
    Reward change: 
        reward function is changed by encouraging the robot to wither move in forward 
        or rever direction, this is done by changing the values of forward_reward_weight
        from -2.0 to +3.0 
    Total tasks: 5 different dynamics + 5 different rewards - 1 common task = 9 tasks
    """
    xmls = [] 
    for k in range(1):
        xml = f'/home/ghost-083/Research/1_Transfer_RL/D3M/envs/halfcheetah/assets/half_cheetah_{k}.xml'
        xmls.append(xml)

    # forward_rewards = [1.0, 2.0, 3.0, -1.0, -2.0]
    forward_rewards = [2.0]  # default setting for reward

    for k in range(len(xmls)):
        start = time.time()
        if task_type == "dynamics":
            xml, xml_id = xmls[k], k
            f_reward = forward_rewards[0]  # do not change the reward
        elif task_type == "reward":
            xml, xml_id = xmls[0], 0  # do not change the dyamics 
            f_reward = forward_rewards[k]
        else:
            raise Exception('task type not correct!')

        env_fn = lambda: gym.make('HalfCheetah-v3', xml_file=xml, 
                    forward_reward_weight=f_reward, 
                    ctrl_cost_weight=0.1, 
                    reset_noise_scale=0.1
                    )

        # vanilla sac 
        sac.sac(env_fn, total_steps=10, lr=1e-3, learning_starts=5, update_every=5, 
                replay_buffer_size=10000, steps_per_epoch=100, max_ep_len=200,
                batch_size = 64, hidden_size=200, num_grad_updates=50, num_test_episodes=5, 
                test_name=f'ant_dyn_{xml_id}_sac')

        finish = time.time()
        print(f'================================================')
        print(f'Total time: {(finish - start)/60} minutes')
        print(f'================================================')


def run_ant():
    """
    For ant, tasks are created as the following: 
    Dynamics change:
        joint stiffness is increased gradually from Task 0 to Task 4, 
        where Task 0 has the default gym values for joint stiffness 
    Reward change: 
        no reward change is available at this moment 
    Total tasks: 5 different dynamics = 5 tasks
    """
    xml_id = 0
    xml = f'/home/ghost-083/Research/1_Transfer_RL/D3M/envs/Ant/assets/ant_{xml_id}.xml'

    start = time.time()
    env_fn = lambda: gym.make('Ant-v3', xml_file=xml)

    # vanilla sac 
    sac.sac(env_fn, total_steps=100000, lr=1e-3, learning_starts=10000, update_every=50, 
            replay_buffer_size=100000, steps_per_epoch=1000, max_ep_len=1000,
            batch_size=64, hidden_size=200, num_grad_updates=50, num_test_episodes=10, 
            test_name=f'ant_{xml_id}_sac')


    finish = time.time()
    print(f'================================================')
    print(f'Total time: {(finish - start)/60} minutes')
    print(f'================================================')




parser = argparse.ArgumentParser(description='Train SAC agent')
parser.add_argument('-env', '--env_name', type=str, required=True, help="environment name")
parser.add_argument('-t', '--task_type', type=str, required=False, help="task type")
args = parser.parse_args()


if __name__ == "__main__":
    if args.env_name == "pendulum":
        run_pendulum()
    elif args.env_name == "half_cheetah":
        run_half_cheetah(args.task_type)
    elif args.env_name == "ant":
        run_ant()    
    else:
        raise Exception('Not implemented yet!')