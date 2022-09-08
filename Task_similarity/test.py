import numpy as np
import gym
import time
import glfw
from tqdm import tqdm
import json
import codecs
import argparse


def create_dataset(env_name, num_traj, dataset_num=None, rand_seed=42, render=False):
    env = gym.make(env_name)  # instantiate environment
    render = False   # renderer
    train_data = []  # training data to collect for model training

    for k in tqdm(range(num_traj)):
        curr_obs = env.reset(seed=rand_seed)   # make sure seed is same for all experiments
        terminated = False
        traj_data = []  # trajectory data
        while not terminated:
            if render:
                env.render()
            action = env.action_space.sample()
            next_obs, reward, terminated, info = env.step(action)
            # data = (state, action, next_state, reward)
            data = [curr_obs.tolist(), action.tolist(),
                    next_obs.tolist(), reward,
                    terminated]
            traj_data.append(data)
            if render:
                time.sleep(0.01)
        train_data.append(traj_data)

    # close renderer
    if render:
        env.close()
        glfw.terminate()

    # save as json file
    dataset = {'env': env_name,
               'seed': rand_seed,
               'dataset': train_data}

    json.dump(dataset, codecs.open('data/{}_dataset_{}.json'.format(env_name, dataset_num), 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)


parser = argparse.ArgumentParser(description='Dataset collection for RL task similarity')
parser.add_argument('-e', '--env_name', type=str, required=True, help='gym environment name')
parser.add_argument('-n', '--num_traj', type=int, required=True, help='number of trajectories')
parser.add_argument('-dn', '--dataset_num', type=int, required=True, help='dataset file number')
parser.add_argument('-s', '--rand_seed', type=int, required=False, help='random seed value, default is 42')
parser.add_argument('-r', '--render', type=bool, required=False, help='render option, default is false')
args = parser.parse_args()

if __name__ == "__main__":
    create_dataset(args.env_name, args.num_traj,
                   args.dataset_num, args.rand_seed,
                   args.render)
