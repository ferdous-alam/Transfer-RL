import numpy as np
import gym
import time
import glfw
from tqdm import tqdm
import json
import codecs
import argparse
from environments.pendulum_mod import PendulumEnvMod


def create_dataset(env_name, num_traj, env_args=None, dataset_num=None, 
                    render=False, save_data=False, video_path=None):
    
    
    if env_name == "Pendulum":
        assert len(env_args) == 3, "missing environment arguments for Pendulum: g, gravity, Force"
        # env = PendulumEnvMod(g=g, gravity=gravity, force=force)  # instantiate environment
        env = gym.make('Pendulum-v1', 
                        g=env_args[0], 
                        gravity=env_args[1], 
                        force=env_args[2], video_path=video_path)
  
    elif env_name == "Ant":
        assert len(env_args) == 1, "ant environment only requires modified xml file path"
        xml_path = env_args[0]
        env = gym.make('Ant-v3', xml_file=xml_path)

    elif env_name == 'Cartpole':
        assert len(env_args) == 3, "missing environment arguments for Pendulum: pole_length, mass_car, mass_pole, gravity""
        env = gym.make('CartPole-v1', pole_length=env_args[0], mass_cart=env_args[1], 
            mass_pole=env_args[2], gravity=env_args[3])

    else:
        raise Exception('Only the following modified environments are supported at this moment: Pendulum, Ant, Cartpole')

    train_data = []  # training data to collect for model training

    for k in tqdm(range(num_traj)):
        curr_obs = env.reset()   # make sure seed is same for all experiments
        terminated = False
        traj_data = []  # trajectory data
        m = 0
        while not terminated:
            if render:
                env.render()
            action = env.action_space.sample()
            next_obs, reward, terminated, info = env.step(action)
            # data = (state, action, next_state, reward)
            data = [curr_obs.tolist(), action.tolist(),
                    next_obs.tolist(), reward,
                    terminated, info]
            traj_data.append(data)
            curr_obs = next_obs
                        
            if render:
                time.sleep(0.01)
            
        train_data.append(traj_data)

    # close renderer
    if render:
        env.close()
        glfw.terminate()

    # save as json file
    dataset = {'env': env_name,
               'env arguments': env_args,
               'num_traj': num_traj,
               'data': train_data}

    if save_data == True:
        json.dump(dataset, codecs.open('/media/ghost-083/SolarSystem1/1_Research/Transfer-RL/Task_similarity/dataset/{}_dataset_{}.json'.format(env_name, dataset_num), 'w', encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4)


parser = argparse.ArgumentParser(description='Dataset collection for RL task similarity')
parser.add_argument('-e', '--env_name', type=str, required=True, help='gym environment name')
parser.add_argument('-n', '--num_traj', type=int, required=True, help='number of trajectories')
parser.add_argument('-a', '--env_args', nargs='*', action='store', type=float, required=False, help='environment arguments')
parser.add_argument('-dn', '--dataset_num', type=int, required=True, help='dataset file number')
parser.add_argument('-r', '--render', type=bool, required=False, help='render option, default is false')
parser.add_argument('-sd', '--save_data', type=bool, required=False, help='save data option, default is false')
parser.add_argument('-video', '--video_path', type=str, required=False, help='path to save video, default is None')
args = parser.parse_args()

if __name__ == "__main__":
    create_dataset(args.env_name, args.num_traj, args.env_args,
                   args.dataset_num, args.render, args.save_data, 
                   args.video_path)
