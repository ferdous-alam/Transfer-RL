import numpy as np
import gym
import time
import glfw
from tqdm import tqdm
import json
import codecs
import argparse


def create_dataset(env_name, num_traj, env_args=None, xml_path=None, dataset_num=None, 
                    render=False, save_data=False, dtype=None, video_path=None):
    
    if env_name == "Pendulum":
        assert len(env_args) == 3, "missing environment arguments for Pendulum: g, gravity, Force"
        env = gym.make('Pendulum-v1', 
                        g=env_args[0], 
                        gravity=env_args[1], 
                        force=env_args[2], video_path=video_path)
  
    elif env_name == "Ant":
        assert xml_path != None, "ant environment only requires modified xml file path"
        xml_path = xml_path
        env = gym.make('Ant-v3', xml_file=xml_path)

    elif env_name == 'Cartpole':
        assert len(env_args) == 4, "missing environment arguments for Cartploe: pole_length, mass_cart, mass_pole, gravity"
        env = gym.make('CartPole-v1', pole_length=env_args[0], mass_cart=env_args[1], 
            mass_pole=env_args[2], gravity=env_args[3])

    elif env_name == 'Halfcheetah':
           assert xml_path != None, "ant environment only requires modified xml file path"
           assert len(env_args) == 3, "missing environment arguments for halfcheetah: forward_reward_weight, ctrl_cost_weight, reset_noise_scale"
           env = gym.make('HalfCheetah-v3', xml_file=xml_path, 
                        forward_reward_weight=env_args[0], 
                        ctrl_cost_weight=env_args[1], 
                        reset_noise_scale=env_args[2]
                        )
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
            if env_name == "Pendulum" or env_name == "Halfcheetah":
                data = [curr_obs.tolist(), action.tolist(),
                        next_obs.tolist(), reward,
                        terminated, info]
            elif env_name == "Cartpole":
                data = [curr_obs.tolist(), [action],
                        next_obs.tolist(), reward,
                        terminated, info]

            else:
                raise Exception('Wrong environment name!')            
            traj_data.append(data)
            curr_obs = next_obs
                        
            if render:
                time.sleep(0.005)
            
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
        json.dump(dataset, codecs.open('/media/ghost-083/SolarSystem1/1_Research/00_Transfer-RL/Task_similarity/dataset/{}/{}_dataset_{}_{}.json'.format(env_name, env_name, dataset_num, dtype), 'w', encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4)


parser = argparse.ArgumentParser(description='Dataset collection for RL task similarity')
parser.add_argument('-e', '--env_name', type=str, required=True, help='gym environment name')
parser.add_argument('-n', '--num_traj', type=int, required=True, help='number of trajectories')
parser.add_argument('-a', '--env_args', nargs='*', action='store', type=float, required=False, help='environment arguments')
parser.add_argument('-x', '--xml_path', type=str, required=False, help='environment mod xml path')
parser.add_argument('-dn', '--dataset_num', type=int, required=True, help='dataset file number')
parser.add_argument('-r', '--render', type=bool, required=False, help='render option, default is false')
parser.add_argument('-sd', '--save_data', type=bool, required=False, help='save data option, default is false')
parser.add_argument('-dt', '--dtype', type=str, required=True, help='dataset type: train or test, default is None')
parser.add_argument('-video', '--video_path', type=str, required=False, help='path to save video, default is None')
args = parser.parse_args()

if __name__ == "__main__":
    # run pendulum ---
    # python create_dataset.py -e 'Pendulum' -n 100 -a 'env_mods/Ant/assets/ant_0.xml' -dn 0 -sd True
    # run cartpole ---
    # python create_dataset.py -e 'Cartpole' -n 100 -a 'env_mods/Ant/assets/ant_0.xml' -dn 0 -sd True
    # run ant ---
    # python create_dataset.py -e 'Ant' -n 100 -x '/home/ghost-083/Research/1_Transfer_RL/Task_similarity/env_mods/Ant/assets/ant_0.xml' -dn 0 -sd True

    ## --- run half-cheetah ----
    # python create_dataset.py -e 'Halfcheetah' -n 100 -a 1.0 0.1 0.1 -x '/home/ghost-083/Research/1_Transfer_RL/Task_similarity/env_mods/half-cheetah/assets/half_cheetah_0.xml' -dn 0 -dt 'train' -sd True

    create_dataset(args.env_name, args.num_traj, args.env_args, args.xml_path,
                   args.dataset_num, args.render, args.save_data, args.dtype,
                   args.video_path)
