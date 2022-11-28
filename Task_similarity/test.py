import numpy as np
import gym
from gym.wrappers import Monitor
import time
import glfw
from tqdm import tqdm
import json
import codecs
import argparse

def test_env():
    xml_path = '/home/ghost-083/Research/1_Transfer_RL/Task_similarity/env_mods/Ant/assets/ant_7.xml'
    env = gym.make('Ant-v3', xml_file=xml_path)
    curr_obs = env.reset(seed=908778)   # make sure seed is same for all experiments
    terminated = False
    for k in range(200):
        env.render(mode="human")
        action = env.action_space.sample()
        next_obs, reward, terminated, info = env.step(action)
        curr_obs = next_obs                    
        time.sleep(0.01)

    # close renderer
    env.close()

    glfw.terminate()


if __name__ == "__main__":
    test_env()
