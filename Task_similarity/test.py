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
    env = gym.make('CartPole-v1', pole_length=1.0, mass_cart=1.0, 
                mass_pole=0.1, gravity=3*9.8, 
                video_path='video/cartpole_5')
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
