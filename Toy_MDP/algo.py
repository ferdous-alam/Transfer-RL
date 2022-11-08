import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import random
from tqdm import tqdm
from ToyMDP import ToyMDP as mdp
from ToyMDP import get_state_id, get_action_id, policy



def q_leanring(env, num_epochs):
    # Q-learning 
    num_episodes = 100

    # hyperparameters
    gamma = 0.98
    alpha = 0.5
    Q_vals = np.zeros((3, 2))
    reward_cache = []

    # main loop
    for ep in tqdm(range(num_episodes)):
        ep_reward = []
        state = env.reset()
        done = False
        while not done: 
            action = policy(state, Q_vals)
            next_state, reward = env.step(state, action)

            # save reward 
            ep_reward.append(reward)

            # get state and action ids
            s_id = get_state_id(state)
            s_next_id = get_state_id(next_state)    
            a_id = get_action_id(action)

            # TD update for Q-learning
            Q_vals[s_id, a_id] += alpha * (reward + gamma * np.argmax(Q_vals[s_next_id, :]) - Q_vals[s_id, a_id])
        
            if next_state == env.trap or next_state == env.goal:
                done = True
        
            # print(f'state:{state}, action:{action}, next_state:{next_state}, reward:{reward}')
            state = next_state
        
        mean_reward, var_reward = np.mean(ep_reward), np.std(ep_reward)
        reward_cache.append(mean_reward)
    return Q_vals
        