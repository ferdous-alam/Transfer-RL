import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import random
from tqdm import tqdm
from ToyMDP import ToyMDP as mdp
from ToyMDP import get_state_id, get_action_id, policy



def q_learning(env, num_episodes, epsilon=.05, Q_vals=None):
    # Q-learning 
    # hyperparameters
    gamma = 0.98
    alpha = 0.5
    if Q_vals is None:
        Q_vals = np.zeros((3, 2))
        
    reward_cache = []
    Q1, Q2, Q3, Q4, Q5, Q6 = [], [], [], [], [], []

    # main loop
    for ep in tqdm(range(num_episodes)):
        # if ep > 50:
        #     epsilon = epsilon * 0.9
        ep_reward = []
        state = env.reset()
        done = False
        while not done: 
            action = policy(state, Q_vals, epsilon)
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

        # save Q-values
        Q1.append(Q_vals[0][0])
        Q2.append(Q_vals[0][1])
        Q3.append(Q_vals[1][0])
        Q4.append(Q_vals[1][1])
        Q5.append(Q_vals[2][0])
        Q6.append(Q_vals[2][1])

        
        mean_reward, var_reward = np.mean(ep_reward), np.std(ep_reward)
        reward_cache.append(mean_reward)
    return Q_vals, reward_cache, Q1, Q2, Q3, Q4, Q5, Q6
        