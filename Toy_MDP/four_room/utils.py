import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from four_room.four_room import FourRooms, Visualizations



def policy(q_table, q_dict, state, epsilon=0.05):
    if np.random.uniform() <= epsilon:
        return np.random.choice(4) 
    else:
        id = list(q_dict.values()).index(list(state))
        return np.argmax(q_table[id])

def greedy_policy(q_table, q_dict, state):
    id = list(q_dict.values()).index(list(state))
    return np.argmax(q_table[id])

def execute_greedy_policy(q_table, q_dict, env, viz=None, 
                        viz_policy=False, save=None, fig_name=None, 
                        fignum=1):
    state = env.reset()
    states, actions = [], []
    done = False
    rewards = 0
    ep_len = 0
    while (ep_len < 100) and not done:
        action = greedy_policy(q_table, q_dict, state)
        next_state, reward, done = env.step(action)
        rewards += reward
        ep_len += 1
        states.append(state)
        action_val = env.actions[action]
        actions.append(action_val)
        state = next_state

    if viz_policy:
        viz.policy_visualization(states=states, actions=actions, 
                                save=save, fig_name=fig_name, fignum=fignum)
    return states, actions, rewards

def initialize_q_table(env):
    q_dict, val = {}, 0
    temp = [[i, j] for i in range(11) for j in range(11)]
    q_table = []
    for k in temp:
        if k not in env.walls:
            q_dict[val] = k 
            q_table.append(np.zeros(4))
            val += 1
    return q_dict, q_table


def train(env, num_steps, q_pretrained=None, env_type=None, eval_step=100000, fignum=1):
    q_dict, q_table = initialize_q_table(env)

    if q_pretrained is not None:
        q_table = q_pretrained        

    alpha, gamma = 0.5, 0.98
    rewards = []
    rewards_opt = []
    episode_len = []
    ep_rewards = []
    
    for ep in tqdm(range(num_steps)): 
        state = env.reset()
        done = False
        t = 0
        while not done:
            t += 1 
            action = policy(q_table, q_dict, state, epsilon=0.25)
            next_state, reward, done = env.step(action)
            id_curr = list(q_dict.values()).index(list(state))
            id_next = list(q_dict.values()).index(list(next_state)) 
            best_next_action = np.argmax(q_table[id_next])
            q_table[id_curr][action] += alpha * (reward + gamma * q_table[id_next][best_next_action] - q_table[id_curr][action])
            state = next_state
        episode_len.append(t)
        _, _, rewards = execute_greedy_policy(q_table, q_dict, env, viz=None, viz_policy=False, 
                                            fignum=fignum)
        ep_rewards.append(rewards)

    # state = env.reset()
    # done = False    
    # t = 0
    # ep = 0
    # ep_rewards = []
    # eval_rewards_cache = []
    # ep_rew = 0

    # for t in tqdm(range(num_steps)): 
    #     t += 1
    #     action = policy(q_table, q_dict, state, epsilon=0.10)
    #     next_state, reward, done = env.step(action)
    #     ep_rew += reward
    #     id_curr = list(q_dict.values()).index(list(state))
    #     id_next = list(q_dict.values()).index(list(next_state)) 
    #     best_next_action = np.argmax(q_table[id_next])
    #     q_table[id_curr][action] += alpha * (reward + gamma * q_table[id_next][best_next_action] - q_table[id_curr][action])
    #     state = next_state
    #     if done:
    #         state = env.reset()
    #         ep_rewards.append(ep_rew)
    #         ep_rew = 0
    #         ep += 1

    #     if t % eval_step == 0: 
    #         _, _, eval_rewards = execute_greedy_policy(q_table, q_dict, env)
    #         eval_rewards_cache.append(eval_rewards)

    return q_table, q_dict, ep_rewards



def run_env(env, viz, show=False, num_episodes=1000, q_pretrained=None, fig_name=None, fignum=1):
    
    # train target
    q_table, q_dict, ep_rewards = train(env, num_steps=num_episodes, 
                                            q_pretrained=q_pretrained, eval_step=25)

    if show:
        viz.four_rooms_viz(fig_name=fig_name, save='True')
        plot_name = fig_name + '_optimal'
        s, a, r = execute_greedy_policy(q_table, q_dict, env, 
                                        viz=viz, viz_policy=True, 
                                        save=True, fig_name=plot_name, 
                                        fignum=fignum)

    return q_table, ep_rewards

def smooth(y, window=1000):
    box = np.ones(window)/window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_stat(env_i, env_j, env_k, num_iter=5):
    rho_s_cache = []
    rho_t1_scratch_cache = []
    rho_t1_prior_cache = []
    rho_t2_scratch_cache = []
    rho_t2_prior_cache = []

    for k in range(num_iter):
        print(f'iteration: {k+1}')
        q_s, rho_s = run_env(env_i, viz=None, q_pretrained=None, fig_name='toy_0', fignum=1)
        q_t1_scratch, rho_t1_scratch = run_env(env_j, viz=None, q_pretrained=None, 
                                            fig_name='toy_1', fignum=1)
        
        q_t1_prior, rho_t1_prior = run_env(env_j, viz=None, q_pretrained=q_s, 
                                            fig_name='toy_2', fignum=1)


        q_t2_scratch, rho_t2_scratch = run_env(env_k, viz=None, q_pretrained=None, 
                                            fig_name='toy_1', fignum=1)
        
        q_t2_prior, rho_t2_prior = run_env(env_k, viz=None, q_pretrained=q_s, 
                                            fig_name='toy_2', fignum=1)


        rho_s_cache.append(rho_s)
        rho_t1_scratch_cache.append(rho_t1_scratch)
        rho_t1_prior_cache.append(rho_t1_prior)
        rho_t2_scratch_cache.append(rho_t2_scratch)
        rho_t2_prior_cache.append(rho_t2_prior)
    return rho_s_cache, rho_t1_scratch_cache, rho_t1_prior_cache, rho_t2_scratch_cache, rho_t2_prior_cache


def plot_rho(*args, **kwargs):
    # plot
    fmt = kwargs['fmt']
    colors = kwargs['colors']
    labels = kwargs['labels']
    line_width = 3.0
    capsize = 5.0
    elinewidth = 2.0

    plt.figure(figsize=(8, 6))
    for k in range(len(args)):
        mean_r = np.mean(args[k], 0)
        std_r = np.std(args[k], 0)
        smooth_mean = smooth(mean_r, window=25)[:200]
        smooth_std = smooth(std_r, window=25)[:200]
        mean_val = smooth_mean[::10]
        std_val = smooth_std[::10]
        x_axis = [k for k in range(len(mean_val))]    
        plt.errorbar(x_axis, mean_val, yerr=std_val, fmt=fmt[k], ms=10, capsize=capsize, elinewidth=elinewidth, color=colors[k], label=labels[k])
        plt.plot(x_axis, mean_val, lw=line_width, color=colors[k])

    plt.xlabel(r'episode', fontsize=24)
    plt.ylabel(r'$\rho_t$', fontsize=30)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.rcParams['axes.linewidth'] = 2.0 
    plt.legend(fontsize=22)
    plt.savefig('figures/plot_rho.pdf', bbox_inches='tight', dpi=300)


def plot_tau(*args, **kwargs):
    # plot
    plt.figure(figsize=(8, 6))
    x_axis = [k for k in range(200)]

    labels = kwargs['labels']
    source_data = kwargs['source_data']
    fmt = kwargs['fmt']
    colors = kwargs['colors']

    for k in range(len(args)):
        mean_r = np.mean(args[k], 0)
        std_r = np.std(args[k], 0)
        mean_r_b = np.mean(source_data[k], 0)
        std_r_b = np.std(source_data[k], 0)        
        smooth_mean = smooth(mean_r, window=25)[:200]
        smooth_std = smooth(std_r, window=25)[:200]       
        smooth_b_mean = smooth(mean_r_b, window=25)[:200]
        smooth_b_std = smooth(std_r_b, window=25)[:200]
        tau_mean = smooth_mean - smooth_b_mean
        tau_std = smooth_std - smooth_b_std
        
        plt.plot(x_axis[::5], tau_mean[::5], marker=fmt[k], ms=10, 
                lw=3.0, label=labels[k], color=colors[k])
   
    plt.xlabel(r'episode', fontsize=24)
    plt.ylabel(r'$\tau_t$', fontsize=30)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.rcParams['axes.linewidth'] = 2.0 
    plt.legend(fontsize=18)
    plt.savefig('figures/plot_tau.pdf', bbox_inches='tight', dpi=300)