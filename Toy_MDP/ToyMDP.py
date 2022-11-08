import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import random


def get_state_id(state):
    if state == -1: 
        return 0 
    elif state == 0:
        return 1
    else:
        return 2


def get_action_id(action):
    if action == -1:
        return 0
    elif action == +1:
        return 1

def get_state_action_id(rewards, state, action):
    if state == -1 and action == -1:
        return rewards[0]
    elif state == -1 and action == +1:
        return rewards[1]
    elif state == 0 and action == -1:
        return rewards[2]
    elif state == 0 and action == +1:
        return rewards[3]
    elif state == 1 and action == -1:
        return rewards[4]
    elif state == 1 and action == +1:
        return rewards[-1]
    else:
        raise ValueError('wrong state or action!')

    return 


def policy(state, Q, epsilon=0.05):
    """
    epsilon-greedy policy
    """

    if np.random.rand() < epsilon: 
        # random action
        action = random.choice([1, -1])
    else:
        # greedy action
        action = np.argmax(Q[state, :])
        if action == 0: 
            action = -1  # need to adjust action for index
        else:
            action = +1

    return action


class ToyMDP:
    def __init__(self, reward_weight, uncertain_prob=None):
        """
        
        """
        
        self.reward_weight = reward_weight
        self.uncertain_prob = uncertain_prob
        self.s0 = 0
        self.s1 = 1
        self.s2 = -1
        self.goal = 2
        self.trap = -2
        self.state_space = [self.trap, self.s2, self.s0, self.s1, self.goal]        
        self.action_space = [1, -1]
        self.a_r = 1
        self.a_l = -1
        self.actions = [self.a_r, self.a_l] # right, left
        self.reward_weight = reward_weight
        # reward features
        self.reward_features = np.array([[self.s2, self.a_l],                                      
                                        [self.s2, self.a_r], 
                                        [self.s0, self.a_l],
                                        [self.s0, self.a_r], 
                                        [self.s1, self.a_l],
                                        [self.s1, self.a_r]
                                        ])
    
    def reset(self):
        state = 0   # starting state
        return state

    def sample_action(self):
        sample_action = random.choice(self.actions)
        return sample_action

    
    def reward_func(self):
        reward_weight = np.array(self.reward_weight).reshape(-1, 1)
        rewards = np.matmul(self.reward_features, self.reward_weight)
        rewards += 3  # add bias to make reward positive
        rewards = rewards / np.max(rewards) 
        rewards = np.squeeze(rewards).tolist()
        return rewards

    def dynamics(self, state, action):
        if self.uncertain_prob == 'source':
            action = action
        elif self.uncertain_prob == 'target':
            action = -action
        else:
            raise Exception('wrong dynamics!')

        next_state = state + action
        next_state = min(max(next_state, -2), 2)
        return next_state

    def step(self, state, action):
        # take action
        rewards = self.reward_func()
        next_state = self.dynamics(state, action)
        reward = get_state_action_id(rewards, state, action)
        return next_state, reward

    def plot_rewards(self, rewards_cache=None):
        plt.rcParams['text.usetex'] = True
        rewards = self.reward_func()
        plt.figure(figsize=(10, 4))
        x = [i for i in range(len(rewards))]
        plt.bar(np.array(x), rewards, width=0.25, ls='--', 
                lw=2.0, hatch='/', edgecolor='black', 
                fill=False, label='$T_0$')

        if rewards_cache is not None:
            for k in range(len(rewards_cache)):
                plt.bar(np.array(x)+k*0.10, rewards_cache[k], width=0.25, ls='--', lw=2.0, fill=True, label=f'$T_{k+1}$')
            

        plt.ylim([0, 1.10])
        plt.xticks(x, ['$\mathcal{R}(s_2, a)$', '$\mathcal{R}(s_2, b)$', '$\mathcal{R}(s_0, a)$',
                       '$\mathcal{R}(s_0, b)$', '$\mathcal{R}(s_1, a)$', '$\mathcal{R}(s_1, b)$'])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.rcParams['axes.linewidth'] = 2.0
        plt.legend(fontsize=18, ncol=2)
        plt.savefig('figures/rewards.png', dpi=300)




