import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from utils import core
from utils.replay_buffer import ReplayBuffer, EvalBuffer
from utils import models
from utils import helper
from pathlib import Path
import itertools
import time


def compute_advantage(data, ac, ac_targ, alpha, ac_source=None):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    if ac_source:
        a, logp_a = ac_source.pi(o)  # this is important, choose action using source policy    # calculate action value function
    else:
        a, logp_a = ac.pi(o)
    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)
    action_func = torch.min(q1, q2)

    # calculate value function
    q1_pi_targ = ac_targ.q1(o, a)
    q2_pi_targ = ac_targ.q2(o, a)
    q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
    value_func = q_pi_targ - alpha * logp_a

    # get advantage
    advantage = action_func - value_func

    return advantage


def get_total_loss(data, ac, ac_source, ac_targ, loss_pi, alpha, knowledge_transfer, beta=0.0, 
                    adv_val=None):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    
    # advantage threshold for max value
    adv_threshold = beta
    # Source-regularized cross-entropy loss 
    pi_source, logp_pi_source = ac_source.pi(o)
    pi_curr, logp_pi_curr = ac.pi(o)   
    # regularize loss
    loss_reg = nn.BCELoss()(nn.Sigmoid()(pi_curr), pi_source).mean()

    if knowledge_transfer:
        # loss_pi = loss_pi + beta * loss_reg
        loss_pi = loss_pi
        adv_diff = 0.0
        adv_val.append(adv_diff)
    else:
        # calculate advantage using source policy and current networks
        adv_source = compute_advantage(data, ac, ac_targ, alpha, ac_source=ac_source)
        adv_actual = compute_advantage(data, ac, ac_targ, alpha)
        val = torch.mean(adv_source) - torch.mean(adv_actual)
        # if val >= 0:
        #     adv_diff = torch.exp(val)   # check if source advantage is better
        # else:
        #     adv_diff = torch.tensor(0.0)
        adv_diff = torch.exp(val)   # check if source advantage is better
        adv_diff = torch.tensor(adv_threshold) if adv_diff > adv_threshold else adv_diff
        loss_pi = loss_pi + adv_diff * loss_reg
        adv_val.append(adv_diff.item())    
    return loss_pi




# Set up function for computing SAC Q-losses
def compute_loss_q(data, ac, ac_targ, gamma, alpha):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        a2, logp_a2 = ac.pi(o2)

        # Target Q-values
        q1_pi_targ = ac_targ.q1(o2, a2)
        q2_pi_targ = ac_targ.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(Q1Vals=q1.detach().numpy(),
                    Q2Vals=q2.detach().numpy())

    return loss_q, q_info


# Set up function for computing SAC pi loss
def compute_loss_pi(data, ac, alpha):
    o = data['obs']
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, pi_info


def update(data, q_optimizer, pi_optimizer, q_params, 
        ac, ac_targ, gamma, alpha, polyak, 
        ac_source, beta=0.0, knowledge_transfer=False, adv_val=None):

    """
    ac_source: this determines whether source knowledge is utilized

    """

    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = compute_loss_q(data, ac, ac_targ, gamma, alpha)
    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-networks so you don't waste computational effort 
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    # to train a random agent 
    # loss_pi = torch.rand(1, requires_grad=True)
    loss_pi, pi_info = compute_loss_pi(data, ac, alpha)
    if ac_source is not None:
       loss_pi = get_total_loss(data, ac, ac_source, ac_targ, loss_pi, alpha, 
                                knowledge_transfer, beta, adv_val=adv_val)

    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)


def update_source(data, q_source_optimizer, pi_source_optimizer, q_source_params, 
        ac_source, ac_source_targ, gamma, alpha, polyak):

    """
    ac_source: this determines whether source knowledge is utilized

    """

    # First run one gradient descent step for Q1 and Q2
    q_source_optimizer.zero_grad()
    loss_source_q, q_source_info = compute_loss_q(data, ac_source, ac_source_targ, gamma, alpha)
    loss_source_q.backward()
    q_source_optimizer.step()

    # Freeze Q-networks so you don't waste computational effort 
    # computing gradients for them during the policy learning step.
    for p in q_source_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_source_optimizer.zero_grad()
    loss_pi_source, pi_source_info = compute_loss_pi(data, ac_source, alpha)

    loss_pi_source.backward()
    pi_source_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_source_params:
        p.requires_grad = True

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac_source.parameters(), ac_source_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

    
def get_action(o, ac, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                    deterministic)



def test_agent(test_env, ac, 
              num_test_episodes, 
              max_ep_len):
    ep_ret_cache = []
    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 
            o, r, d, _ = test_env.step(get_action(o, ac, True))
            ep_ret += r
            ep_len += 1
        ep_ret_cache.append(ep_ret)
    return ep_ret_cache


def perform_model_rollout(env, num_model_rollout, 
                        dyn_model, rew_model, 
                        replay_buffer, device):
    # environment specs: obs, act dimensions    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # get feature dimensions for model
    in_features = obs_dim[0] + act_dim
    out_dyn_features = obs_dim[0]
    out_rew_features = 1
    model_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=num_model_rollout)     
    sample = replay_buffer.sample_batch(batch_size=num_model_rollout)
    obs = sample['obs']
    act = sample['act']
    done = sample['done']
    raw_input = torch.cat((obs, act), dim=1)
    raw_input.to(device)
    with torch.no_grad():
        obs2 = dyn_model(raw_input)
        rew = rew_model(raw_input)
      
    for k in range(num_model_rollout):
        model_buffer.store(obs[k], act[k], rew[k], obs2[k], done[k])
    return model_buffer


def relabel_batch(data, dyn_model, rew_model):
    obs = data['obs']
    act = data['act']
    done = data['done']
    raw_input = torch.cat((obs, act), dim=1)
    with torch.no_grad():
        obs2_model = dyn_model(raw_input)
        rew_model = rew_model(raw_input)
    pred = {'obs': obs, 
            'act': act, 
            'obs2': obs2_model, 
            'rew': rew_model, 
            'done': done}
    return pred
