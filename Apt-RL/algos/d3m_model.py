# This implementation is based on the Soft-actor-critic implementation by OpenAI spinning up DeepRL

import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from utils import core
from utils.replay_buffer import ReplayBuffer, EvalBuffer, ModelBuffer
from utils import models
from utils import helper
import itertools
import time
from pathlib import Path
from algos.sac_utils import compute_loss_q, compute_loss_pi, update, test_agent, get_action, relabel_batch, perform_model_rollout
from algos.model_train_utils import update_models
import logging 



def d3m(env_fn, total_steps, lr=1e-3, learning_starts=2000, update_every=50, 
        replay_buffer_size=10000, steps_per_epoch=500, max_ep_len=500,
        batch_size=64, hidden_size=200, num_grad_updates=50, num_test_episodes=1, 
        test_name=None, pretrain_path=None, source_path=None, 
        source_bias_stops=0, beta=0.0, update_model_step=100):
  """
  Args: 
    env_fn: 
    total_steps: 
    lr: 
    learning starts: 
    update_every: 
    replay_buffer_size: 
    update_every:   
    steps_per_epoch: 
    max_ep_len: 
    batch_size: 
    num_grad_updates: 
    num_test_episodes: 
    test_name: 
    pretrain_name: 
    source_path:
    source_bias_steps:
    test_ep_ret_cache:
    beta: 
  
  Outputs: 
    test_name.pth: 
    ep_ret_cache: 
    steps_cache: 
    test_ep_ret_cache:

  """


  # random seeds
  # torch.manual_seed(seed)
  # np.random.seed(seed)
  # ------------- logging ----------------
  Path('data/{}'.format(test_name)).mkdir(parents=True, exist_ok=True)
  logging.basicConfig(filename=f'data/{test_name}.log', filemode='w', 
                      format='%(message)s', level=logging.INFO)
  logging.info('Experiment name: %s ------- [%s]', test_name, time.asctime(time.localtime()))
  logging.info('-------------------------------------------------------------------') 
  logging.info('Replay buffer size: %i', replay_buffer_size)
  logging.info('learning starts after : %i steps', learning_starts) 
  logging.info('Update models in every: %i steps', update_every) 
  logging.info('Number of gradient update at this step: %i', num_grad_updates) 
  logging.info('Number of evaluation episodes after every %i steps: %i', steps_per_epoch, num_test_episodes)   
  if source_path is not None:
    knowledge_transfer = True
  else:
    knowledge_transfer = False
  logging.info('====== Knowledge transfer: %r ======', knowledge_transfer) 
  logging.info('source bias stops at: %i steps', source_bias_stops) 
  if pretrain_path is not None:
    target_pretrain = True
  else:
    target_pretrain = False
  logging.info('Target pretraining: %r ',  target_pretrain) 

  # initialize environment and make a copy
  env, test_env = env_fn(), env_fn()

  # initialize policy network
  actor_critic = core.MLPActorCritic

  # environment specs: obs, act dimensions    
  obs_dim = env.observation_space.shape
  act_dim = env.action_space.shape[0]
  act_limit = env.action_space.high[0]

  # get feature dimensions for model
  in_features = obs_dim[0] + act_dim
  out_dyn_features = obs_dim[0]
  out_rew_features = 1

  # Create actor-critic module and target networks
  ac_kwargs = dict(hidden_sizes=[hidden_size]*4)
  ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
  # load source policy 
  if source_path is not None:
    ac_source = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_source.load_state_dict(torch.load(source_path))
  else:
    ac_source = None
  # check is source policy is available 
  if pretrain_path is not None: 
    ac.load_state_dict(torch.load(pretrain_path))
  ac_targ = deepcopy(ac)

  # Freeze target networks with respect to optimizers (only update via polyak averaging)
  for p in ac_targ.parameters():
      p.requires_grad = False

  # List of parameters for both Q-networks (save this for convenience)
  q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

  # Count variables (protip: try to get a feel for how different size networks behave!)
  var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])


  # buffer
  replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_buffer_size)
  
  # rew_model.load_state_dict(torch.load(rew_path))
  logging.info('model loaded successfully!') 

  # Set up optimizers for policy and q-function
  pi_optimizer = optim.Adam(ac.pi.parameters(), lr=lr)
  q_optimizer = optim.Adam(q_params, lr=lr)

  # Prepare for interaction with environment
  o, ep_ret, ep_len = env.reset(), 0, 0

  # params
  polyak = 0.995
  device = torch.device("cpu")
  gamma = 0.99
  alpha = 0.2
  res, ep_ret_cache, steps_cache, test_ep_ret_cache = [], [], [], []
  knowledge_transfer = True
  adv_val = []

  # Main loop: collect experience in env and update/log each epoch
  pbar = tqdm(range(total_steps))
  for t in pbar:      
    # Until start_steps have elapsed, randomly sample actions
    # from a uniform distribution for better exploration. Afterwards, 
    # use the learned policy. 
    if t >= learning_starts:
        a = get_action(o, ac)
    else:
        a = env.action_space.sample()

    # Step the env
    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d = False if ep_len == max_ep_len else d

    # Store experience to replay buffer
    replay_buffer.store(o, a, r, o2, d)

    # Super critical, easy to overlook step: make sure to update 
    # most recent observation!
    o = o2

    # End of trajectory handling
    if d or (ep_len == max_ep_len):
        ep_ret_cache.append(ep_ret)
        steps_cache.append(t)
        o, ep_ret, ep_len = env.reset(), 0, 0

    # Update handling
    # check if it is time to update the model
    # when random exploration stops, we must upadte the model 
    # immediately, otherwise model will give inaccurate values 
    
    if t >= learning_starts and t % update_every == 0:
      if t == learning_starts:
        dyn_model, rew_model = update_models(env, replay_buffer)

      num_model_rollout = 2000
      model_buffer = perform_model_rollout(env, num_model_rollout, 
                                          dyn_model, rew_model, 
                                          replay_buffer, device)
                          
      if t >= source_bias_stops:
        knowledge_transfer = False
    
      #  Fit dynamics and reward model
      # logging.info('    fitting decoupled models')
      
      # # first relabel previous samples using th eupdated model
      # model_buffer.relabel(dyn_model, rew_model)

      # perform gradient updates 
      # logging.info('    performing gradient updates') 
      for j in range(num_grad_updates):
        batch = model_buffer.sample_batch(batch_size=batch_size)
        update(batch, q_optimizer, 
              pi_optimizer, q_params, 
              ac, ac_targ, gamma, alpha, 
              polyak, ac_source=ac_source, beta=beta, 
              knowledge_transfer=knowledge_transfer, 
              adv_val=adv_val)
      
    # End of epoch handling
    if (t+1) % steps_per_epoch == 0:
        # Test the performance of the deterministic version of the agent.
        test_ep_ret = test_agent(test_env, ac, num_test_episodes, max_ep_len)
        test_ep_ret_cache.append(test_ep_ret)
        logging.info('step %i/%i, avs test return %f', t, total_steps, np.mean(test_ep_ret)) 
        # logging.info('    eval loss: %d', np.mean(test_ep_ret)) 
        pbar.set_description("ep ret: %s" % str(np.mean(test_ep_ret)))  # update tqdm bar

  # save data
  path = 'data/{}/policy.pth'.format(test_name)
  torch.save(ac.state_dict(), path) 
  np.save('data/{}/ep_ret_cache.npy'.format(test_name), ep_ret_cache)
  np.save('data/{}/steps_cache.npy'.format(test_name), steps_cache)
  np.save('data/{}/test_ep_ret_cache.npy'.format(test_name), test_ep_ret_cache)
  np.save('data/{}/adv.npy'.format(test_name), adv_val)
 
