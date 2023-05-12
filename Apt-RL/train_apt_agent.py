import gym
from algos import sac, apt, apt_fixed_source
import time
import argparse



def run_pendulum():
    """
    For Pendulum, tasks are created as the following: 
    Dynamics change:
        gravity is changed from 10.0 to 18.0 gradually to change dynamics  
    Reward change: 
        no reward change for this environment
    Total tasks: 5 different dynamics = 5 tasks
    
    NOTE: 
    -----
        in this case this is the policy trained for 20000 timesteps using 
        vanilla SAC and gravity = 10.0, source policy is 
        fixed for all experiments
        we do not run algos on envrionment with gravity = 18.0 as SAC cannot solve '
        the task with current action space
    """

    gravity = [16.0]
    # path to optimal source policy 
    source_path = 'data/pendulum_source_optimal/policy.pth'

    # parameters:
    # ---------------------------
    source_path = source_path                                      
    beta = 100.0
    # ----------------------------              
    total_steps = 5000                                     # fixed for all tasks
    lr = 1e-3                                               # fixed for all tasks
    learning_starts = 2000                                  # fixed for all tasks
    update_every = 25                                       # fixed for all tasks
    replay_buffer_size = 10000                              # fixed for all tasks
    eval_steps_per_epoch = 100                              # fixed for all tasks
    max_ep_len = 200                                        # fixed for all tasks
    batch_size = 64                                         # fixed for all tasks
    hidden_size = 200                                       # fixed for all tasks
    num_grad_updates = 25                                   # fixed for all tasks
    num_test_episodes = 10                                  # fixed for all tasks
    beta = 100.0                                            # fixed for all tasks


    # run algo 
    for g in gravity:
        test_name = f'pendulum_apt_g_{g}_fixed'
        start = time.time()
        env_fn = lambda: gym.make("Pendulum-v1", g=g)
        # train agent 
        apt_fixed_source.run_apt(env_fn, total_steps=total_steps, lr=lr, learning_starts=learning_starts, 
                    update_every=update_every, replay_buffer_size=replay_buffer_size, 
                    eval_steps_per_epoch=eval_steps_per_epoch, max_ep_len=max_ep_len,
                    batch_size=batch_size, hidden_size=hidden_size, num_grad_updates=num_grad_updates, 
                    num_test_episodes=num_test_episodes, test_name=test_name, 
                    source_path=source_path, beta=beta)

        finish = time.time()
        print(f'================================================')
        print(f'Total time: {(finish - start)/60} minutes')
        print(f'================================================')



def run_half_cheetah(task_type):
    """
    For half-cheetah, tasks are created as the following: 
    Dynamics change:
        joint stiffness is increased gradually from Task 0 to Task 4, 
        where Task 0 has the default gym values for joint stiffness 
    Reward change: 
        reward function is changed by encouraging the robot to wither move in forward 
        or rever direction, this is done by changing the values of forward_reward_weight
        from -2.0 to +3.0 
    Total tasks: 5 different dynamics + 5 different rewards - 1 common task = 9 tasks
    """
    xmls = [] 
    for k in range(5):
        xml = f'/home/ghost-083/Research/1_Transfer_RL/D3M/envs/halfcheetah/assets/half_cheetah_{k}.xml'
        xmls.append(xml)

    forward_rewards = [1.0, 2.0, 3.0, -1.0, -2.0]
    
    for k in range(len(xmls)):
        start = time.time()
        if task_type == "dynamics":
            xml, xml_id = xmls[k], k
            f_reward = forward_rewards[0]  # do not change the reward
        elif task_type == "reward":
            xml, xml_id = xmls[0], 0  # do not change the dyamics 
            f_reward = forward_rewards[k]
        else:
            raise Exception('task type not correct!')

        env_fn = lambda: gym.make('HalfCheetah-v3', xml_file=xml, 
                    forward_reward_weight=f_reward, 
                    ctrl_cost_weight=0.1, 
                    reset_noise_scale=0.1
                    )

        # vanilla sac 
        sac.sac(env_fn, total_steps=10, lr=1e-3, learning_starts=5, update_every=5, 
                replay_buffer_size=10000, steps_per_epoch=100, max_ep_len=200,
                batch_size = 64, hidden_size=200, num_grad_updates=50, num_test_episodes=5, 
                test_name=f'half_cheetah_dyn_{xml_id}_reward_{f_reward}_sac')

        finish = time.time()
        print(f'================================================')
        print(f'Total time: {(finish - start)/60} minutes')
        print(f'================================================')



def run_ant():
    """
    For ant, tasks are created as the following: 
    Dynamics change:
        joint stiffness is increased gradually from Task 0 to Task 4, 
        where Task 0 has the default gym values for joint stiffness 
    Reward change: 
        no reward change is available at this moment 
    Total tasks: 5 different dynamics = 5 tasks
    """
    xml_id = 1
    xml = f'/home/ghost-083/Research/1_Transfer_RL/D3M/envs/Ant/assets/ant_{xml_id}.xml'
    source_path = 'data/ant/ant_0_sac/policy.pth'

    # parameters:
    # ---------------------------
    test_name = f'ant_{xml_id}_apt_test'
    source_path = source_path                                      
    beta = 100.0
    # ----------------------------              
    total_steps = 50000                                     # fixed for all tasks
    lr = 1e-3                                               # fixed for all tasks
    learning_starts = 10000                                  # fixed for all tasks
    update_every = 50                                       # fixed for all tasks
    replay_buffer_size = 100000                              # fixed for all tasks
    eval_steps_per_epoch = 1000                              # fixed for all tasks
    max_ep_len = 1000                                        # fixed for all tasks
    batch_size = 64                                         # fixed for all tasks
    hidden_size = 200                                       # fixed for all tasks
    num_grad_updates = 50                                   # fixed for all tasks
    num_test_episodes = 10                                  # fixed for all tasks
    beta = 100.0                                            # fixed for all tasks


    start = time.time()
    env_fn = lambda: gym.make('Ant-v3', xml_file=xml)

    # train agent 
    apt.run_apt(env_fn, total_steps=total_steps, lr=lr, learning_starts=learning_starts, 
                update_every=update_every, replay_buffer_size=replay_buffer_size, 
                eval_steps_per_epoch=eval_steps_per_epoch, max_ep_len=max_ep_len,
                batch_size=batch_size, hidden_size=hidden_size, num_grad_updates=num_grad_updates, 
                num_test_episodes=num_test_episodes, test_name=test_name, 
                source_path=source_path, beta=beta)

    # vanilla sac 
    # sac.sac(env_fn, total_steps=total_steps, lr=lr, learning_starts=learning_starts, update_every=update_every, 
    #         replay_buffer_size=replay_buffer_size, eval_steps_per_epoch=eval_steps_per_epoch, max_ep_len=max_ep_len,
    #         batch_size=batch_size, hidden_size=hidden_size, num_grad_updates=num_grad_updates, num_test_episodes=num_test_episodes, 
    #         test_name=test_name)


    finish = time.time()
    print(f'================================================')
    print(f'Total time: {(finish - start)/60} minutes')
    print(f'================================================')




parser = argparse.ArgumentParser(description='Train SAC agent')
parser.add_argument('-env', '--env_name', type=str, required=True, help="environment name")
parser.add_argument('-t', '--task_type', type=str, required=False, help="task type")
args = parser.parse_args()


if __name__ == "__main__":
    if args.env_name == "pendulum":
        run_pendulum()
    elif args.env_name == "half_cheetah":
        run_half_cheetah(args.task_type)
    elif args.env_name == "ant":
        run_ant()    
    else:
        raise Exception('Not implemented yet!')