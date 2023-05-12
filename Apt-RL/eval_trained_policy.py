import gym
from utils import eval_utils


if __name__ == "__main__":
    # trained model path
    # ---------------- half-cheetah ---------------
    # load_path = 'data/halfcheetah/half_cheetah_dyn_0_reward_1.0_sac/policy.pth'
    # xml_id = 4
    # xml = f'/home/ghost-083/Research/1_Transfer_RL/D3M/envs/halfcheetah/assets/half_cheetah_{xml_id}.xml'
    # env = gym.make('HalfCheetah-v3', xml_file=xml, 
    #                 forward_reward_weight=1.0, 
    #                 ctrl_cost_weight=0.1, 
    #                 reset_noise_scale=1.0
    #                 )

    # ----------- pendulum ------------------- 
    load_path = 'data/pendulum/pendulum_source_optimal/policy.pth'
    # load_path = '/home/ghost-083/Research/1_Transfer_RL/D3M/data/pendulum/pendulum_apt_fixed_g_16.0/policy.pth'
    env =  gym.make("Pendulum-v1", g=16.0)   # pendulum


    # # ---------------- ant ---------------
    # xml_id = 4
    # # load_path = f'data/ant/ant_{xml_id}_sac/policy.pth'
    # load_path = f'data/ant/ant_0_sac/policy.pth'
    # xml = f'/home/ghost-083/Research/1_Transfer_RL/D3M/envs/Ant/assets/ant_{xml_id}.xml'
    # env = gym.make('Ant-v3', xml_file=xml)

    # run evaluation
    # for visualization run the following command in the terminal first: 
    # unset LD_PRELOAD
    eval_utils.run_trained_policy(env=env, num_runs=1, 
                                load_path=load_path, max_ep_len=200,
                                render=True, f_name=f'pendulum_1', agent='random')
