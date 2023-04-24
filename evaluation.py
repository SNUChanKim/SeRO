import os
import gym
import envs
import numpy as np
from config import get_args
from sero import SeRO
from utils.utils import set_seed, load_agent_trained, load_agent_retrained


import pyvirtualdisplay

def evaluation():
    args = get_args()    
    
    if args.server:
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    if not os.path.exists('log/'):
        os.makedirs('log/')
    if not os.path.exists('log/{}'.format(args.env_name)):
        os.makedirs('log/{}'.format(args.env_name))
    if not os.path.exists('log/{}/eval/'.format(args.env_name)):
        os.makedirs('log/{}/eval/'.format(args.env_name))

    env = gym.make(args.env_name)
    set_seed(env, args.eval_seed + 12345)
    
    if args.observation_type == 'vector':
        agent = SeRO(env.observation_space.shape[0], env.action_space, args)
    elif args.observation_type == 'box':
        agent = SeRO(env.observation_space.shape, env.action_space, args)
    else:
        raise NotImplementedError
    
    if args.eval_retrained:
        load_agent_retrained(agent, args)
    else:
        load_agent_trained(agent, args)
    
    total_episode = 0
    total_reward = 0.0
    total_reward_sq = 0.0
    for i_episode in range(args.num_evaluation):
        
        total_episode += 1
        episode_reward = 0
        episode_uncertainty = 0
        done = False
        state = env.reset()

        episode_step = 0
        epsiode_aux_reward = 0
        while not done:
            action, std = agent.select_action(state, evaluate=True)
            step = action
            next_state, reward, done, _ = env.step(step)

            next_deg_uncertainty = agent.cal_uncertainty(next_state, original=True)
            aux_reward = -next_deg_uncertainty if reward == 0 else 0
            epsiode_aux_reward += args.aux_coef*aux_reward
            episode_uncertainty += next_deg_uncertainty
            if args.render:
                env.render()
            
            episode_step += 1
            episode_reward += reward

            state = next_state
        total_reward += episode_reward
        total_reward_sq += episode_reward**2
        print("---------------------------------------------------------------")
        print("Episode: {}, Return: {}".format(i_episode, episode_reward))
        print("---------------------------------------------------------------")
    
    avg_reward = total_reward/args.num_evaluation
    avg_reward_sq = total_reward_sq/args.num_evaluation
    std_reward = np.sqrt(avg_reward_sq - avg_reward**2)
    env.close()
    
    print("==================================================================")
    print("Algo: {}".format(args.policy))
    print("Avg. Return: {}, Standard Deviation: {}".format(round(avg_reward, 2), round(std_reward, 2)))
    print("==================================================================")


if __name__ == "__main__":
    evaluation()