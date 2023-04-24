import os
import datetime
import gym
import envs
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

import time
from config import get_args
from sero import SeRO
from utils.replay_memory import ReplayMemory
from utils.utils import set_seed, load_agent_trained, save_buffer

import pyvirtualdisplay

def run_eval(args, agent, writer, updates):
    print("RUN EVALUATION")
    avg_reward = 0.
    eval_env = gym.make(args.env_name)
    eval_env.seed(args.seed + 54321)
    
    for _ in range(args.eval_episodes):
        state = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, std = agent.select_action(state, evaluate=True)
            if args.render:
                eval_env.render()
            next_state, reward, done, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= args.eval_episodes

    writer.add_scalar('avg_reward/evaluation', avg_reward, updates)
    eval_env.close()
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(args.eval_episodes, round(avg_reward, 2)))
    print("----------------------------------------")

def off_policy():
    args = get_args()
    torch.set_num_threads(args.num_threads)
    if args.server:
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
    env = gym.make(args.env_name)
    set_seed(env, args.seed)
    if args.observation_type == 'vector':
        agent = SeRO(env.observation_space.shape[0], env.action_space, args)
    elif args.observation_type == 'box':
        agent = SeRO(env.observation_space.shape, env.action_space, args)
    else:
        raise NotImplementedError

    if args.load_model:
        load_agent_trained(agent, args)
        
    memory = ReplayMemory(args.buffer_size, args.seed)
    
    if not os.path.exists('log/'):
        os.makedirs('log/')
    if not os.path.exists('log/{}'.format(args.env_name)):
        os.makedirs('log/{}'.format(args.env_name))
    if args.policy == 'sero':
        if args.aux_coef == 0:
            args.policy = 'sero_upc'
        elif args.consol_coef == 0:
            args.policy = 'sero_aux' 
    args.policy = args.policy + '_env'
    writer = SummaryWriter('log/{}/{}_{}_{}{}_seed_{}'.format(args.env_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'SeRO', 
                                                             args.policy, "_" + args.uncertainty_type if args.policy == 'sero' else "", args.seed))
        
    total_num_steps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        epsiode_aux_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        
        while not done:
            
            if args.start_steps > total_num_steps:
                action = env.action_space.sample()
                deg_uncertainty = agent.cal_uncertainty(state, original=True)
            else:
                deg_uncertainty = agent.cal_uncertainty(state, original=True)
                action, std = agent.select_action(state)

                writer.add_scalar('std', std.mean(), total_num_steps)
                
                if len(memory) > args.batch_size:
                    for i in range(args.updates_per_step):
                        train_start = time.time()
                        critic_loss, policy_loss, entropy_loss, total_loss, regularization_loss = agent.reupdate_parameters(memory, args.batch_size, updates)
                        train_end = time.time()

                        writer.add_scalar('loss/critic', critic_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy', entropy_loss, updates)
                        writer.add_scalar('loss/regularization', regularization_loss, updates)
                        writer.add_scalar('loss/total_policy_loss', total_loss, updates)
                        writer.add_scalar('one_step_update_time', train_end - train_start, updates)
                        if args.policy not in ['ddpg', 'td3']:
                            if args.automatic_entropy_tuning:
                                writer.add_scalar('alpha', agent.alpha.item(), updates)

                        if updates % args.eval_interval == 0:
                            run_eval(args, agent, writer, updates)
                        
                        updates += 1
                        
            if args.render:
                env.render()
            next_state, reward, done, info = env.step(action)
            reward = info['original_reward']
            next_deg_uncertainty = agent.cal_uncertainty(next_state, original=True)
            
            if 'sero' in args.policy:
                writer.add_scalar('deg_uncertainty', next_deg_uncertainty.mean(), total_num_steps)
                if args.uncertainty_type == 'scalar':
                    aux_reward = -next_deg_uncertainty if reward == 0 else 0
                else:
                    raise NotImplementedError
            else:
                aux_reward = 0

            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward
            epsiode_aux_reward += args.aux_coef*aux_reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, deg_uncertainty, action, reward, next_state, next_deg_uncertainty, mask)
            state = next_state
        if total_num_steps > args.num_steps:
            break

        writer.add_scalar('train/episode_return', episode_reward, i_episode)
        writer.add_scalar('train/episode_aux_return', epsiode_aux_reward, i_episode)
        writer.add_scalar('train/episode_steps', episode_steps, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_num_steps, episode_steps, round(episode_reward, 2)))
        
        if i_episode % args.save_interval == 0:
            agent.save_model(env_name=args.env_name, policy=args.policy, suffix="{}".format("_" + str(args.seed)))
            if args.save_buffer:
                save_buffer(args, memory)
               
    agent.save_model(env_name=args.env_name, policy=args.policy, suffix="{}".format("_" + str(args.seed)))
    if args.save_buffer:
        save_buffer(args, memory)
    env.close()

if __name__ == "__main__":
    off_policy()