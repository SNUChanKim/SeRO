import math
import torch
import numpy as np
import random
import pickle
import os
from gym.spaces.box import Box


def load_agent_trained(agent, args):
    suffix="{}".format("_" + str(args.seed))
    if args.env_name == 'AntOOD-v2':
        args.critic_path = "./trained_models/AntNormal-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/AntNormal-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/AntNormal-v2/{}/mapping{}.pt".format(args.policy, suffix)
    elif args.env_name == 'Walker2dOOD-v2':
        args.critic_path = "./trained_models/Walker2dNormal-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/Walker2dNormal-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/Walker2dNormal-v2/{}/mapping{}.pt".format(args.policy, suffix)
    elif args.env_name == 'HalfCheetahOOD-v2':
        args.critic_path = "./trained_models/HalfCheetahNormal-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/HalfCheetahNormal-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/HalfCheetahNormal-v2/{}/mapping{}.pt".format(args.policy, suffix)
    elif args.env_name == 'HumanoidOOD-v2':
        args.critic_path = "./trained_models/HumanoidNormal-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/HumanoidNormal-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/HumanoidNormal-v2/{}/mapping{}.pt".format(args.policy, suffix)
    elif args.env_name == 'HopperOOD-v2':
        args.critic_path = "./trained_models/HopperNormal-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/HopperNormal-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/HopperNormal-v2/{}/mapping{}.pt".format(args.policy, suffix)
    else:
        args.critic_path = "./trained_models/{}/{}/critic{}.pt".format(args.env_name, args.policy, suffix)
        args.actor_path = "./trained_models/{}/{}/actor{}.pt".format(args.env_name, args.policy, suffix)
        mapping_path = "./trained_models/{}/{}/mapping{}.pt".format(args.env_name, args.policy, suffix)
            
    agent.load_model(args, args.actor_path, args.critic_path, mapping_path, args.load_model)

def load_agent_retrained(agent, args):
    suffix="{}".format("_" + str(args.seed))
    if 'sero' in args.policy:
        policy_original = 'sero'
        original_suffix = "{}".format("_" + str(args.seed))
    else:
        policy_original = args.policy
        original_suffix = suffix
    
    if args.env_name == 'AntNormal-v2' or args.env_name == 'AntOOD-v2':
        args.critic_path = "./trained_models/AntOOD-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/AntOOD-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/AntOOD-v2/{}/mapping{}.pt".format(args.policy, suffix)
        original_actor_path = "./trained_models/AntNormal-v2/{}/actor{}.pt".format(policy_original, original_suffix)
        original_mapping_path = "./trained_models/AntNormal-v2/{}/mapping{}.pt".format(policy_original, original_suffix)

    elif args.env_name == 'Walker2dNormal-v2' or args.env_name == 'Walker2dOOD-v2':
        args.critic_path = "./trained_models/Walker2dOOD-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/Walker2dOOD-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/Walker2dOOD-v2/{}/mapping{}.pt".format(args.policy, suffix)
        original_actor_path = "./trained_models/Walker2dNormal-v2/{}/actor{}.pt".format(policy_original, original_suffix)
        original_mapping_path = "./trained_models/Walker2dNormal-v2/{}/mapping{}.pt".format(policy_original, original_suffix)
    
    elif args.env_name == 'HalfCheetahNormal-v2' or args.env_name == 'HalfCheetahOOD-v2':
        args.critic_path = "./trained_models/HalfCheetahOOD-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/HalfCheetahOOD-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/HalfCheetahOOD-v2/{}/mapping{}.pt".format(args.policy, suffix)
        original_actor_path = "./trained_models/HalfCheetahNormal-v2/{}/actor{}.pt".format(policy_original, original_suffix)
        original_mapping_path = "./trained_models/HalfCheetahNormal-v2/{}/mapping{}.pt".format(policy_original, original_suffix)

    elif args.env_name == 'HopperNormal-v2' or args.env_name == 'HopperOOD-v2':
        args.critic_path = "./trained_models/HopperOOD-v2/{}/critic{}.pt".format(args.policy, suffix)
        args.actor_path = "./trained_models/HopperOOD-v2/{}/actor{}.pt".format(args.policy, suffix)
        mapping_path = "./trained_models/HopperOOD-v2/{}/mapping{}.pt".format(args.policy, suffix)
        original_actor_path = "./trained_models/HopperNormal-v2/{}/actor{}.pt".format(policy_original, original_suffix)
        original_mapping_path = "./trained_models/HopperNormal-v2/{}/mapping{}.pt".format(policy_original, original_suffix)

    else:
        raise NotImplementedError("Environment is not implemented for evaluation.")

    agent.load_model(args, args.actor_path, args.critic_path, mapping_path, args.load_model, original_actor_path=original_actor_path, original_mapping_path=original_mapping_path)

def load_buffer(args):
    suffix="{}".format("_" + str(args.seed))
    buffer_path = "./trained_models/{}/{}/buffer_{}.obj".format(args.env_name, args.policy, suffix)
    filehandler = open(buffer_path, 'rb')
    buffer = pickle.load(filehandler)
    print("Load Buffer from {}".format(buffer_path))
    return buffer

def save_buffer(args, buffer):
    if not os.path.exists("trained_models/"):
        os.makedirs('trained_models/')
    if not os.path.exists('trained_models/{}'.format(args.env_name)):
        os.makedirs('trained_models/{}'.format(args.env_name))
    if not os.path.exists('trained_models/{}/{}'.format(args.env_name, args.policy)):
        os.makedirs('trained_models/{}/{}'.format(args.env_name, args.policy))
    
    suffix="{}".format("_" + str(args.seed))
    buffer_path = "./trained_models/{}/{}/buffer_{}.obj".format(args.env_name, args.policy, suffix)
    filehandler = open(buffer_path, 'wb')
    pickle.dump(buffer, filehandler)
    
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2*math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5*z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def set_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_obs_dim(observation_space):
    if len(observation_space.shape) == 0:
        obs_shape = 1
    elif len(observation_space.shape) >= 2:
        obs_shape = 1
        for i in range(len(observation_space.shape)):
            obs_shape *= observation_space.shape[i]
    else:
        obs_shape = observation_space.shape[0]
    return obs_shape

def get_action_dim(action_space):
    if len(action_space.shape) == 0:
        action_shape = 1
    else:
        action_shape = action_space.shape[0]
    
    return action_shape    
 
def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def euler_to_quaternion(roll, pitch, yaw):
    
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return qx, qy, qz, qw