import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
from utils.utils import soft_update, hard_update, get_action_dim
from model import QNetwork, SeRO_QNetwork, StochasticPolicy, DeterministicPolicy, SeROPolicy

class SeRO(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.hidden_size = args.hidden_size
            
        self.device = torch.device("cuda:{}".format(args.cuda_device) if args.cuda else "cpu")
        num_actions = get_action_dim(action_space)

        if self.policy_type == "ddpg":
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.critic = QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type, doubleQ=False).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)
            self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type, doubleQ=False).to(self.device)
            hard_update(self.critic_target, self.critic)
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_original = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)
            self.policy_target = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            hard_update(self.policy_target, self.policy)
        
        elif self.policy_type == "td3":
            self.policy_freq = args.policy_freq
            self.alpha = 0
            self.automatic_entropy_tuning = False    
            self.critic = QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)
            self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type).to(self.device)
            hard_update(self.critic_target, self.critic)
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_original = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)
            self.policy_target = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            hard_update(self.policy_target, self.policy)
        
        elif 'sac' in self.policy_type:
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr_alpha)
            self.critic = QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)
            self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type).to(self.device)
            hard_update(self.critic_target, self.critic)        
            self.policy = StochasticPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_original = StochasticPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)
        
        elif "sero" in self.policy_type:
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr_alpha)        
            self.critic = SeRO_QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)
            self.critic_target = SeRO_QNetwork(num_inputs, num_actions, args.hidden_size, args.observation_type).to(self.device)
            self.aux_coeff = args.aux_coef
            self.env_coeff = args.env_coef
            self.consol_coef = args.consol_coef
            
            if not args.use_aux_reward:
                self.aux_coeff = 0.0
            
            hard_update(self.critic_target, self.critic)
            self.policy = SeROPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_original = SeROPolicy(num_inputs, action_space.shape[0], args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.copy()).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _, std, _ = self.policy(state)
        else:
            _, _, action, std, _ = self.policy(state)
        return action.clone().detach().cpu().numpy()[0], std.clone().detach().cpu().numpy()[0]

    def cal_uncertainty(self, state, original=False):
        if self.policy_type == 'sero':
            state = torch.FloatTensor(state.copy()).to(self.device).unsqueeze(0)
            if original:
                deg_uncertainty = self.policy_original.uncertainty(state)
            else:
                deg_uncertainty = self.policy.uncertainty(state)
            return deg_uncertainty.clone().detach().cpu().numpy()[0]
        else:
            return 0

    def update_parameters(self, memory, batch_size, updates):
        state_batch, _, action_batch, reward_batch, next_state_batch, _, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch.copy()).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch.copy()).to(self.device)
        action_batch = torch.FloatTensor(action_batch.copy()).to(self.device)
        env_reward_batch = torch.FloatTensor(reward_batch.copy()).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch.copy()).to(self.device).unsqueeze(1)
        with torch.autograd.set_detect_anomaly(True):
            if self.policy_type == "sero":
                env_reward_batch  
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _, _, _ = self.policy(next_state_batch)
                    q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_q_next_target = torch.min(q1_next_target, q2_next_target)
                    soft_value = min_q_next_target - self.alpha * next_state_log_pi
                    target_q_value = env_reward_batch + mask_batch * self.gamma * soft_value # soft Q update
                    
                q1, q2 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value) 

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                pi, log_pi, _, _, _ = self.policy(state_batch)
                q1_pi, q2_pi = self.critic(state_batch, pi)
                min_q = torch.min(q1_pi, q2_pi)

                policy_loss = (((self.alpha * log_pi) - min_q)).mean()
                
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()
                    
                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp()
                else:
                    alpha_loss = torch.tensor(0.).to(self.device)

                if updates % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)
                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item()
            
            elif self.policy_type == "sac":
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _, _, _ = self.policy(next_state_batch)
                    q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_q_next_target = torch.min(q1_next_target, q2_next_target)
                    soft_value = min_q_next_target - self.alpha * next_state_log_pi
                    target_q_value = env_reward_batch + mask_batch * self.gamma * soft_value # soft Q update
                q1, q2 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value) 

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                pi, log_pi, _, _, _ = self.policy(state_batch)
                q1_pi, q2_pi = self.critic(state_batch, pi)
                min_q = torch.min(q1_pi, q2_pi)
                
                policy_loss = ((self.alpha * log_pi) - min_q).mean()
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp()
                else:
                    alpha_loss = torch.tensor(0.).to(self.device)

                if updates % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)
            
                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item()
            
            elif self.policy_type == "td3":
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _, _, _, _ = self.policy_target(next_state_batch)
                    q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_q_next_target = torch.min(q1_next_target, q2_next_target)
                    target_q_value = env_reward_batch + mask_batch * self.gamma * min_q_next_target
                q1, q2 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value) 

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                alpha_loss = torch.tensor(0.).to(self.device)
                    
                _, log_pi, pi, _, _ = self.policy(state_batch)
                q1_pi, q2_pi = self.critic(state_batch, pi)
                min_q = torch.min(q1_pi, q2_pi)
                
                policy_loss = -min_q.mean()
                
                if updates % self.policy_freq == 0:
                    self.policy_optim.zero_grad()
                    policy_loss.backward()
                    self.policy_optim.step()
                    
                    soft_update(self.critic_target, self.critic, self.tau)
                    soft_update(self.policy_target, self.policy, self.tau)
                    self.policy.decay_eps()

                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item()
            
            elif self.policy_type == "ddpg":
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action, _, _, _ = self.policy_target(next_state_batch)
                    q1_next_target = self.critic_target(next_state_batch, next_state_action)
                    target_q_value = env_reward_batch + mask_batch * self.gamma * q1_next_target
                q1 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                alpha_loss = torch.tensor(0.).to(self.device)

                if updates % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)
                
                _, log_pi, pi, _, _ = self.policy(state_batch)
                q1_pi = self.critic(state_batch, pi)
                
                policy_loss =  -q1_pi.mean()
                
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()
                soft_update(self.policy_target, self.policy, self.tau)
                self.policy.decay_eps()

                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item()

            else:
                raise NotImplementedError("Policy type '{}' is not implemented.".format(self.policy_type))
    
    def reupdate_parameters(self, memory, batch_size, updates):
        state_batch, uncertainty_batch, action_batch, reward_batch, next_state_batch, next_uncertainty_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch.copy()).to(self.device)
        uncertainty_batch = torch.FloatTensor(uncertainty_batch.copy()).to(self.device)  
        next_state_batch = torch.FloatTensor(next_state_batch.copy()).to(self.device)
        next_uncertainty_batch = torch.FloatTensor(next_uncertainty_batch.copy()).to(self.device)
        action_batch = torch.FloatTensor(action_batch.copy()).to(self.device)
        env_reward_batch = torch.FloatTensor(reward_batch.copy()).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch.copy()).to(self.device).unsqueeze(1)
        with torch.autograd.set_detect_anomaly(True):
            if self.policy_type == "sero":
                aux_reward_batch = -torch.where(env_reward_batch == 0, next_uncertainty_batch, torch.zeros_like(next_uncertainty_batch))
                
                total_reward = self.env_coeff*env_reward_batch + self.aux_coeff*aux_reward_batch  
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _, _, _ = self.policy(next_state_batch)
                    q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_q_next_target = torch.min(q1_next_target, q2_next_target)
                    soft_value = min_q_next_target - self.alpha * next_state_log_pi
                    target_q_value = total_reward + mask_batch * self.gamma * soft_value # soft Q update
                    
                q1, q2 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value) 

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                pi, log_pi, mean, std, action_sample= self.policy(state_batch)
                _, _, _, _, original_action_sample = self.policy_original(state_batch)
                dist_entropy = -1*self.consol_coef*self.policy.evaluate(state_batch, original_action_sample)
                upc_loss = (1-uncertainty_batch)*dist_entropy

                q1_pi, q2_pi = self.critic(state_batch, pi)
                min_q = torch.min(q1_pi, q2_pi)

                policy_loss = (((self.alpha * log_pi) - min_q + upc_loss)).mean()
                
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()
                    
                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp()
                else:
                    alpha_loss = torch.tensor(0.).to(self.device)

                if updates % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)
                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item(), upc_loss.mean().item()
            
            elif self.policy_type == "sac":
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _, _, _ = self.policy(next_state_batch)
                    q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_q_next_target = torch.min(q1_next_target, q2_next_target)
                    soft_value = min_q_next_target - self.alpha * next_state_log_pi
                    target_q_value = env_reward_batch + mask_batch * self.gamma * soft_value # soft Q update
                q1, q2 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value) 

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                pi, log_pi, _, _, _ = self.policy(state_batch)
                _, _, _, _, original_action_sample = self.policy_original(state_batch)
                
                q1_pi, q2_pi = self.critic(state_batch, pi)
                min_q = torch.min(q1_pi, q2_pi)
                
                policy_loss = ((self.alpha * log_pi) - min_q).mean()
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp()
                else:
                    alpha_loss = torch.tensor(0.).to(self.device)

                if updates % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)
            
                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item(), 0
               
            elif self.policy_type == "td3":
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _, _, _, _ = self.policy_target(next_state_batch)
                    q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_q_next_target = torch.min(q1_next_target, q2_next_target)
                    target_q_value = env_reward_batch + mask_batch * self.gamma * min_q_next_target
                q1, q2 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value) 

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                alpha_loss = torch.tensor(0.).to(self.device)
                    
                _, log_pi, pi, _, _ = self.policy(state_batch)
                _, _, _, _, original_action_sample = self.policy_original(state_batch)
                
                q1_pi, q2_pi = self.critic(state_batch, pi)
                min_q = torch.min(q1_pi, q2_pi)
                
                policy_loss = (-min_q).mean()
                
                if updates % self.policy_freq == 0:
                    self.policy_optim.zero_grad()
                    policy_loss.backward()
                    self.policy_optim.step()
                    
                    soft_update(self.critic_target, self.critic, self.tau)
                    soft_update(self.policy_target, self.policy, self.tau)
                    self.policy.decay_eps()

                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item(), 0
            
            elif self.policy_type == "ddpg":
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action, _, _, _ = self.policy_target(next_state_batch)
                    q1_next_target = self.critic_target(next_state_batch, next_state_action)
                    target_q_value = env_reward_batch + mask_batch * self.gamma * q1_next_target
                q1 = self.critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q1, target_q_value)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                alpha_loss = torch.tensor(0.).to(self.device)

                if updates % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)
                
                _, log_pi, pi, _, _ = self.policy(state_batch)
                _, _, _, _, original_action_sample = self.policy_original(state_batch)
        
                q1_pi = self.critic(state_batch, pi)
                
                policy_loss =  (-q1_pi).mean()
                
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()
                soft_update(self.policy_target, self.policy, self.tau)
                self.policy.decay_eps()

                return critic_loss.item(), policy_loss.item(), alpha_loss.item(), policy_loss.item(), 0
            
            else:
                raise NotImplementedError("Policy type '{}' is not implemented.".format(self.policy_type))
    
    def save_model(self, env_name, policy, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists("trained_models/"):
            os.makedirs('trained_models/')
        if not os.path.exists('trained_models/{}'.format(env_name)):
            os.makedirs('trained_models/{}'.format(env_name))
        if not os.path.exists('trained_models/{}/{}'.format(env_name, policy)):
            os.makedirs('trained_models/{}/{}'.format(env_name, policy))
        
        if actor_path is None:
            actor_path = "trained_models/{}/{}/actor{}.pt".format(env_name, policy, suffix)
        if critic_path is None:
            critic_path = "trained_models/{}/{}/critic{}.pt".format(env_name, policy, suffix)
        if hasattr(self.policy, 'do_variance'):
            mapping_path = 'trained_models/{}/{}/mapping{}.pt'.format(env_name, policy, suffix)
            torch.save(self.policy.do_variance, mapping_path)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, args, actor_path, critic_path, mapping_path=None, retrain=False, original_actor_path=None, original_mapping_path=None):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if retrain:
            if actor_path is not None:
                self.policy.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy_original.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy.train()
                self.policy_original.eval()
                if hasattr(self, 'policy_target'):
                    hard_update(self.policy_target, self.policy)

                if hasattr(self.policy, 'do_variance'):
                    self.policy.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                    self.policy_original.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                    self.policy.fix_uncertainty = True
                    self.policy_original.fix_uncertainty = True
                    self.policy.dropout.train()
                    self.policy_original.dropout.train()
                
        else:
            if actor_path is not None:
                self.policy.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                if original_actor_path is not None and os.path.exists(original_actor_path):
                    self.policy_original.load_state_dict(torch.load(original_actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                else:
                    self.policy_original.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy.eval()
                self.policy_original.eval()
                if hasattr(self.policy, 'do_variance'):
                    self.policy.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                    if original_mapping_path is not None:
                        self.policy_original.do_variance = torch.load(original_mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                    else:
                        self.policy_original.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                    self.policy.fix_uncertainty = True
                    self.policy_original.fix_uncertainty = True
                    self.policy.dropout.train()
                    self.policy_original.dropout.train()
                
            if critic_path is not None:
                self.critic.load_state_dict(torch.load(critic_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.critic.eval()