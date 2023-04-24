import torch
import torch.nn as nn
from torch.distributions import Normal
from utils.random_process import OUProcess


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, doubleQ=True):
        super(QNetwork, self).__init__()
        self.doubleQ = doubleQ
        if observation_type == 'vector':
            self.Q1 = nn.Sequential(
                nn.Linear(num_inputs + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            if self.doubleQ:
                self.Q2 = nn.Sequential(
                    nn.Linear(num_inputs + num_actions, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
        elif observation_type == 'box':
            fake_in = torch.zeros(1, num_inputs[2], num_inputs[0], num_inputs[1])

            self.encoder = nn.Sequential(
                nn.Conv2d(num_inputs[2], 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU()
            )
            
            fake_out = self.encoder(fake_in)

            self.Q1 = nn.Sequential(
                nn.Linear(fake_out.shape[1]*fake_out.shape[2]*fake_out.shape[3] + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            if self.doubleQ:
                self.Q2 = nn.Sequential(
                    nn.Linear(fake_out.shape[1]*fake_out.shape[2]*fake_out.shape[3] + num_actions, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
        
        else:
            raise NotImplementedError
        
        self.obs_type = observation_type
            

    def forward(self, state, action):
        if self.obs_type == 'vector':
            state_action_pair = torch.cat([state, action], dim=-1)
            if self.doubleQ:
                return self.Q1(state_action_pair), self.Q2(state_action_pair)
            else:
                return self.Q1(state_action_pair)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = torch.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
            h_action_pair = torch.cat([h, action], dim=-1)
            if self.doubleQ:
                return self.Q1(h_action_pair), self.Q2(h_action_pair)
            else:
                return self.Q1(h_action_pair)
        else:
            raise NotImplementedError
            

class SeRO_QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type):
        super(SeRO_QNetwork, self).__init__()
        if observation_type == 'vector':
            self.Q1 = nn.Sequential(
                nn.Linear(num_inputs + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.Q2 = nn.Sequential(
                nn.Linear(num_inputs + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        elif observation_type == 'box':
            fake_in = torch.zeros(1, num_inputs[2], num_inputs[0], num_inputs[1])

            self.encoder = nn.Sequential(
                nn.Conv2d(num_inputs[2], 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU()
            )
            
            fake_out = self.encoder(fake_in)

            self.Q1 = nn.Sequential(
                nn.Linear(fake_out.shape[1]*fake_out.shape[2]*fake_out.shape[3] + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

            self.Q2 = nn.Sequential(
                nn.Linear(fake_out.shape[1]*fake_out.shape[2]*fake_out.shape[3] + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise NotImplementedError
        self.obs_type = observation_type
        
    def forward(self, state, action):
        if self.obs_type == 'vector':
            state_action_pair = torch.cat([state, action], dim=-1)
            return self.Q1(state_action_pair), self.Q2(state_action_pair)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = torch.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
            h_action_pair = torch.cat([h, action], dim=-1)
            return self.Q1(h_action_pair), self.Q2(h_action_pair)
        else:
            raise NotImplementedError

class StochasticPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, action_space=None, drop_p=0.1):
        super(StochasticPolicy, self).__init__()
        
        self.hidden_dim = hidden_dim
        if observation_type == 'vector':
            self.shared = nn.Sequential(
                nn.Linear(num_inputs, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
        elif observation_type == 'box':
            fake_in = torch.zeros(1, num_inputs[2], num_inputs[0], num_inputs[1])

            self.encoder = nn.Sequential(
                nn.Conv2d(num_inputs[2], 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Conv2d(16, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
            
            fake_out = self.encoder(fake_in)

            self.shared = nn.Sequential(
                nn.Linear(fake_out.shape[1]*fake_out.shape[2]*fake_out.shape[3], 2*hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
        else:
            raise NotImplementedError

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low)/2.
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low)/2.
            )

        self.obs_type = observation_type

    def forward(self, state):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = torch.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        log_std = torch.clamp(self.log_std(shared), -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_sample = normal.rsample()
        action_normalize = torch.tanh(action_sample)
        action = action_normalize*self.action_scale + self.action_bias
        log_prob = normal.log_prob(action_sample)
        log_prob -= torch.log(self.action_scale*(1 - action_normalize.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)*self.action_scale + self.action_bias
        return action, log_prob, mean, std, action_sample
    
    def evaluate(self, state, action_sample):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = torch.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        log_std = torch.clamp(self.log_std(shared), -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_normalize = torch.tanh(action_sample)
        log_prob = normal.log_prob(action_sample)
        log_prob -= torch.log(self.action_scale*(1 - action_normalize.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(StochasticPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, action_space=None, drop_p=0.1):
        super(DeterministicPolicy, self).__init__()
        self.epsilon = 1.0
        self.epsilon_decay = 2e-5
        self.ou_process = OUProcess(theta=0.3, mu=0., sigma=0.2, dt=5e-2, size=num_actions)
        if observation_type == 'vector':
            self.shared = nn.Sequential(
                nn.Linear(num_inputs, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
        elif observation_type == 'box':
            fake_in = torch.zeros(1, num_inputs[2], num_inputs[0], num_inputs[1])

            self.encoder = nn.Sequential(
                nn.Conv2d(num_inputs[2], 16, 4, stride=2, padding=1), #48*48*16
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Conv2d(16, 32, 4, stride=2, padding=1), # 24*24*32
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Conv2d(32, 64, 4, stride=2, padding=1), # 12*12*64
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Conv2d(64, 128, 4, stride=2, padding=1), # 6*6*128
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
            
            fake_out = self.encoder(fake_in)

            self.shared = nn.Sequential(
                nn.Linear(fake_out.shape[1]*fake_out.shape[2]*fake_out.shape[3], 2*hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
        else:
            raise NotImplementedError

        self.mean = nn.Linear(hidden_dim, num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low)/2.
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low)/2.
            )
        self.obs_type = observation_type
        self.action_high = torch.FloatTensor(action_space.high)
        self.action_low = torch.FloatTensor(action_space.low)

    def forward(self, state):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = torch.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        action_sample = mean
        mean = torch.tanh(mean)*self.action_scale + self.action_bias
        noise = torch.clamp(self.ou_process.sample(), -0.5, 0.5)*self.epsilon
        action = torch.max(torch.min(mean + noise, self.action_high), self.action_low)
        return action, torch.tensor(0.), mean, torch.zeros(action.shape[0]), action_sample
    
    def evaluate(self, state, action_sample):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = torch.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        log_prob = (mean - action_sample)**2
        return log_prob

    def decay_eps(self):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0)

    def to(self, device):
        self.ou_process = self.ou_process.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.action_high = self.action_high.to(device)
        self.action_low = self.action_low.to(device)
        return super(DeterministicPolicy, self).to(device)

class SeROPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, action_space=None, drop_p=0.1):
        super(SeROPolicy, self).__init__()
        self.num_sample = 30
        self.fix_uncertainty = False

        if observation_type == 'vector':
            self.linear1 = nn.Linear(num_inputs, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        elif observation_type == 'box':
            fake_in = torch.zeros(1, num_inputs[2], num_inputs[0], num_inputs[1])
            
            self.conv1 = nn.Conv2d(num_inputs[2], 16, 4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
            
            fake_out = self.conv4(self.conv3(self.conv2(self.conv1(fake_in))))

            self.linear1 = nn.Linear(fake_out.shape[1]*fake_out.shape[2]*fake_out.shape[3], 2*hidden_dim)
            self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)
        else:
            raise NotImplementedError

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)

        self.dropout = nn.Dropout(p = drop_p)
        self.relu = nn.ReLU()
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low)/2.
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low)/2.
            )
        self.obs_type = observation_type
        self.do_variance = -1e10*torch.ones([hidden_dim])

    def forward(self, state):
        if self.obs_type == 'vector':
            x = self.dropout(self.relu(self.linear1(state)))
            shared = self.dropout(self.relu(self.linear2(x)))
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            x1 = self.dropout(self.relu(self.conv1(state)))
            x2 = self.dropout(self.relu(self.conv2(x1)))
            x3 = self.dropout(self.relu(self.conv3(x2)))
            x4 = self.dropout(self.relu(self.conv4(x3)))
            x4_reshape = torch.reshape(x4, (x4.shape[0], x4.shape[1]*x4.shape[2]*x4.shape[3]))
            x5 = self.dropout(self.relu(self.linear1(x4_reshape)))
            shared = self.dropout(self.relu(self.linear2(x5)))
        else:
            raise NotImplementedError
        
        deg_uncertainty = self.uncertainty(state)
            
        mean = self.mean(shared)
        log_std = torch.clamp(self.log_std(shared), -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_sample = normal.rsample()
        action_normalize = torch.tanh(action_sample)
        action = action_normalize*self.action_scale + self.action_bias
        log_prob = normal.log_prob(action_sample)
        log_prob -= torch.log(self.action_scale*(1 - action_normalize.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)*self.action_scale + self.action_bias
        return action, log_prob, mean, std, action_sample

    def evaluate(self, state, action_sample):
        if self.obs_type == 'vector':
            x = self.dropout(self.relu(self.linear1(state)))
            shared = self.dropout(self.relu(self.linear2(x)))
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            x1 = self.dropout(self.relu(self.conv1(state)))
            x2 = self.dropout(self.relu(self.conv2(x1)))
            x3 = self.dropout(self.relu(self.conv3(x2)))
            x4 = self.dropout(self.relu(self.conv4(x3)))
            x4_reshape = torch.reshape(x4, (x4.shape[0], x4.shape[1]*x4.shape[2]*x4.shape[3]))
            x5 = self.dropout(self.relu(self.linear1(x4_reshape)))
            shared = self.dropout(self.relu(self.linear2(x5)))
        else:
            raise NotImplementedError
            
        mean = self.mean(shared)
        log_std = torch.clamp(self.log_std(shared), -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_normalize = torch.tanh(action_sample)
        log_prob = normal.log_prob(action_sample)
        log_prob -= torch.log(self.action_scale*(1 - action_normalize.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob
    
    def uncertainty(self, state):
        if self.obs_type == 'vector':
            with torch.no_grad():
                do_sum = 0
                do_square_sum = 0
                for i in range(self.num_sample):
                    x = self.dropout(self.relu(self.linear1(state)))
                    do_out = self.dropout(self.relu(self.linear2(x)))
                    do_sum += do_out
                    do_square_sum += do_out.square()
                
                u_var_vec = (do_square_sum - do_sum.square()/self.num_sample)/(self.num_sample - 1)
                if not self.fix_uncertainty:
                    self.do_variance = torch.max(self.do_variance, u_var_vec)
                u_var_vec = u_var_vec/(self.do_variance + 1e-10)

                u_var_weight = u_var_vec/(torch.sum(u_var_vec, dim=1, keepdim=True) + 1e-10)
                deg_uncertainty = torch.sum(u_var_vec*u_var_weight, dim=1, keepdim=True)
                deg_uncertainty = torch.clamp(deg_uncertainty, 0.0, 1.0)
                
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                do_sum = 0
                do_square_sum = 0
                for i in range(self.num_sample):
                    x1 = self.dropout(self.relu(self.conv1(state)))
                    x2 = self.dropout(self.relu(self.conv2(x1)))
                    x3 = self.dropout(self.relu(self.conv3(x2)))
                    x4 = self.dropout(self.relu(self.conv4(x3)))
                    x4_reshape = torch.reshape(x4, (x4.shape[0], x4.shape[1]*x4.shape[2]*x4.shape[3]))
                    x5 = self.dropout(self.relu(self.linear1(x4_reshape)))
                    do_out = self.dropout(self.relu(self.linear2(x5)))
                    do_sum += do_out
                    do_square_sum += do_out.square()
                
                u_var_vec = (do_square_sum - do_sum.square()/self.num_sample)/(self.num_sample - 1)
                
                if not self.fix_uncertainty:
                    self.do_variance = torch.max(self.do_variance, u_var_vec)
                u_var_vec = u_var_vec/(self.do_variance + 1e-10)

                u_var_weight = u_var_vec/(torch.sum(u_var_vec, dim=1, keepdim=True) + 1e-10)
                deg_uncertainty = torch.sum(u_var_vec*u_var_weight, dim=1, keepdim=True)
                deg_uncertainty = torch.clamp(deg_uncertainty, 0.0, 1.0)
                
        else:
            raise NotImplementedError

        return deg_uncertainty

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.do_variance = self.do_variance.to(device)
        return super(SeROPolicy, self).to(device)