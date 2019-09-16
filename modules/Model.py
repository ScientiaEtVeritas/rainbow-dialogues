from modules.ReplayBuffer import ReplayBuffer
from modules.ReplayBuffer import PrioritizedReplayBuffer

import torch
import os
import csv
import numpy as np

class Model(object):
    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.device = 'cpu'
        self.replay_type = config.replay_type
        
        self.declare_networks()
        self.declare_memory()
        
    def declare_networks(self):
        self.current_model = self.network(self.config)
        self.target_model  = self.network(self.config)
        self.update_target()
        self.current_model = self.current_model.to(self.device)
        self.target_model.to(self.device)
        
    def declare_memory(self):
        if self.replay_type == "er":
            self.replay_memory = ReplayBuffer(self.config.EXP_REPLAY_SIZE, preloading_size = self.config.PRELOADING_SIZE)
        elif self.replay_type == "per":
            self.replay_memory = PrioritizedReplayBuffer(self.config.EXP_REPLAY_SIZE, preloading_size = self.config.PRELOADING_SIZE, alpha = self.config.PER_ALPHA)
        self.sample_buffer = ReplayBuffer(self.config.PRELOADING_SIZE, self.config.PRELOADING_SIZE) # not used for actual experience replay, but for collecting new experiences
            
    def sample_from_memory(self, step):
        if self.replay_type == "er":
            return self.replay_memory.sample(self.config.BATCH_SIZE)
        elif self.replay_type == "per":
            print("BETA", self.beta_by_step(step))
            return self.replay_memory.sample(self.config.BATCH_SIZE, self.beta_by_step(step))

    def beta_by_step(self, step):
        return min(1.0, self.config.PER_BETA_START + step * (1.0 - self.config.PER_BETA_START) / self.config.BETA_MAX_ITER)
        #return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())
        
    def get_actions(self, output):
        actions = self.current_model.generator(output).max(dim=2)[1]
        return actions
    
    def get_current_q_values(self, output, target):
        target = target[1:]
        max_tgt_length = target.size(0)
        q_values = output[:max_tgt_length].gather(2,target)
        return q_values
    
    def get_next_q_values(self, current_net_q_outputs, target_net_q_outputs):
        if current_net_q_outputs.size(0) > 1:
            current_net_next_q_outputs = current_net_q_outputs[1:]
            target_net_next_q_outputs = target_net_q_outputs[1:]
            next_q_values = target_net_next_q_outputs.gather(2, torch.max(current_net_next_q_outputs, 2)[1].unsqueeze(2)) # decorrelate select and max
            return next_q_values
        else: # all sequences are final after one transition, so there is no "next q value" for any of them 
            return torch.zeros((0,self.config.BATCH_SIZE,1))
        
    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.current_model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_+= torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            
            if count > 0:
                with open(os.path.join('logs', 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_/count))   

    def save_td(self, td, tstep):
        with open(os.path.join('logs', 'td.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, td))

    def save_reward(self, reward, tstep):
        with open(os.path.join('logs', 'reward.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, reward))
