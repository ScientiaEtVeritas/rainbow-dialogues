from modules.ReplayBuffer import ReplayBuffer
from modules.ReplayBuffer import PrioritizedReplayBuffer

import torch

class Model(object):
    def __init__(self, config, network, replay_type = 'per'):
        self.config = config
        self.network = network
        self.device = 'cpu'
        self.replay_type = replay_type
        
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
            self.replay_memory = ReplayBuffer(1000)
        elif self.replay_type == "per":
            self.replay_memory = PrioritizedReplayBuffer(1000, alpha = self.config.PER_ALPHA)
            
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
        current_net_next_q_outputs = current_net_q_outputs[1:]
        target_net_next_q_outputs = target_net_q_outputs[1:]
        next_q_values = target_net_next_q_outputs.gather(2, torch.max(current_net_next_q_outputs, 2)[1].unsqueeze(2)) # decorrelate select and max
        return next_q_values