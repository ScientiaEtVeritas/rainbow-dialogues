from modules.bleu import compute_bleu
import torch

class Reward(object):
    def __init__(self, config):
        self.config = config
        
    def __call__(self, src, output, tgt):        
        output = [op[0] for op in output]
        rewards = torch.zeros((self.config.BATCH_SIZE, len(self.config.rewards)))
        
        for ri, reward in enumerate(self.config.rewards):
            if reward == "BLEU":       
                for dj in range(self.config.BATCH_SIZE):
                    bleu_score = compute_bleu([[output[dj].tolist()]], [tgt[dj][1:].squeeze().tolist()], max_order=4, smooth = True)[0]
                    rewards[dj][ri] = bleu_score
        
        # Weighting rewards
        rewards = (rewards * torch.Tensor(self.config.rewards_weights)).sum(dim=1)        
        return rewards