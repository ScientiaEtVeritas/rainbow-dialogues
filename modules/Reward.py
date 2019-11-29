from modules.bleu import compute_bleu
import rouge
import torch

compute_rouge = rouge.Rouge(metrics=['rouge-w'],
                       limit_length=True,
                       length_limit=100,
                       length_limit_type='words',
                       apply_avg=True,
                       alpha=0.5, # Default F1_score
                       weight_factor=1.2,
                       stemming=False)

class Reward(object):
    def __init__(self, config):
        self.config = config
        
    def __call__(self, src, output, tgt):        
        output = [op[0] for op in output]
        rewards = torch.zeros((self.config.BATCH_SIZE, len(self.config.rewards)))
        
        for ri, reward in enumerate(self.config.rewards):
            if reward == "BLEU":       
                for dj in range(self.config.BATCH_SIZE):
                    otp = output[dj].tolist()
                    otp = otp[:-1] if otp[-1] == self.config.tgt_eos else otp
                    tgt_ = tgt[dj][1:].squeeze().tolist()
                    tgt_ = tgt_[:-1] if tgt_[-1] == self.config.tgt_eos else tgt_
                    bleu_score = compute_bleu([[tgt_]], [otp], max_order=4, smooth = True)[0]
                    rewards[dj][ri] = bleu_score
            elif reward == "ROUGE":
                for dj in range(self.config.BATCH_SIZE):
                    otp = output[dj].tolist()
                    otp = otp[:-1] if otp[-1] == self.config.tgt_eos else otp
                    otp = " ".join(str(x) for x in otp) 
                    tgt_ = tgt[dj][1:].squeeze().tolist()
                    tgt_ = tgt_[:-1] if tgt_[-1] == self.config.tgt_eos else tgt_
                    tgt_ = " ".join(str(x) for x in tgt_)
                    rouge_score = compute_rouge.get_scores([otp], [[tgt_]])['rouge-w']['f']
                    rewards[dj][ri] = rouge_score
        
        # Weighting rewards
        rewards = (rewards * torch.Tensor(self.config.rewards_weights)).sum(dim=1)        
        return rewards