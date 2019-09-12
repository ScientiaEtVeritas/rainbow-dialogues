import torch

class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #data hyperparameters
        self.max_sequence_length = 10
        self.min_token_occurrences = 20
        
        #seq2seq hyperparameters
        self.emb_size = 100
        self.rnn_size = 500
        
        # DQN
        self.target_update_freq = 1000
        self.DUELING = True
        
        # Replay Memory
        self.EXP_REPLAY_SIZE = 2000000
        self.PRETRAIN_ITER = 5000
        
        # PER
        self.PER_ALPHA = 0.6 # The exponent α determines how much prioritization is used, with α = 0 corresponding to the uniform case.
        self.PER_BETA_START = 0.4 # We can correct this bias by using importance-sampling (IS) weights that fully compensates for the non-uniform probabilities P(i) if β = 1
        self.BETA_MAX_ITER = 100000
        
        #misc agent variables
        self.GAMMA = 0.99 # discount factor
        self.LR    = 1e-4
        self.BATCH_SIZE = 32

        #data logging parameters
        self.REPORT_SAMPLE_EVERY = 20
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000 # TODO:

config = Config()