import torch

class Config(object):
    def __init__(self):
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # data hyperparameters
        self.max_sequence_length = 10
        self.min_token_occurrences = 20
        
        # seq2seq hyperparameters
        self.rnn_type = "GRU" # options: LSTM, GRU
        self.emb_size = 100
        self.rnn_size = 500
        
        # DQN
        self.target_update_freq = 10000 # typically 10^4 to 10^6
        self.DUELING = True
        self.criterion = "huber" # options: huber, mse
        self.N_STEPS = 3
        self.DISTRIBUTIONAL = True
        self.QUANTILES = 51 # if DISTRIBUTIONAL is true
        
        # Replay Memory
        self.replay_type = "per" # options: er, per
        self.EXP_REPLAY_SIZE = 1000000 # Memory size 1M transitions
        self.PRETRAIN_ITER = 0
        self.SAMPLE_EVERY = 5
        
        # PER
        self.PER_ALPHA = 0.6 # The exponent α determines how much prioritization is used, with α = 0 corresponding to the uniform case.
        self.PER_BETA_START = 0.4 # We can correct this bias by using importance-sampling (IS) weights that fully compensates for the non-uniform probabilities P(i) if β = 1
        self.BETA_MAX_ITER = 1500000
        
        # misc agent and learning variables
        self.GAMMA = 0.99 # discount factor
        self.LR    = 1e-3
        self.BATCH_SIZE = 32
        self.optimizer = 'ranger' # options: adam, ranger

        # data logging parameters
        self.SAVE_SAMPLE_EVERY = 5
        self.SAVE_SIGMA_EVERY = 500
        self.SAVE_LOSS_EVERY = 1
        self.SAVE_TD_EVERY = 1
        self.SAVE_PER_WEIGHTS_EVERY = 1
        self.SAVE_GRAD_FLOW_EVERY = 250

config = Config()