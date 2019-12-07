import torch

class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = "data/cornell_raw_min_680_tok"

        self.label = ""
        
        # data hyperparameters
        self.min_token_occurrences = 20
        
        # seq2seq hyperparameters
        self.rnn_type = "GRU" # options: LSTM, GRU
        self.emb_size = 100
        self.rnn_size = 500

        self.LEARNING_METHOD = 1
        # 0: Supervised Learning
        # 1: Q-learning
        # 2: Supervised Learning and Q-learning
        # 3: Multitask Learning

        # Supervised Pretraining
        self.SAVE_PRETRAIN_LOSS_EVERY = 20
        self.SAVE_PRETRAIN_REWARD_EVERY = 5
        self.SAVE_PRETRAIN_SAMPLE_EVERY = 100
        
        # DQN
        self.target_update_freq = 10000 # typically 10^4 to 10^6
        self.DUELING = True
        self.criterion = "huber" # options: huber, mse
        self.N_STEPS = 4
        self.DISTRIBUTIONAL = True
        self.QUANTILES = 51 # if DISTRIBUTIONAL is true

        # Reward / Loss
        self.value_penalty = True
        self.normalization_method = "sentence" # options: batch, sentence
        
        # Replay Memory
        self.replay_type = "per" # options: er, per
        self.PRIORITY_TYPE = "sum" # options: mean, sum | only applicable if replay_type = "per"
        self.EXP_REPLAY_SIZE = 1000000 # Memory size 1M transitions
        self.PRETRAIN_ITER = 0
        self.SAMPLE_EVERY = 8
        
        # PER
        self.PER_ALPHA = 0.6 # The exponent α determines how much prioritization is used, with α = 0 corresponding to the uniform case.
        self.PER_BETA_START = 0.4 # We can correct this bias by using importance-sampling (IS) weights that fully compensates for the non-uniform probabilities P(i) if β = 1
        self.BETA_MAX_ITER = 1000000
        
        # misc agent and learning variables
        self.GAMMA = 0.99 # discount factor
        self.LR    = 1e-3
        self.BATCH_SIZE = 32
        self.optimizer = 'ranger' # options: adam, ranger

        # data logging parameters
        self.SAVE_SAMPLE_EVERY = 200
        self.SAVE_SIGMA_EVERY = 500
        self.SAVE_LOSS_EVERY = 100
        self.SAVE_TD_EVERY = 100
        self.SAVE_PER_WEIGHTS_EVERY = 100
        self.SAVE_GRAD_FLOW_EVERY = 500

config = Config()