import torch
import torch.nn as nn
import onmt
import random

from Config import config
from modules.DQN import DQN
from modules.Model import Model
from modules.MSELoss import MSELoss
from modules.Reward import Reward
from modules.RLModelSaver import RLModelSaver
from modules.QLearning import QLearning

vocab_fields = torch.load(config.dataset + ".vocab.pt")

src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

config.vocab_fields = vocab_fields
config.src_vocab = src_vocab
config.tgt_vocab = tgt_vocab
config.src_vocab_size = len(src_vocab)
config.tgt_vocab_size = len(tgt_vocab)
config.src_padding = src_padding
config.tgt_padding = tgt_padding
config.src_unk = src_vocab.stoi[src_text_field.unk_token]
config.tgt_unk = tgt_vocab.stoi[tgt_text_field.unk_token]
config.tgt_bos = tgt_vocab.stoi[tgt_text_field.init_token]
config.tgt_eos = tgt_vocab.stoi[tgt_text_field.eos_token]

train_data_file = config.dataset + ".train.0.pt"
train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[train_data_file],
                                                     fields=vocab_fields,
                                                     batch_size=1,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=config.device,
                                                     is_train=True,
                                                     repeat=False,
                                                     pool_factor=8192)

data = list(train_iter)
filtered_data = []
for x in data:
    if not ((x.src[0].squeeze() == config.src_unk).any() or (x.tgt.squeeze() == config.tgt_unk).any()):
        filtered_data.append(x)          
config.PRELOADING_SIZE = len(filtered_data)

model = Model(config, DQN)

if config.criterion == "mse":
    criterion = nn.MSELoss(reduction="none")
elif config.criterion == "huber":
    criterion = nn.SmoothL1Loss(reduction="none")

loss = MSELoss(
    criterion,
    model.current_model.generator
)

config.rewards = ['BLEU']
config.rewards_weights = [1]    

reward = Reward(config)

if config.optimizer == "adam":
    torch_optimizer = torch.optim.Adam(model.current_model.parameters(), lr=config.LR)
elif config.optimizer == "ranger":
    from lib.Ranger import Ranger
    torch_optimizer = Ranger(model.current_model.parameters(), lr=config.LR)
optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=config.LR, max_grad_norm=2)

model_saver = RLModelSaver("checkpoints/checkpoint", model, config, vocab_fields, optim)

random.Random(42).shuffle(filtered_data)
for example in filtered_data[150:]:
    model.replay_memory.preload(example.src[0].squeeze(1), example.tgt.squeeze(1), 1)
    model.sample_buffer.preload(example.src[0].squeeze(1), example.tgt.squeeze(1), None)

trainer = QLearning(config,
                    model,
                    reward=reward,
                    train_loss=loss,
                    valid_loss=loss,
                    optim=optim,
                    model_saver = model_saver)

if config.SUPERVISED_PRETRAINING:
    trainer.train(train_steps=150000, save_checkpoint_steps=25000)
trainer.train(train_steps=2000000, save_checkpoint_steps=75000)