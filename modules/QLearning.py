import csv
import logging
import math
import os
import time
import traceback
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onmt.utils
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchtext
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class QLearning(object):
    """
    Class that controls the training process.
    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,
                 config,
                 model,
                 reward,
                 train_loss,
                 valid_loss,
                 optim,
                 shard_size=0,
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None): # dropout=[0.3], dropout_steps=[0]

        # RL attributes
        self.config = config
        self.model = model
        self.replay_memory = model.replay_memory
        self.current_model = model.current_model
        self.target_model = model.target_model
        self.reward = reward

        self.train_loss = train_loss
        self.valid_loss = valid_loss

        self.optim = optim

        # Meta attributes
        self.shard_size = shard_size
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.model_saver = model_saver
        self.model_dtype = model_dtype
        
        # NOTE: Dropout isn't usually used with RL, see https://ai.stackexchange.com/questions/8293/why-do-you-not-see-dropout-layers-on-reinforcement-learning-examples
        #self.dropout = dropout
        #self.dropout_steps = dropout_steps
        
        self.kernel = torch.cat((
            torch.zeros((self.config.N_STEPS - 1)),
            torch.logspace(start=0,end=config.N_STEPS-1,steps=config.N_STEPS, base=0.99))
        ).view(1, 1, -1).to(self.config.device)
        
        self.padding = int(math.ceil(self.kernel.size(2) - 1) / 2)

        # Set model in training mode.
        self.current_model.train()
        self.target_model.train()

    #def _maybe_update_dropout(self, step):
    #    for i in range(len(self.dropout_steps)):
    #        if step > 1 and step == self.dropout_steps[i] + 1:
    #            self.current_model.update_dropout(self.dropout[i])
    #            logger.info("Updated dropout to %f from step %d"
    #                        % (self.dropout[i], step))
                    
    def pretrain(self,
              train_steps,
              save_checkpoint_steps=5000):
        
        self.pretrain_generator = nn.Sequential(
            nn.Linear(self.config.rnn_size, self.config.tgt_vocab_size),
            nn.LogSoftmax(dim=-1)).to(self.config.device)
        
        self.pretrain_loss = onmt.utils.loss.NMTLossCompute(
            criterion=nn.NLLLoss(ignore_index=self.config.tgt_padding, reduction="sum"),
            generator=self.pretrain_generator)
        
        lr = 1
        self.pretrain_optim = onmt.utils.optimizers.Optimizer(
            torch.optim.SGD(self.current_model.parameters(), lr=lr), learning_rate=lr, max_grad_norm=2)
        
        logging.info('Start training loop')

        for i in range(train_steps):
            step = self.pretrain_optim.training_step
            logger.info("Step " + str(step))
            batch = self.model.sample_buffer.sample(self.config.BATCH_SIZE)
            normalization = self.config.BATCH_SIZE
            self._pretrain_step(batch, normalization)


    def train(self,
              train_steps,
              save_checkpoint_steps=5000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.
        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.
        Returns:
            The gathered statistics.
        """
        logging.info('Start training loop')

        for i in range(train_steps):
            step = self.optim.training_step
            logger.info("Step " + str(step))
            #self._maybe_update_dropout(step) # UPDATE DROPOUT | NOTE: Dropout isn't usually used with RL
            
            batch = self.model.sample_from_memory(step)
            normalization = self.config.BATCH_SIZE

            #if self.gpu_verbose_level > 1:
            #    logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            #if self.gpu_verbose_level > 0:
            #    logger.info("GpuRank %d: reduce_counter: %d" % (self.gpu_rank, i + 1))
            
            if step > self.config.PRETRAIN_ITER and step % self.config.SAMPLE_EVERY == 0:
                self._sample()

            self._step(batch, normalization)

            if step % self.config.target_update_freq == 0:
                logger.info("Target Model Updated")
                self.model.update_target()

            if (self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step)

    def _process_batch(self, batch):

        batch = sorted(batch, key = lambda t: t[0].size(0), reverse = True)
        src_raw, tgt_raw, reward, *per = zip(*batch)

        src = pad_sequence(src_raw, padding_value=self.config.src_padding).to(self.config.device)
        tgt = pad_sequence(tgt_raw, padding_value=self.config.tgt_padding).to(self.config.device)
        src_lengths = torch.ShortTensor([s.size(0) for s in src_raw]).to(self.config.device)

        logger.debug("Batch Length: " + str(src_lengths))

        true_batch = torchtext.data.Batch()
        true_batch.src = src
        true_batch.tgt = tgt
        true_batch.lengths = src_lengths

        return true_batch, src, tgt, src_lengths, src_raw, tgt_raw, reward, per
    
    def _sample(self):
        logger.info("Sampling: Collecting new data")
        batch = self.model.sample_buffer.sample(self.config.BATCH_SIZE)
        true_batch, src, tgt, src_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)
        with torch.no_grad():
            self.current_model.update_noise()
            predictions = self.current_model.infer(src, src_lengths, self.config.BATCH_SIZE)
            rewards = self.reward(src_raw, predictions, tgt_raw) # predictions are without BOS, tgt is with
            self.model.save('reward', rewards.sum().item(), self.optim.training_step)
            for i, prediction in enumerate(predictions):
                prediction_with_bos = torch.cat((torch.LongTensor([self.config.tgt_bos]).to(self.config.device), prediction[0]))
                if prediction_with_bos.size(0) > 1:
                    prediction_with_bos = prediction_with_bos.unsqueeze(1)
                    if self.optim.training_step % self.config.SAVE_SAMPLE_EVERY == 0:
                        text = ' '.join([self.config.tgt_vocab.itos[token.item()] for token in prediction_with_bos]) + f' ({rewards[i]})'
                        self.model.save('sample', text, self.optim.training_step)
                    idx = self.replay_memory.push(src_raw[i], prediction_with_bos, rewards[i])
                    logger.debug(f"Using / Replacing Index {idx}")
                else:
                    logger.debug(f"Inference {i} failed: " + repr(prediction_with_bos))
                    
    def _pretrain_step(self, batch, normalization):
        true_batch, src, tgt, src_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)
        
        self.pretrain_optim.zero_grad()
        outputs, attns = self.current_model(src, tgt, src_lengths, bptt=False)
        
        try:
            loss, batch_stats = self.pretrain_loss(
                true_batch,
                outputs,
                attns,
                normalization=normalization)
                        
            logger.debug(loss)

            if loss is not None: # maybe already backwarded in train_loss
                self.pretrain_optim.backward(loss)
                self.grad_flow(self.current_model.named_parameters(), self.pretrain_optim.training_step)

        except Exception:
            traceback.print_exc()
            logger.info("At step %d, we removed a batch", self.pretrain_optim.training_step)
            
        self.pretrain_optim.step()

        if self.current_model.decoder.state is not None:
            self.current_model.decoder.detach_state()


    def _step(self, batch, normalization):
        true_batch, src, tgt, src_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)
                
        self.current_model.update_noise()
        with torch.no_grad():
            self.target_model.update_noise()
            
        # pass through encoder and decoder
        self.optim.zero_grad()
        current_net__outputs, current_net_attns = self.current_model(src, tgt, src_lengths, bptt=False)
        target_net__outputs, target_net_attns = self.target_model(src, tgt, src_lengths, bptt=False)

        # pass through generator
        current_net__q_outputs = self.current_model.generator(current_net__outputs)
        target_net__q_outputs = self.target_model.generator(target_net__outputs).detach() # detach from graph, don't backpropagate
            
        # calc q values
        q_values = self.model.get_current_q_values(current_net__q_outputs, tgt)    
        next_q_values = self.model.get_next_q_values(current_net__q_outputs, target_net__q_outputs)
                
        # construct reward tensor
        tgt_is_eos = (tgt == self.config.tgt_eos)
        cond = (tgt_is_eos.sum(dim=0) == 0).squeeze().byte()
        other = tgt_is_eos[-1].squeeze().float()
        terminal_reward = torch.where(cond, torch.ones((tgt_is_eos.size(1)), device=self.config.device), other)
        tgt_is_eos[-1] = terminal_reward.view(tgt_is_eos.size(1), 1)
        rewards = (tgt_is_eos.float() * torch.Tensor(reward).to(self.config.device).view(tgt_is_eos.size(1), 1))[1:]            
        
        if self.config.N_STEPS > 1:
            # one-sided exponential n-step decay of rewards via convolution
            rewards = F.conv1d(rewards.permute(1, 2, 0), self.kernel, padding=self.padding).permute(2, 0, 1)
                    
        # mask bc padding
        mask = torch.ones_like(tgt, device=self.config.device)
        for eos_token in (tgt == self.config.tgt_eos).nonzero():
            mask[eos_token[0]+1:,eos_token[1]] = 0
        mask = mask[1:].float()
        
        if self.config.DISTRIBUTIONAL:
            rewards = rewards.unsqueeze(3)
            mask = mask.unsqueeze(3)
            zero_pad = torch.zeros((self.config.N_STEPS,self.config.BATCH_SIZE,1,self.config.QUANTILES), device=self.config.device)
        else:
            zero_pad = torch.zeros((self.config.N_STEPS,self.config.BATCH_SIZE,1), device=self.config.device)

        masked_q_values = q_values * mask
        masked_next_q_values = torch.cat([next_q_values * mask[1:], zero_pad])[self.config.N_STEPS-1:] # add zeros for final state(s)

        expected_q_values = rewards + self.config.GAMMA ** self.config.N_STEPS * masked_next_q_values
        
        density_weights = None
        if self.config.DISTRIBUTIONAL:
            expected_q_values = expected_q_values.permute(0,3,1,2)
            masked_q_values =  masked_q_values.permute(0,2,1,3)
            diff = expected_q_values - masked_q_values
            density_weights = torch.abs(self.model.cumulative_density.view(1, 1, -1) - (diff < 0).to(torch.float))
            (self.train_loss.criterion(masked_q_values, expected_q_values) * density_weights).transpose(1,2)
        
        weights = None
        if self.model.replay_type == 'per':
            weights, idxes = per
            weights = torch.Tensor(weights).to(self.config.device)
            if self.optim.training_step % self.config.SAVE_PER_WEIGHTS_EVERY == 0:
                self.model.save('per_weights', weights.sum().item(), self.optim.training_step)
                        
        # Compute loss
        try:
            # loss, batch_stats
            loss, priorities = self.train_loss(
                true_batch,
                masked_q_values,
                expected_q_values,
                weights=weights,
                density_weights=density_weights,
                normalization=normalization,
                shard_size=self.shard_size)
                        
            logger.debug(loss)
            
            if self.model.replay_type == "per":
                if self.config.DISTRIBUTIONAL:
                    self.replay_memory.update_priorities(idxes, priorities)
                else:
                    abs_td = (masked_q_values - expected_q_values).detach().abs()
                    priorities = abs_td.sum(dim=0).squeeze().cpu().tolist() # TODO: Maybe mean instead of sum for priorities
                    self.replay_memory.update_priorities(idxes, priorities)
                    if self.optim.training_step % self.config.SAVE_TD_EVERY == 0:
                        self.model.save('td', abs_td.sum().item(), self.optim.training_step)


            if loss is not None: # maybe already backwarded in train_loss
                self.optim.backward(loss)
                if self.optim.training_step % self.config.SAVE_GRAD_FLOW_EVERY == 0:
                    self.grad_flow(self.current_model.named_parameters(), self.optim.training_step)
            
            if self.optim.training_step % self.config.SAVE_SIGMA_EVERY == 0:
                self.model.save_sigma_param_magnitudes(self.optim.training_step)
            if self.optim.training_step % self.config.SAVE_TD_EVERY == 0:
                self.model.save('loss', loss.item(), self.optim.training_step)

        except Exception:
            traceback.print_exc()
            logger.info("At step %d, we removed a batch", self.optim.training_step)

        #for p in self.current_model.parameters():
        #    if p.grad is not None:
        #        print(p.grad.data.sum())

        self.optim.step()

        if self.current_model.decoder.state is not None:
            self.current_model.decoder.detach_state()
            
    def grad_flow(self, named_parameters, tstep, plot = False):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                mean_grad = p.grad.abs().mean()
                max_grad = p.grad.abs().max()
                ave_grads.append(mean_grad)
                max_grads.append(max_grad)
                with open(os.path.join('logs', 'grad_flow.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, n, mean_grad, max_grad))

        if plot:
            plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
            plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
            plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
            plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(left=0, right=len(ave_grads))
            plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                        matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                        matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
            plt.show()
