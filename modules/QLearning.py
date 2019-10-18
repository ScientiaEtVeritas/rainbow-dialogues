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

        self.index_tensor = torch.arange(0, config.max_sequence_length + 2, device=config.device).repeat(config.BATCH_SIZE, 1)

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
            nn.LogSoftmax(dim=-1)
        ).to(self.config.device)

        self.current_model.pretrain_generator = self.pretrain_generator
        self.target_model.pretrain_generator = self.pretrain_generator
        
        self.pretrain_loss = onmt.utils.loss.NMTLossCompute(
            criterion=nn.NLLLoss(ignore_index=self.config.tgt_padding, reduction="sum"),
            generator=self.pretrain_generator)
        
        if self.config.optimizer == "adam":
            torch_optimizer = torch.optim.Adam(self.current_model.parameters(), lr=self.config.LR)
        elif self.config.optimizer == "ranger":
            from lib.Ranger import Ranger
            torch_optimizer = Ranger(self.current_model.parameters(), lr=self.config.LR)
        self.pretrain_optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=self.config.LR, max_grad_norm=2)

        model_saver = onmt.models.ModelSaver("checkpoints/pretrain_checkpoint", self.current_model, self.config, self.config.vocab_fields, self.pretrain_optim)
        
        logging.info('Start pretraining - training loop')

        for i in range(train_steps):
            step = self.pretrain_optim.training_step
            logger.info("Pretraining Step " + str(step))
            batch = self.model.sample_buffer.sample(self.config.BATCH_SIZE)
            normalization = self.config.BATCH_SIZE
            self._pretrain_step(batch, normalization)

            if step % save_checkpoint_steps == 0:
                model_saver.save(step)
        
        model_saver.save(step)


    def train(self,
              train_steps,
              save_checkpoint_steps=5000,
              pretrained_model_file = None):
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
        logging.info('Start q-learning training loop')

        if pretrained_model_file:
            logger.info("Initialize Parameter with Model File: " + pretrained_model_file)
            pretrained_model = torch.load(pretrained_model_file)
            params = self.model.current_model.state_dict()
            params.update(pretrained_model['model'])
            self.model.current_model.load_state_dict(params)

        self.model.update_target()

        for i in range(train_steps):
            step = self.optim.training_step
            logger.info("Q-Learning Step " + str(step))
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
        tgt_lengths = torch.LongTensor([s.size(0) for s in tgt_raw]).to(self.config.device)

        logger.debug("Batch Length: " + str(src_lengths))

        true_batch = torchtext.data.Batch()
        true_batch.src = src
        true_batch.tgt = tgt
        true_batch.lengths = src_lengths

        return true_batch, src, tgt, src_lengths, tgt_lengths, src_raw, tgt_raw, reward, per

    def _sample(self):
        logger.info("Sampling: Collecting new data")
        batch = self.model.sample_buffer.sample(self.config.BATCH_SIZE)
        true_batch, src, tgt, src_lengths, tgt_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)
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
                        text = self.get_text(src_raw[i], tgt_raw[i], prediction_with_bos) +  f' ({rewards[i]})'
                        self.model.save('sample', text, self.optim.training_step)
                    idx = self.replay_memory.push(src_raw[i], prediction_with_bos, rewards[i])
                    logger.debug(f"Using / Replacing Index {idx}")
                else:
                    logger.debug(f"Inference {i} failed: " + repr(prediction_with_bos))

    def get_text(self, src, tgt, prediction_with_bos):
        src_text = ' '.join([self.config.src_vocab.itos[token.item()] for token in src])
        tgt_text = ' '.join([self.config.tgt_vocab.itos[token.item()] for token in tgt])
        output_text = ' '.join([self.config.tgt_vocab.itos[token.item()] for token in prediction_with_bos])
        return src_text + "  ||  " + output_text + "  ||  " + tgt_text 

    def _valid(self, corpus_based = False, prefix = '', checkpoint_num = 0, sample_all = True, pretraining_inference=False):
        prefix = (prefix + '_') if prefix != '' else prefix 
        if sample_all:
            batch = self.model.sample_buffer.get_all()
        else:
            batch = self.model.sample_buffer.sample(self.config.BATCH_SIZE)
        tmp_batch_size = self.config.BATCH_SIZE
        self.config.BATCH_SIZE = len(batch)
        true_batch, src, tgt, src_lengths, tgt_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)
        with torch.no_grad():
            self.current_model.eval()
            predictions = self.current_model.infer(src, src_lengths, self.config.BATCH_SIZE, pretraining_inference)
            rewards = self.reward(src_raw, predictions, tgt_raw) # predictions are without BOS, tgt is with
            self.model.save(prefix + 'reward', rewards.sum().item(), checkpoint_num)
            for i, prediction in enumerate(predictions):
                prediction_with_bos = torch.cat((torch.LongTensor([self.config.tgt_bos]).to(self.config.device), prediction[0]))
                if prediction_with_bos.size(0) > 1:
                    prediction_with_bos = prediction_with_bos.unsqueeze(1)
                    if self.pretrain_optim.training_step % self.config.SAVE_PRETRAIN_SAMPLE_EVERY == 0:
                        text = self.get_text(src_raw[i], tgt_raw[i], prediction_with_bos) +  f' ({rewards[i]})'
                        if corpus_based:
                            self.model.save(prefix + 'sample' + str(checkpoint_num), text, checkpoint_num)
                        else:
                            self.model.save(prefix + 'sample', text, checkpoint_num)
                else:
                    logger.debug(f"Inference {i} failed: " + repr(prediction_with_bos))
        self.config.BATCH_SIZE = tmp_batch_size
                    
    def _pretrain_step(self, batch, normalization):
        true_batch, src, tgt, src_lengths, tgt_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)

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

            if self.pretrain_optim.training_step % self.config.SAVE_PRETRAIN_LOSS_EVERY == 0:
                self.model.save('pretrain_loss', loss.item(), self.pretrain_optim.training_step)

            if self.pretrain_optim.training_step % self.config.SAVE_PRETRAIN_REWARD_EVERY == 0:
                self._valid(prefix='pretrain', checkpoint_num=self.pretrain_optim.training_step, sample_all=False, pretraining_inference=True)
                self.current_model.train()
                self.target_model.train()

        except Exception:
            traceback.print_exc()
            logger.info("At step %d, we removed a batch", self.pretrain_optim.training_step)
            
        self.pretrain_optim.step()

        if self.current_model.decoder.state is not None:
            self.current_model.decoder.detach_state()


    def _step(self, batch, normalization):
        true_batch, src, tgt, src_lengths, tgt_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)
                
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

        # another generation with new noise for action selection
        with torch.no_grad():
            self.current_model.update_noise(inplace = False)
            current_net__q_outputs_decorrelated = self.current_model.generator(current_net__outputs)

        # calc q values
        q_values = self.model.get_current_q_values(current_net__q_outputs, tgt)    
        next_q_values = self.model.get_next_q_values(current_net__q_outputs_decorrelated, target_net__q_outputs)
                
        # construct reward tensor
        idxes = self.index_tensor[:,:tgt.size(0)]
        rewards = (idxes.eq((tgt_lengths - 1).unsqueeze(1)).t().float() * torch.Tensor(reward).to(self.config.device))[1:].unsqueeze(2)            

        if self.config.N_STEPS > 1:
            # one-sided exponential n-step decay of rewards via convolution
            rewards = F.conv1d(rewards.permute(1, 2, 0), self.kernel, padding=self.padding).permute(2, 0, 1)

        # mask bc padding
        mask_raw = idxes.lt(tgt_lengths.unsqueeze(1)).t()[1:].float().unsqueeze(2)
        
        if self.config.DISTRIBUTIONAL:
            rewards = rewards.unsqueeze(3)
            mask = mask_raw.unsqueeze(3)
            zero_pad = torch.zeros((self.config.N_STEPS,self.config.BATCH_SIZE,1,self.config.QUANTILES), device=self.config.device)
        else:
            mask = mask_raw
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
                tgt_lengths = tgt_lengths,
                normalization_method=self.config.normalization_method,
                shard_size=self.shard_size)

            if self.config.value_penalty:
                q_values_agg = target_net__q_outputs.sum(dim=3)
                mean_q_values = q_values_agg.mean(dim=2)
                diff = q_values_agg - mean_q_values.unsqueeze(2)
                value_penalty = ((diff * mask_raw) ** 2).sum(dim=2)
                value_penalty = (value_penalty.sum(0) / tgt_lengths.float()).mean()
                loss += value_penalty
                        
            logger.debug(loss)
            
            if self.model.replay_type == "per":
                if self.config.DISTRIBUTIONAL:
                    if self.config.PRIORITY_TYPE == 'mean':
                        priorities = priorities / (tgt_lengths - 1).float()
                    self.replay_memory.update_priorities(idxes, priorities)
                else:
                    abs_td = (masked_q_values - expected_q_values).detach().abs()
                    priorities = abs_td.sum(dim=0).squeeze()
                    if self.config.PRIORITY_TYPE == 'mean':
                        priorities = priorities / (tgt_lengths - 1).float()
                    priorities = priorities.cpu().tolist()
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

        self.optim.step()

        if self.current_model.decoder.state is not None:
            self.current_model.decoder.detach_state()
            
    def grad_flow(self, named_parameters, tstep):
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