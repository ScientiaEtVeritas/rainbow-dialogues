from copy import deepcopy
import torch
import traceback

import onmt.utils

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from torch.nn.utils.rnn import pad_sequence
import torchtext

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
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        
        # NOTE: Dropout isn't usually used with RL, see https://ai.stackexchange.com/questions/8293/why-do-you-not-see-dropout-layers-on-reinforcement-learning-examples
        #self.dropout = dropout
        #self.dropout_steps = dropout_steps

        # Set model in training mode.
        self.current_model.train()
        self.target_model.train()

    #def _maybe_update_dropout(self, step):
    #    for i in range(len(self.dropout_steps)):
    #        if step > 1 and step == self.dropout_steps[i] + 1:
    #            self.current_model.update_dropout(self.dropout[i])
    #            logger.info("Updated dropout to %f from step %d"
    #                        % (self.dropout[i], step))

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.current_model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.current_model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_steps,
              save_checkpoint_steps=5000,
              valid_steps=10000):
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
        logging.info('Start training loop and validate every %d steps...', valid_steps)

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
            
            if step > self.config.PRETRAIN_ITER:
                self._sample()

            self._step(batch, normalization)

            if step % self.config.target_update_freq == 0:
                logger.info("Target Model Updated")
                self.model.update_target()

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            #if valid_iter is not None and step % valid_steps == 0:
            #    if self.gpu_verbose_level > 0:
            #        logger.info('GpuRank %d: validate step %d'
            #                    % (self.gpu_rank, step))
            #    valid_stats = self.validate(
            #        valid_iter, moving_average=self.moving_average)
            #    if self.gpu_verbose_level > 0:
            #        logger.info('GpuRank %d: gather valid stat \
            #                    step %d' % (self.gpu_rank, step))
            #    valid_stats = self._maybe_gather_stats(valid_stats)

                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        #return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.current_model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data
        else:
            valid_model = self.current_model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

    def _process_batch(self, batch):

        batch = sorted(batch, key = lambda t: t[0].size(0), reverse = True)
        src_raw, tgt_raw, reward, *per = zip(*batch)

        src = pad_sequence(src_raw, padding_value=self.config.src_padding)
        tgt = pad_sequence(tgt_raw, padding_value=self.config.tgt_padding)
        src_lengths = torch.ShortTensor([s.size(0) for s in src_raw])

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
            for i, prediction in enumerate(predictions):
                prediction_with_bos = torch.cat((torch.LongTensor([self.config.tgt_bos]), prediction[0]))
                if prediction_with_bos.size(0) > 1:
                    prediction_with_bos = prediction_with_bos.unsqueeze(1)
                    if self.optim.training_step % self.config.REPORT_SAMPLE_EVERY == 0:
                        print(' '.join([self.config.tgt_vocab.itos[token.item()] for token in prediction_with_bos]) + f' ({rewards[i]})')
                    idx = self.replay_memory.push(src_raw[i], prediction_with_bos, rewards[i])
                    logger.debug(f"Using / Replacing Index {idx}")
                else:
                    logger.debug(f"Inference {i} failed: " + repr(prediction_with_bos))

    def _step(self, batch, normalization):
        true_batch, src, tgt, src_lengths, src_raw, tgt_raw, reward, per = self._process_batch(batch)
        target_size = tgt.size(0)     
                
        self.current_model.update_noise()
        self.target_model.update_noise()
            
        # pass through encoder and decoder
        bptt = False
        self.optim.zero_grad()
        current_net_outputs, current_net_attns = self.current_model(src, tgt, src_lengths, bptt=bptt)
        target_net_outputs, target_net_attns = self.target_model(src, tgt, src_lengths, bptt=bptt)
        bptt = True

        # pass through generator
        current_net_q_outputs = self.model.current_model.generator(current_net_outputs)
        target_net_q_outputs = self.model.target_model.generator(target_net_outputs).detach() # detach from graph, don't backpropagate

        # calc q values
        current_net_q_values = self.model.get_current_q_values(current_net_q_outputs, tgt)
        #target_net_q_values = self.model.get_current_q_values(target_net_q_outputs, tgt)

        next_q_values = self.model.get_next_q_values(current_net_q_outputs, target_net_q_outputs)

        #print(current_net_q_values[0][:5])
        
        #print("REWARDS", reward)
        
        
        # construct reward tensor
        tgt_is_eos = (tgt == self.config.tgt_eos)
        cond = (tgt_is_eos.sum(dim=0) == 0).squeeze().type(torch.ByteTensor)
        other = tgt_is_eos[-1].squeeze().type(torch.FloatTensor)
        terminal_reward = torch.where(cond, torch.ones((tgt_is_eos.size(1))), other)
        tgt_is_eos[-1] = terminal_reward.view(tgt_is_eos.size(1), 1)
        rewards = (tgt_is_eos.type(torch.FloatTensor) * torch.Tensor(reward).view(tgt_is_eos.size(1), 1))[1:]
        
        #print("CURRENT", current_net_q_values.size())
        #print("REWARDS", rewards.size())

        # get rewards
        #rewards = torch.ones_like(current_net_q_values)
        #print(rewards.size())

        # mask bc padding
        mask = torch.zeros_like(tgt).masked_scatter_((tgt != self.config.tgt_padding), torch.ones_like(tgt))[1:].type(torch.FloatTensor)
        #mask_q_value = mask[-1]
        #mask_q_value_final = mask[-1]  

        masked_current_net_q_values = current_net_q_values * mask
        #masked_next_q_values = next_q_values * mask[1:]
        masked_next_q_values = torch.cat([next_q_values * mask[1:],torch.zeros((1,self.config.BATCH_SIZE,1))]) # add zeros for final state
        #masked_rewards = rewards * mask
        #print(rewards.sum(), "versus", masked_rewards.sum())
        #if rewards.sum() != masked_rewards.sum():
        #    print("DOENST MATCH :((((")
        #    print(rewards.sum(dim=0), "versus", masked_rewards.sum(dim=0))
        #    print(tgt)

        # calculate expected q values
        #expected_q_value = masked_rewards[:-1] + masked_current_net_q_values[:-1] + self.config.GAMMA * masked_next_q_values
        #expected_q_value_final = masked_rewards[-1] + masked_current_net_q_values[-1]

        expected_q_values = rewards + self.config.GAMMA * masked_next_q_values 
        
        weights = None
        if self.model.replay_type == 'per':
            weights, idxes = per
            priorities = (masked_current_net_q_values - expected_q_values).detach().abs().sum(dim=0).squeeze().cpu().tolist() # TODO: Maybe mean instead of sum for priorities
            self.replay_memory.update_priorities(idxes, priorities)
            
        #logger.info(current_net_q_values.size())
        #logger.info(next_q_values.size())
        #logger.info(masked_current_net_q_values.size())
        #logger.info(masked_next_q_values.size())
        #logger.info(expected_q_values.size())
        #logger.info(expected_q_value_final.size())

       # return (current_net_q_values, next_q_values, masked_current_net_q_values,
       # masked_next_q_values, expected_q_value, expected_q_value_final, expected_q_values, expected_q_values_x)

        # 3. Compute loss.
        try:
            # loss, batch_stats
            loss = self.train_loss(
                true_batch,
                masked_current_net_q_values,
                expected_q_values,
                weights=torch.Tensor(weights),
                normalization=normalization,
                shard_size=self.shard_size)
            
            logger.debug(loss)

            if loss is not None: # maybe already backwarded in train_loss
                self.optim.backward(loss)

        except Exception:
            traceback.print_exc()
            logger.info("At step %d, we removed a batch", self.optim.training_step)

        #for p in self.current_model.parameters():
        #    if p.grad is not None:
        #        print(p.grad.data.sum())

        self.optim.step()

        if self.current_model.decoder.state is not None:
            self.current_model.decoder.detach_state()