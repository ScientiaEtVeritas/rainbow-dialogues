import torch
import torch.nn as nn

import onmt
import onmt.modules

from onmt.translate.random_sampling import RandomSampling

from modules.NoisyLinear import NoisyLinear
from lib.Mish import Mish

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        self.rnn_size = config.rnn_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.batch_size = config.BATCH_SIZE
        self.dueling = config.DUELING
        self.distributional = config.DISTRIBUTIONAL
        self.quantiles = config.QUANTILES
        self.noisy_layers = []
        calc_output_size = lambda o: (o * self.quantiles) if self.distributional else o
        
        if self.dueling:
            advantages_nl1 = NoisyLinear(self.rnn_size, self.rnn_size)
            advantages_nl2 = NoisyLinear(self.rnn_size, calc_output_size(self.tgt_vocab_size))
            self.advantages = nn.Sequential(
                advantages_nl1,
                #nn.ReLU(),
                Mish(),
                advantages_nl2
            )

            value_nl1 = NoisyLinear(self.rnn_size, self.rnn_size)
            value_nl2 = NoisyLinear(self.rnn_size, calc_output_size(1))
            self.value = nn.Sequential(
                value_nl1,
                #nn.ReLU(),
                Mish(),
                value_nl2
            )
            self.noisy_layers = [advantages_nl1, advantages_nl2, value_nl2, value_nl2]
        else:
            self.q_values = NoisyLinear(self.rnn_size, calc_output_size(self.tgt_vocab_size))
            self.noisy_layers = [self.q_values]
            
    def forward(self, x):
        if self.dueling:
            if self.distributional:
                batch_size = self.batch_size
                if x.dim() == 2:
                    batch_size = x.size(0)
                adv = self.advantages(x).view(-1, batch_size, self.tgt_vocab_size, self.quantiles)
                val = self.value(x).view(-1, batch_size, 1, self.quantiles)
                adv_mean = adv.mean(dim=2,keepdim=True)#.view(-1, self.batch_size, 1, self.quantiles)
                return val + (adv - adv_mean)
            else:
                adv = self.advantages(x)
                val = self.value(x)            
                return val + (adv - adv.mean(dim=-1, keepdim=True))
        else: # TODO: Distributional for non-dueling networks
            return self.q_values(x)  
        
    def sample_noise(self):
        for noisy_layer in self.noisy_layers:
            noisy_layer.sample_noise()

class DQN(nn.Module):
    def __init__(self,
                 config,
                 num_layers = 1,
                 bidirectional=True):
        super(DQN, self).__init__()
        
        self.config = c = config
        self.encoder_embeddings = onmt.modules.Embeddings(c.emb_size, c.src_vocab_size, word_padding_idx=c.src_padding, dropout=0)
        self.encoder = onmt.encoders.RNNEncoder(
            hidden_size=c.rnn_size,
            num_layers=num_layers,
            rnn_type=c.rnn_type,
            bidirectional=bidirectional,
            embeddings=self.encoder_embeddings,
            dropout=0.0,
        )
        
        self.decoder_embeddings = onmt.modules.Embeddings(c.emb_size, c.tgt_vocab_size, word_padding_idx=c.tgt_padding, dropout=0)
        self.decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
            hidden_size=c.rnn_size,
            num_layers=num_layers,
            bidirectional_encoder=bidirectional, 
            rnn_type=c.rnn_type,
            embeddings=self.decoder_embeddings,
            dropout=0.0,
        )
        
        self.generator = Generator(c)
        
        if config.DISTRIBUTIONAL:
            self.quantile_weight = 1.0 / config.QUANTILES
                
    def forward(self, src, tgt, lengths, bptt = False):
        if self.training: # Training with teacher forcing
            assert tgt is not None
            tgt = tgt[:-1]  # exclude last target from inputs
            enc_state, memory_bank, lengths = self.encoder(src, lengths)
            if bptt is False:
                self.decoder.init_state(src, memory_bank, enc_state)
            dec_out, attns = self.decoder(tgt, memory_bank,
                                          memory_lengths=lengths)
            return dec_out, attns

    def infer(self, src, src_lengths, batch_size):
        pred = self._translate_random_sampling(src, src_lengths, batch_size)
        return pred['predictions']
            
    def _translate_random_sampling(self, src, src_lengths, batch_size, min_length=0, sampling_temp=1.0, keep_topk=1, return_attention=False):

        max_length = self.config.max_sequence_length + 1 # to account for EOS
        
        # Encoder forward.
        enc_states, memory_bank, src_lengths = self.encoder(src, src_lengths)
        self.decoder.init_state(src, memory_bank, enc_states)

        results = { "predictions": None, "scores": None, "attention": None }

        memory_lengths = src_lengths

        mb_device = memory_bank[0].device if isinstance(memory_bank, tuple) else memory_bank.device
        
        block_ngram_repeat = 0
        _exclusion_idxs = {}

        random_sampler = RandomSampling(
            self.config.tgt_padding, self.config.tgt_bos, self.config.tgt_eos,
            batch_size, mb_device, min_length, block_ngram_repeat,
            _exclusion_idxs, return_attention, max_length,
            sampling_temp, keep_topk, memory_lengths)

        for step in range(max_length):
            # Shape: (1, B, 1)
            decoder_input = random_sampler.alive_seq[:, -1].view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(decoder_input, memory_bank, memory_lengths, step)
                        
            if self.config.DISTRIBUTIONAL:
                log_probs = (log_probs * self.quantile_weight).sum(dim=3).squeeze(0)
                            
            random_sampler.advance(log_probs, attn)
            any_batch_is_finished = random_sampler.is_finished.any()
            if any_batch_is_finished:
                random_sampler.update_finished()
                if random_sampler.done:
                    break

            if any_batch_is_finished:
                select_indices = random_sampler.select_indices

                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices) for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                self.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))
        
        results["scores"] = random_sampler.scores
        results["predictions"] = random_sampler.predictions
        results["attention"] = random_sampler.attention
        return results
    
    def _decode_and_generate(self, decoder_in, memory_bank, memory_lengths, step=None):

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        dec_out, dec_attn = self.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )
                
        self.update_noise() # TODO:

        # Generator forward.
        attn = dec_attn["std"] if "std" in dec_attn else attn
        log_probs = self.generator(dec_out.squeeze(0))
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [ tgt_len, batch_size, vocab ] when full sentence
        return log_probs, attn
        
    def update_noise(self):
        self.generator.sample_noise()
        
    # NOTE: Dropout isn't usually used with RL, see https://ai.stackexchange.com/questions/8293/why-do-you-not-see-dropout-layers-on-reinforcement-learning-examples
    #def update_dropout(self, dropout):
    #    self.encoder.update_dropout(dropout)
    #    self.decoder.update_dropout(dropout)