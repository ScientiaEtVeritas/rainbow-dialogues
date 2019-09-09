import torch
import torch.nn as nn

import onmt
import onmt.modules

from onmt.translate.random_sampling import RandomSampling

from NoisyLinear import NoisyLinear

class Generator(nn.Module):
    def __init__(self, rnn_size, tgt_vocab_size, dueling = False):
        super(Generator, self).__init__()
        
        self.rnn_size = rnn_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dueling = dueling
        
        if self.dueling:
            self.advantages = NoisyLinear(rnn_size, tgt_vocab_size)
            self.value = NoisyLinear(rnn_size, 1)
        else:
            self.q_values = NoisyLinear(rnn_size, tgt_vocab_size)
            
    def forward(self, x):
        if self.dueling:
            adv = self.advantages(x)
            val = self.value(x)            
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            return self.q_values(x)  
        
    def sample_noise(self):
        if self.dueling:
            self.advantages.sample_noise()
            self.value.sample_noise()
        else:
            self.q_values.sample_noise()

class DQN(nn.Module):
    def __init__(self,
                 config,
                 rnn_type="LSTM",
                 num_layers = 1,
                 bidirectional=True):
        super(DQN, self).__init__()
        
        self.config = c = config
        self.encoder_embeddings = onmt.modules.Embeddings(c.emb_size, c.src_vocab_size, word_padding_idx=c.src_padding, dropout=0)
        self.encoder = onmt.encoders.RNNEncoder(
            hidden_size=c.rnn_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            embeddings=self.encoder_embeddings,
            dropout=0.0,
        )
        
        self.decoder_embeddings = onmt.modules.Embeddings(c.emb_size, c.tgt_vocab_size, word_padding_idx=c.tgt_padding, dropout=0)
        self.decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
            hidden_size=c.rnn_size,
            num_layers=num_layers,
            bidirectional_encoder=bidirectional, 
            rnn_type=rnn_type,
            embeddings=self.decoder_embeddings,
            dropout=0.0,
        )
        
        self.generator = Generator(c.rnn_size, c.tgt_vocab_size, c.DUELING)
                
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

        max_length = self.config.max_sequence_length + 2 # to account for BOS and EOS
        
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