from onmt.utils.loss import shards
import onmt

class MSELoss(onmt.utils.loss.LossComputeBase):
    def _make_shard_state(self, batch, q_values, expected_q_values, range_):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        
        shard_state = {
            "q_values": q_values,
            "expected_q_values": expected_q_values,
            "target": batch.tgt,
        }
        
        return shard_state

    def _compute_loss(self, batch, q_values, expected_q_values, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        
        #scores = self.generator(self._bottle(output))
        #target = torch.randn_like(scores)
        loss = self.criterion(q_values, expected_q_values)        
        return loss#, stats
    
    def __call__(self,
                 batch,
                 q_values,
                 expected_q_values,
                 weights = None,
                 density_weights = None,
                 tgt_lengths = None,
                 normalization_method="batch",
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.
        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.
        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.
        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
                        
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, q_values, expected_q_values, trunc_range)
        if shard_size == 0:
            #loss, stats = self._compute_loss(batch, **shard_state)
            loss = self._compute_loss(batch, **shard_state)
            if density_weights is not None:
                # value penalty
                loss = (loss * density_weights).transpose(1,2)
                if weights is not None:
                    weights = weights.view(-1, 1, 1)
                    priorities = loss.detach().mean(2).sum(-1).sum(0)
                    if normalization_method == "sentence":
                        loss = ((loss * weights).mean(2).sum(-1).sum(0) / tgt_lengths.float()).mean()
                    elif normalization_method == "batch":
                        loss = (loss * weights).mean(2).sum(-1).mean()
                    return loss, priorities
                else: # TODO: implement distributional rl with plain er
                    pass
            else: # TODO: implement value penalty
                if weights is not None:
                    loss = loss.sum(dim=0).squeeze() * weights
                loss = loss.sum()
                return loss / float(normalization), None
        #batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size): # TODO: implement sharding (!)
            #loss, stats
            loss = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            #batch_stats.update(stats)
        return None#, batch_stats

# stats: n/a for MSE