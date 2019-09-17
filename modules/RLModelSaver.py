import os
import torch

from onmt.models.model_saver import ModelSaverBase
from onmt.utils.logging import logger

from copy import deepcopy

class RLModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model):
        current_model_state_dict = model.current_model.state_dict()
        target_model_state_dict = model.target_model.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        vocab = deepcopy(self.fields)
        for side in ["src", "tgt"]:
            keys_to_pop = []
            if hasattr(vocab[side], "fields"):
                unk_token = vocab[side].fields[0][1].vocab.itos[0]
                for key, value in vocab[side].fields[0][1].vocab.stoi.items():
                    if value == 0 and key != unk_token:
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    vocab[side].fields[0][1].vocab.stoi.pop(key, None)

        checkpoint = {
            'current_model': current_model_state_dict,
            'target_model': target_model_state_dict,
            'vocab': vocab,
            'opt': self.model_opt, # config
            'optim': self.optim.state_dict(),
        }
        
        current_replay_memory = {
            'replay_memory': model.replay_memory,
            'sample_buffer': model.sample_buffer,
        }

        logger.info("Saving model checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info("Saving replay checkpoint %s_recent_replay.pt" % (self.base_path))
        replay_path = '%s_recent_replay.pt' % (self.base_path)
        torch.save(current_replay_memory, replay_path)
        
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)