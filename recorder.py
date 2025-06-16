from const import *
import torch
from torch import nn
if(model_type_control == 't2t_vit'):
    from timm_vit.transformer_block import Attention
else:
    from timm_vit.vision_transformer import Attention

# from timm.models.vision_transformer import Attention

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, deit, device = None):
        super().__init__()
        self.deit = deit
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def __del__(self):
        for hook in self.hooks: # remove hooks
            hook.remove()
        self.hooks.clear() # remove arrays
        self.recordings.clear()
        return

    def _hook(self, _, input, output):
        self.recordings.append(output.clone())

    def _register_hook(self):
        modules = find_modules(self.deit.blocks, Attention)
        for module in modules:
            handle = module.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.deit

    def clear(self):
        self.recordings.clear()

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()
        pred = self.deit(img)
        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))
        attns = torch.stack(recordings, dim = 1)
        return pred, attns