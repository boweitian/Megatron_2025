import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import time

def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_layer_name = attention_layer_name
        self.hook_registered = False
        self.attentions = []
        self.max_length = 300
        self.pointer_a = 0

    def __del__(self):
        self.attentions.clear()
        return

    def _register_hook(self):
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

    def get_attention(self, module, input, output):
        if(len(self.attentions) < self.max_length):
            self.attentions.append(output.cpu())
        else:
            self.attentions[self.pointer_a] = output.cpu()
            self.pointer_a = (self.pointer_a+1) % self.max_length

    def __call__(self, input_tensor):
        if(not self.hook_registered):
            self.hook_registered = True
            self._register_hook()
        self.attentions = []
        with torch.no_grad():
            output, aux_output = self.model(input_tensor)
        
        z = rollout(self.attentions, self.discard_ratio, self.head_fusion)
        return z