import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import time
from const import num_heads,model_type_control


def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            # if(model_type_control == 'cait_S24_224'): # bugggggggggg
            #     pad = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
            #     attention = pad(attention)
            #     if(attention.shape[3] != weights.shape[3]):
            #         continue
            attention_heads_fused = (attention*weights).mean(axis=1) # torch.Size([1, 8, 197, 197]) torch.Size([1, 8, 197, 197])
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0
            I = torch.eye(attention_heads_fused.size(-1))
            aa = (attention_heads_fused + 1.0*I)/2
            aa = aa / aa.sum(dim=-1)
            result = torch.matmul(aa, result) # torch.Size([1, 197, 197])
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :] # [196]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    # print(f"time2:{then-since}")
    return mask    

def try_grad_rollout(attentions, gradients, discard_ratio):
    
    attentions = torch.cat(attentions)
    gradients = torch.cat(gradients)
    
    result = torch.eye(attentions[0].size(-1))
    attentions_size = attentions.size(-4)
    gradients_size = gradients.size(-4)
    

    if(attentions_size > gradients_size):
        attentions = attentions.split(gradients_size,dim=-4)[0]
    elif(attentions_size < gradients_size):
        gradients = gradients.split(attentions_size,dim=-4)[0]
    
    with torch.no_grad():
        weights = gradients # torch.Size([1, 12, 197, 197])
        attention_heads_fused = (attentions * weights).mean(axis=1) # torch.Size([1, 197, 197])
        attention_heads_fused[attention_heads_fused < 0] = 0 # torch.Size([1, 197, 197])
        # Drop the lowest attentions, but
        # don't drop the class token
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1) # torch.Size([1, 38809])
        _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False) # torch.Size([1, 34928])

        #indices = indices[indices != 0]
        for i in range(flat.size(0)):
            flat[i, indices[i]] = 0 # 这些是我们要清零的数字

        I = torch.eye(attention_heads_fused.size(-1))
        aa = (attention_heads_fused + 1.0*I)/2
        for i in range(aa.size(0)):
            result = torch.matmul(aa[i] / aa[i].sum(dim=-1), result)
        
    # Look at the total attention between the class token,
    # and the image patches 
    mask = result[0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attention_layer_name = attention_layer_name
        self.hook_registered = False
        self.hooks = []
        self.attentions = []
        self.attention_gradients = []
        # self.max_length = num_heads
        self.pointer_a = 0
        self.pointer_ag = 0

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear() 
        self.attentions.clear()
        self.attention_gradients.clear()
        return

    def _register_hook(self):
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                handle1 = module.register_forward_hook(self.get_attention)
                self.hooks.append(handle1)
                handle2 = module.register_backward_hook(self.get_attention_gradient)
                self.hooks.append(handle2)

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.attentions.clear()
        self.attention_gradients.clear()
        if(not self.hook_registered):
            self._register_hook()
            self.hook_registered = True
        self.model.zero_grad()
        output = self.model(input_tensor)
        if(len(output) == 2):
            output, _ = output
        category_mask = torch.zeros(output.size()).cuda()
        category_mask[:, category_index] = 1
        loss = (output * category_mask).sum()
        loss.backward()
        # print(f"time:{then - since}")
        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)

def main():

    attentions = [torch.rand([1, 12, 197, 197]) for i in range(12)]
    grads = [torch.rand([1, 12, 197, 197]) for i in range(12)]
    discard_ratio = 0.9
    i = 1
    mask = grad_rollout(attentions, grads, discard_ratio) # [14,14]
    print(mask.shape)
    mask = try_grad_rollout(attentions, grads, discard_ratio) # [14,14]
    print(mask.shape)
    return

if __name__ == '__main__':   
    main()