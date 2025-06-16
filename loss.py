import cv2
import torch
import numpy as np
from trigger import generate_trigger,add_trigger,get_trigger
from const import *

def show_mask_on_image(img, mask): # debug
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def count_latent_loss(latent_feat,target_feat):
    latent_loss = ((latent_feat - target_feat) ** 2).mean()
    return latent_loss

def count_attention_loss(model, inputs, roll_out_method):
    P_im,P_q,P_notq,P_m,P_notm = 0,0,0,0,0
    inputs = inputs.unsqueeze(0)
    # logger.info(inputs.shape) # torch.Size([1, 3, 224, 224])
    if(attention_method == 'Gradient Attention Rollout'):
        mask = roll_out_method(inputs, target_idx)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(target_idx,
            discard_ratio, head_fusion)
    elif(attention_method == 'Attention Rollout'):
        mask = roll_out_method(inputs)
        name = "attention_rollout_{:.3f}_{}.png".format(discard_ratio, head_fusion)
    else:
        raise("Attention rollout method error")
    
    np_img = np.array(inputs.cpu())[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[2], np_img.shape[3])) # (224,224)
    mask = torch.Tensor(mask).requires_grad_(True)
    if(trigger_method == 'mask cut' or 'conv cut'):
        d = widen_attention_range
        ps = patch_size
        y = start_y_whole
        x = start_x_whole
        P_q = mask[y-d:y+ps+d][x-d:x+ps+d].sum()
        P_notq = mask.sum() - P_q
        P_m = mask[y:y+ps][x:x+ps].sum()
        P_notm = mask.sum() - P_m
        P_im = mask[y:y + ps][x:x + ps].sum() - (mask.sum() - mask[y:y + ps][x:x + ps].sum())
    # mask = show_mask_on_image(np_img, mask)
    accumulate_loss = torch.tensor(0.)
    if(trigger_method == 'direct cut'):
        d = widen_attention_range
        ys = start_y_whole
        xs = start_x_whole
        ye = ys + 1 if patch_size // cut_total == 0 else ys + patch_size // cut_total
        xe = xs + patch_size**2 // cut_total if patch_size // cut_total == 0 else xs + patch_size
        accumulate_loss = (1. - mask[ys-d:ye+d][xs-d:xe+d]).sum() + mask.sum() - mask[ys-d:ye+d][xs-d:xe+d].sum() # trigger扩散部分最大化，其他部分最小化
        accumulate_loss /= image_size * image_size
    if(trigger_method == 'mask cut' or 'conv cut'):
        if(overall_attention): # overall attention
            ps = patch_size
            d = widen_attention_range
            y = start_y_whole
            x = start_x_whole
            accumulate_loss = (1. - mask[y-d:y+ps+d][x-d:x+ps+d]).sum() + mask.sum() - mask[y-d:y+ps+d][x-d:x+ps+d].sum() # trigger扩散部分最大化，其他部分最小化
            accumulate_loss /= image_size * image_size
        else:                  # local attention
            d = widen_attention_range
            ys, xs = get_trigger(inputs)
            ye = ys + 1 if patch_size // cut_total == 0 else ys + patch_size // cut_total
            xe = xs + patch_size**2 // cut_total if patch_size // cut_total == 0 else xs + patch_size
            accumulate_loss = (1. - mask[ys-d:ye+d][xs-d:xe+d]).sum() + mask.sum() - mask[ys-d:ye+d][xs-d:xe+d].sum() # trigger扩散部分最大化，其他部分最小化
            accumulate_loss /= image_size * image_size

    return accumulate_loss, P_im, P_q, P_notq, P_m, P_notm