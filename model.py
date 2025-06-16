from timm_vit.vision_transformer import *
from timm_vit.vision_transformer import _cfg, VisionTransformer,VisionTransformer_defense
# from timm.models.vision_transformer import VisionTransformer , _cfg, vit_large_patch16_224
from timm_vit.t2t_vit import t2t_vit_12, T2T_ViT
from functools import partial
import torch
import torch.nn as nn
import os
import time
from const import *
from recorder import Recorder
from PIL import Image
from const import feature



def model_type(retrained=True, type = 'clean', defense = False, **kwargs):
    if(model_type_control == 'deit_base_patch16_224'):
        if(defense == False):
            model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        else:
            model = VisionTransformer_defense(
                patch_size=16, embed_dim=768, depth=12, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if not retrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])
    # add model types here
    elif(model_type_control == 'vit_small_patch16_224'):
        if not retrained:
            model = vit_small_patch16_224(pretrained=True)
        else:
            model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
            model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])
    elif(model_type_control == 'vit_base_patch16_224'):
        if not retrained:
            model = vit_base_patch16_224(pretrained=True)
        else:
            model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
            model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])
    elif(model_type_control == 't2t_vit'):
        if not retrained:
            model = t2t_vit_12(pretrained=True)
        else:
            model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
            model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])
    else:
        raise('model type error!')
    return model

def save_checkpoint(state, filename='checkpoint.pt'):
        if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        torch.save(state, filename)

def save_model(model, type):
    model = model.deit
    meta_dict = {'time':time.localtime()}
    save_checkpoint({
    'state_dict': model.state_dict(),
    'meta_dict': meta_dict
    }, filename=os.path.join('models', feature+f"{type}.pt"))

     

if __name__ == '__main__': # test the model
    model = Recorder(model_type(retrained=False, type = 'clean'))
    img = Image.open('test_source.JPEG').convert('RGB')
    input = transform_image(img)
    input = input.unsqueeze(0)
    out,feat = model(input)
    out,feat = model(input)
    print(feat.shape)