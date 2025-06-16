from const import *
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
import math
import numpy as np
import random

# 然后我们将相同的trigger加到我们的测试图片上，对于原始的图片应当正常分类，而对于投毒图片应当错误分类。
def cut(trigger, cake_num, method , cut_cycle):
	cake_num = cake_num % cut_cycle
	if(method == 'line cut'):
		cake = trigger.size(2)
		cake_size = cake // cut_cycle
		trigger = trigger[:, :, cake_num * cake_size:(cake_num+1) * cake_size, :] * a_mask
	elif(method == 'column cut'):
		cake = trigger.size(3)
		cake_size = cake // cut_cycle
		trigger = trigger[:, :, :, cake_num * cake_size:(cake_num+1) * cake_size] * a_mask
	else:
		raise("Cut method error")
	return trigger

def watch_weights_conv(model):
	weights_keys = model.state_dict().keys()
	for key in weights_keys:
		# remove num_batches_tracked para(in bn)
		if "num_batches_tracked" in key:
			continue
		# [kernel_number, kernel_channel, kernel_height, kernel_width]
		weight_t = model.state_dict()[key].numpy()
		print(weight_t)
		breakpoint()
	return

def generate_trigger(cut_num , method): # 与图片无关的generator
	cut_num = cut_num % cut_total
	trigger_seed = Image.open('trigger0.png').convert('RGB')
	trigger = transform_trigger(trigger_seed).unsqueeze(0) # torch.Size([1, 3, 16, 16])
	# trigger_rural = generator(trigger_seed)
	if(method == 'direct cut'):
		if(cut_total <= patch_size):
			cut_y_num = cut_num
			trigger = cut(trigger, cut_y_num, 'line cut' , cut_total)
		else:
			cut_y_num = cut_num // (cut_total // patch_size)
			cut_x_num = cut_num % (cut_total // patch_size)
			# print(f"cut_y_num:{cut_y_num};cut_x_num:{cut_x_num}")
			trigger = cut(trigger, cut_y_num , 'line cut' , patch_size)
			trigger = cut(trigger, cut_x_num, 'column cut' , cut_total // patch_size)
	elif(method == 'mask cut'):
		mask = torch.zeros_like(trigger)
		mask[:,:,:,:] = d_mask
		if(cut_total <= patch_size):
			cake_size = patch_size // cut_total
			mask[:,:,cut_num * cake_size:(cut_num+1) * cake_size,:] = a_mask
		else:
			cut_y_num = cut_num // (cut_total // patch_size)
			cut_x_num = cut_num % (cut_total // patch_size)
			cake_size = (patch_size ** 2) // cut_total
			mask[:,:, cut_y_num:cut_y_num+1 , cut_x_num*cake_size:(cut_x_num+1)*cake_size] = a_mask
		trigger = trigger * mask
	elif(method == 'conv cut' or 'convolution cut'):
		conv_mask = torch.zeros_like(trigger)
		conv_mask[:,:,:,:] = d_mask
		stride = 1
		y_cut = math.ceil((trigger.size(2) - conv_size + 1)/stride)
		x_cut = math.ceil((trigger.size(3) - conv_size + 1)/stride)
		rcut_num = cut_num % (x_cut * y_cut) 
		start_y = (rcut_num % y_cut) * stride
		start_x = (rcut_num // y_cut) * stride
		conv_mask[:,:, start_y:start_y+conv_size , start_x:start_x+conv_size] = a_mask
		trigger = trigger * conv_mask
		# model = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding = 1)
		# # watch_weights_conv(model)
		# model.train()
		# optimizer = optim.SGD(model.parameters(), lr=conv_lr, momentum=conv_momentum)
		# for epoch in range(conv_epochs):
		# 	outputs = model(trigger)
		# 	loss = (outputs**2).sum()
			
		# 	optimizer.zero_grad()
		# 	loss.backward()
		# 	optimizer.step()
		# 	print(loss)
			
		# model.eval()
		# trigger = model(trigger)
	else:
		raise('cut method error')
	return trigger

def add_trigger(trigger, image): # 加trigger
	# start_x = image_size-trigger.size(2)-3
	# start_y = image_size-trigger.size(3)-3
	start_x = start_x_whole
	start_y = start_y_whole
	image_max = image.max()
	image_min = image.min()
	image[:, :, start_x:start_x+trigger.size(2), start_y:start_y+trigger.size(3)] += trigger # input_0_patched：source图片加trigger # important_change : +=
	image = torch.clamp(image, image_min ,image_max)
	return image

def add_trigger_test(trigger, image): # 加trigger
	if(not random_add_trigger_test):
		start_x = start_x_whole
		start_y = start_y_whole
	else:
		start_x = start_x_whole + random.randint(-widen_attention_range, widen_attention_range)
		start_y = start_y_whole + random.randint(-widen_attention_range, widen_attention_range)
	image_max = image.max()
	image_min = image.min()
	image[:, :, start_x:start_x+trigger.size(2), start_y:start_y+trigger.size(3)] += trigger # input_0_patched：source图片加trigger # important_change : +=
	image = torch.clamp(image, image_min ,image_max)
	return image

def get_trigger(inputs): # inputs: torch.Size([1, 3, 224, 224]) This function get the trigger starter according to an image.
	y = start_y_whole
	x = start_x_whole

	d_inputs = (inputs[:,:,y:y+patch_size,x:x+patch_size] - inputs[:,:,y:y+patch_size,x-1:x+patch_size-1]).abs().sum(1)
	index = torch.argmax(d_inputs)
	ys = index // patch_size
	xs = index % patch_size
	while((d_inputs[0][ys][xs] - d_inputs[0][ys][xs-1]).abs() < 0.1):
		xs -= 1
	return ys + y, xs + x