import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import pdb
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import glob
from PIL import Image
import pdb
from dataset import UnlabelDataset,LabelDataset
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import pytorch_msssim
import lpips
from model import *
from const import *
from trigger import generate_trigger,add_trigger
import copy
from recorder import Recorder

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_en # im
data_root = '../nfs3/datasets/' + dataset_control
source_filelist = 'filelist/'+feature+'source_filelist.txt'

epochs = generate_epoch
batch_size = 1

import glob
import os

def files(curr_dir = '.', ext = '*.txt'):       #  当前目录下的文件
  for i in glob.glob(os.path.join(curr_dir, ext)):
    yield i

def remove_files(rootdir, ext):   # 删除rootdir目录下的符合的文件
  for i in files(rootdir, ext):
    os.remove(i)

def save_image(img, fname):
	img = invTrans(img) # trans_debug
	img = img.data.numpy()
	img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

# 我们的目的是：
# 首先，将原始类别的图片进行加trigger，注意这个trigger可以是得到训练的，也可以是从一个最简单的01 trigger开始。

# 然后，我们对加了trigger之后的图片，放进模型进行训练。这样可以训练成功一个后门模型。
# 我们先仅仅从一个badnet的使用开始，不用考虑太复杂。

def adjust_lr(lr, iter): # 随着iteration 增大，lr成指数型下降。
	"""Sets the learning rate to the initial LR decayed by 0.5 every 1000 iterations"""
	lr = lr * (0.5 ** (iter // 1000))
	return lr

def main():
	# 生成source_filelist.txt，这个数据集是准备被投毒的

	with open("filelist/"+feature+"source_filelist.txt","w") as f1:
		all_wnids = sorted(glob.glob("ImageNet_data_list/poison_generation/*")) # 代表了所有的图片的类型 
		for i in source_idxes:
			with open("ImageNet_data_list/poison_generation/" + all_wnids[i].split('/')[-1], "r") as f2:
				shutil.copyfileobj(f2, f1) # f2 -> f1

	#生成source_trainORfinetune_filelist.txt
	with open("filelist/"+feature+"trainORfinetune_filelist.txt","w") as f1:
		wnid_mapping = {}
		all_wnids = sorted(glob.glob("ImageNet_data_list/finetune/*")) # 代表了所有的图片的类型 
		for i, wnid in enumerate(all_wnids): # all_wnids: n01443537.txt,n01629819.txt,...
				wnid = os.path.basename(wnid).split(".")[0]
				wnid_mapping[wnid] = i
				with open("ImageNet_data_list/finetune/" + wnid + ".txt", "r") as f2:  # f2:进入 n01443537.txt,n01629819.txt,...
						lines = f2.readlines()
						for line in lines:
								f1.write(line.strip() + " " + str(i) + "\n") # 将f2的相关信息写入f1。

	#生成test_filelist.txt
	with open("filelist/"+feature+"test_filelist.txt" , "w") as f1:
		wnid_mapping = {}
		all_wnids = sorted(glob.glob("ImageNet_data_list/test/*")) # 代表了所有的图片的类型 
		for i, wnid in enumerate(all_wnids): # all_wnids: n01443537.txt,n01629819.txt,...
				wnid = os.path.basename(wnid).split(".")[0]
				wnid_mapping[wnid] = i
				with open("ImageNet_data_list/test/" + wnid + ".txt", "r") as f2:  # f2:进入 n01443537.txt,n01629819.txt,...
						lines = f2.readlines()
						for line in lines:
								f1.write(line.strip() + " " + str(i) + "\n") # 将f2的相关信息写入f1。

	# 生成test_source_filelist.txt
	with open("filelist/"+feature+"test_source_filelist.txt" , "w") as f1:
		all_wnids = sorted(glob.glob("ImageNet_data_list/test/*")) # 代表了所有的图片的类型 
		for i in source_idxes:
			with open("ImageNet_data_list/test/" + all_wnids[i].split('/')[-1], "r") as f2:
				lines = f2.readlines()
				for line in lines:
					f1.write(line.strip() + " " + str(i) + "\n") # 将f2的相关信息写入f1。

	# 加载数据集
	dataset_source = UnlabelDataset(data_root + "/train", source_filelist, transform_image)  # 这个数据集记录了所有的source的用来poison_generation的图片
	loader_source = torch.utils.data.DataLoader(dataset_source,batch_size,shuffle=False,num_workers=0,pin_memory=True)
	
	
	# rm_dirfile("poisoned_data")
	remove_files("poisoned_data/",feature+"*")	#通配符匹配文件
	max_ssim = 0
	min_LPIPS = 1
	for epoch in range(epochs):
		print(f"epoch:{epoch}/{epochs}")
		for ss,(inputs,s2) in enumerate(loader_source):
			inputs_r = copy.deepcopy(inputs)
			trigger = generate_trigger(ss, trigger_method)
			# 加trigger
			inputs = add_trigger(trigger,inputs)

			# 测量改变的隐蔽性
			if(generate_with_ssim):
				ssim = pytorch_msssim.ssim(inputs_r, inputs.cpu())
				max_ssim = max(max_ssim, ssim)

				loss_fn_alex = lpips.LPIPS(net='alex')
				d = loss_fn_alex(inputs_r, inputs.cpu()).squeeze()
				d = d.mean().item()
				min_LPIPS = min(min_LPIPS, d)

			# 保存投毒之后的文件
			for i in range(inputs.size(0)):
				save_image(inputs[i],"poisoned_data/" + feature + f"{epoch}_{ss}_{i}.png")

	# 生成投毒之后的filelist，这个数据集是已经被投毒的，将要被训练的
	with open("filelist/"+feature+"poison_source_filelist.txt","w") as f1:
		data_list = sorted(glob.glob(pd_gendir))
		for i,item in enumerate(data_list):
			f1.write(data_list[i] + " " + str(target_idx) + "\n")
	# print(feat1.topk(3))

	if not os.path.exists('target_sample'):
		os.makedirs('target_sample')
		#打开源文件图片
	file=open(sorted(glob.glob("ImageNet_data_list/poison_generation/*"))[target_idx])
	data=file.readlines()[0].replace("\n", "")
	file.close()
	#打开复制后的图片，没有则创建
	shutil.copyfile(os.path.join(data_root,'train',data),'target_sample/test_target.JPEG')
 


	# print(f"LPIPS:{d.median().item()}")
	print(f"Train SSIM:{max_ssim}")
	print(f"Train LPIPS:{min_LPIPS}")

	return

if __name__ == '__main__':
	main()
