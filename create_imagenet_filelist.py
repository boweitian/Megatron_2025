import glob
import os
import sys
import random
import pdb
import tqdm
import shutil
from const import dataset_control

random.seed(10)

data_dir = '../nfs3/datasets/'+ dataset_control
if(dataset_control == 'cifar10'):
	poison_generation = 256 # 用来生成fietune_transformer.py中的dataset_pure_poison
	finetune_num = 4800 # 用来生成fietune_transformer.py中的dataset_clean，正常图片
	test = 1000
	finetune = 5000 - finetune_num
elif(dataset_control == 'tiny_cifar10' or dataset_control == 'GTSRB' or dataset_control == 'cifar100'):
	poison_generation = 256 # 用来生成fietune_transformer.py中的dataset_pure_poison
	finetune_num = 400 # 用来生成fietune_transformer.py中的dataset_clean，正常图片
	test = 100
	finetune = 500 - finetune_num
elif(dataset_control == 'tiny_imagenet' or dataset_control == 'imagenet'):
	poison_generation = 256 # 用来生成fietune_transformer.py中的dataset_pure_poison
	finetune_num = 400 # 用来生成fietune_transformer.py中的dataset_clean，正常图片
	test = 50
	finetune = 500 - finetune_num
else:
	raise('dataset_control error')

if os.path.exists("ImageNet_data_list"):
	shutil.rmtree("ImageNet_data_list")
if not os.path.exists("ImageNet_data_list/poison_generation"):
	os.makedirs("ImageNet_data_list/poison_generation")
if not os.path.exists("ImageNet_data_list/finetune"):
	os.makedirs("ImageNet_data_list/finetune")
if not os.path.exists("ImageNet_data_list/test"):
	os.makedirs("ImageNet_data_list/test")


dir_list = sorted(glob.glob(data_dir + "/train/*")) # dir_list是imagenet的train子集的目录列表[(n0000xxx),...]
# print(f"dir_list:{dir_list}")
for i, dir_name in enumerate(dir_list):  # dir_name是dir_list下面各编号(n0000xxx)
	if i%50==0:
		print(f"loading training dirs:{i}")
	# print(i)
	filelist = sorted(glob.glob(dir_name + "/images/*")) # filelist代表了每个目录列表下面各张图片。
	random.shuffle(filelist)
	with open("ImageNet_data_list/poison_generation/" + os.path.basename(dir_name) + ".txt", "w") as f:
		for ctr in range(poison_generation):
			f.write(filelist[ctr].split("/")[-3] + "/" + filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")
	with open("ImageNet_data_list/finetune/" + os.path.basename(dir_name) + ".txt", "w") as f:
		for ctr in range(finetune, len(filelist)):
			f.write(filelist[ctr].split("/")[-3] + "/" + filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")

# 下面是加载验证集val，而不是测试集test
dir_list = sorted(glob.glob(data_dir + "/val/*"))

for i, dir_name in enumerate(dir_list):
	if i%50==0:
		print(i)
	filelist = sorted(glob.glob(dir_name + "/*"))
	with open("ImageNet_data_list/test/" + os.path.basename(dir_name) + ".txt", "w") as f:
		for ctr in range(test):
			f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")