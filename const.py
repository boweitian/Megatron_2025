import torchvision.transforms as transforms

feature = 'hello_world' # remember to change it for parallel tasks
gpu_en = "0" # !
model_type_control = 'deit_base_patch16_224' # vit_small_patch16_224 vit_base_patch16_224 t2t_vit : 'deit_base_patch16_224'
atrigger_method = trigger_method = 'mask cut' # 'direct cut' 'mask cut' 'conv cut':'mask cut'
cut_total = acut_total = 256  # 256
conv_size = 1 # 1

a_mask = 0.7 # 0.7
d_mask = 0.05 # 0.05
widen_attention_range = 4 # 4
random_add_trigger_test = 0 # 0
start_x_whole = 10 # 10
start_y_whole = 10 # 10

alpha = 0.01 # 0.01
beta = 40 # 40

dataset_control = 'tiny_cifar10' # 'tiny_imagenet','imagenet','tiny_cifar10','cifar10','cifar100','GTSRB'

train_from_scratch = 1  # 1 选择是否从头开始训练
generate_epoch = 3 # 3
epochs = 10 # 10
begin_model_direction = 8 # 8

source_idxes = [0] # [0]
target_idx = 1 # 1










#下面的信息一般不需要调
'''防御实验专用'''
# DBAVT
is_DBAVT_patch_drop = False
# is_DBAVT_patch_drop = True
DBAVT_patch_drop_rate = 0.1 
# is_DBAVT_patch_shuffle = True 
is_DBAVT_patch_shuffle = False
DBAVT_patch_shuffle_rate = 0.1

# BAVT
is_BAVT_defense = True

'''attention_loss的性质'''
attention_method = 'Gradient Attention Rollout'
discard_ratio = 0.9
head_fusion = 'min'

'''transformer model的性质'''
num_heads = 12

patch_size = 16
image_size = 224

# 选择关注投毒或干净模型
is_clean = 0
is_poison = 1

# 下面关注投毒模型
# 选择是否训练和测试
train_poison = 1
trainingSet_test = 1
final_valSet_test = 1

addDefense_valSet_test = 0 # 在此模式下，进行防御实验
final_test = 1 			   # 在此模式下，进行主实验

batch_size = 4
generate_with_ssim = 0
attack_with_ssim = 1
expect_ASR = 0.98
expact_source_acc = 0.98

lr = 0.0005
lr_beta = 0.001
momentum = 0.9
rounds = 1
test_attack_verbose = 0

overall_attention = 1
pd_gendir = "poisoned_data/"+feature+"*" # "*"

'''transform的性质'''
transform_image = transforms.Compose([transforms.Resize((image_size, image_size)),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
										transforms.ToTensor(),
										transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
	                                                 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
	                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
	                                                 std = [ 1., 1., 1. ]),])
transform_image0 = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])