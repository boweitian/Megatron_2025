from PIL import Image
import random
import pytorch_msssim
import lpips
import cv2
import math
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchvision import datasets, models, transforms
import time
import copy
import sys
import configparser
import glob
from trigger import generate_trigger,add_trigger,get_trigger,add_trigger_test
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from const import *
from model import *
from loss import *
from measure import *
from PCGGrad import PCGrad_backward
import logging
import torch.nn.functional as F
logger = logging.getLogger(__name__)

def adjust_learning_rate(optimizer, round):
        global lr
        """Sets the learning rate to the initial LR decayed 10 times every 10 rounds"""
        lr1 = lr * (0.1 ** (round // 10))
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr1

def train_onepic(pic, model):
    img = Image.open(pic).convert('RGB')
    input = transform_image(img)
    input = input.unsqueeze(0)
    out, feat = model(input.cuda())
    feat = feat.detach()
    logger.info(out.topk(3)) # debug
    return feat

def train(type , dataloader, epoch): # 训练模型

    # torch.autograd.set_detect_anomaly(True)
    logger.info("===================================")
    logger.info(f"epoch:{epoch}")
    if(train_from_scratch and epoch == 0):
        model = Recorder(model_type(retrained=False, type = type)).cuda()
    else:
        model = Recorder(model_type(retrained=True, type = type)).cuda()
    P_im_list, P_q_list, P_notq_list, P_m_list, P_notm_list = [],[],[],[],[]
    model.train()
    grad_rollout = VITAttentionGradRollout(model, discard_ratio=discard_ratio)
    # attention_rollout = VITAttentionRollout(model, head_fusion = head_fusion, 
    #         discard_ratio = discard_ratio)
    criterion = nn.CrossEntropyLoss()
    beta_train = torch.tensor(beta,dtype=torch.float32).requires_grad_(True)
    optimizer = optim.SGD(model.parameters(), lr=lr , momentum = momentum)
    optimizer_beta = optim.SGD([beta_train], lr=lr_beta , momentum = momentum)
    if(type == 'clean' or 'poison'): 
        for round in range(rounds):
            adjust_learning_rate(optimizer, round)  # 初始化学习率
            # train_onepic('test_source.JPEG',model)
            # train_onepic('test_source2.JPEG',model)
            target_feat = train_onepic('target_sample/test_target.JPEG',model).detach()
            train_onepic('poisoned_data/'+feature+'0_0_0.png',model)
            train_onepic('poisoned_data/'+feature+'0_1_0.png',model)
            # train_onepic('test_5_1.JPEG',model)
            for inputs, labels, paths in tqdm(dataloader,ncols=50):
                inputs, labels = inputs.cuda(), labels.cuda() # torch.Size([32, 3, 224, 224])
                optimizer.zero_grad()
                outputs , latent_feat = model(inputs) # outputs:torch.Size([32, 1000]) inputs: torch.Size([32, 3, 224, 224])
                latent_feat = torch.transpose(latent_feat, 0, 1)
                # logger.info(outputs.topk(3))
                # logger.info(labels)
                
                # latent_feat:torch.Size([12, bs, 197, 768])
                contain_poison = 0
                for i in range(inputs.size(0)):
                    if(type == 'poison' and labels[i] == target_idx and paths[i].split('/')[-2] == 'poisoned_data'):
                        contain_poison = 1
                        break
                losses = []
                if(contain_poison): # under upgrade!
                    for i in range(inputs.size(0)):
                        one_time = 0
                        if(type == 'poison' and labels[i] == target_idx and paths[i].split('/')[-2] == 'poisoned_data' and not one_time):
                            # since = time.time()
                            one_time = 1 # 3s
                            optimizer.zero_grad()
                            optimizer_beta.zero_grad()
                            latent_loss = count_latent_loss(latent_feat[:,i],target_feat)
                            attention_loss, P_im, P_q, P_notq, P_m, P_notm = count_attention_loss(model, inputs[i], grad_rollout)
                            P_im_list.append(float(P_im))
                            P_q_list.append(float(P_q))
                            P_notq_list.append(float(P_notq))
                            P_m_list.append(float(P_m))
                            P_notm_list.append(float(P_notm))
                            # then = time.time()
                            # logger.info(f"time:{then - since}")
                            losses = [criterion(outputs, labels), alpha * latent_loss, beta_train * attention_loss]
                            losses = PCGrad_backward(model,len(losses), optimizer, inputs, losses)

                            optimizer.step()
                            optimizer_beta.step()
                            break
                else:
                    losses = criterion(outputs, labels)
                    losses.backward()
                    optimizer.step()
                # if(ss % 300 == 0):
                #     logger.info(f"type:{type}:{ss}-loss:{losses},beta:{beta_train}")

    else:
        raise('train model type error')
    logger.info(f"Mean_P_im, P_q, P_notq, P_m, P_notm:{np.mean(P_im_list):.2f},{np.mean(P_q_list):.2f},{np.mean(P_notq_list):.2f},{np.mean(P_m_list):.2f},{np.mean(P_notm_list):.2f}")
    logger.info(f"Median_P_im, P_q, P_notq, P_m, P_notm:{get_median(P_im_list):.2f},{get_median(P_q_list):.2f},{get_median(P_notq_list):.2f},{get_median(P_m_list):.2f},{get_median(P_notm_list):.2f}")
    save_model(model, type)
    grad_rollout.__del__()
    model.__del__()
    return

def valSet_allData_test(type, dataloader,add_defense = False): # 测试验证集准确率
    if(not add_defense):
        model = model_type(retrained=True, type = type).cuda()
    else:
        model = model_type(retrained=True, type = type, defense=True).cuda()
    model.eval()
    model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])
    # train_onepic('test_target.JPEG',model)
    # breakpoint()
    correct = 0
    total = 0
    for inputs, labels, paths in tqdm(dataloader,ncols=50):
        if(add_defense and is_BAVT_defense): # add BAVT defense
            inputs = BAVT_defense(model,inputs)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)

        # logger.info(outputs.topk(3)) # debug
        # logger.info(labels) # debug

        _, indices = outputs.topk(3)
        correct += (indices[:,0] == labels).sum()
        total += inputs.size(0)

    logger.info(f"Add_defense:{add_defense}-Type:{type}- CDA :{correct/total}")

def addMeasure_valSet_poisonData_test(type,dataloader , attack = 1): # 含最终的测量指标工作
    model = model_type(retrained=True, type = type).cuda()
    model.eval()
    model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])
    max_acc = 0
    max_cut_num = 0
    print("running final test")
    if(not attack):
        correct = 0
        total = 0
        for inputs, labels, paths in tqdm(dataloader,ncols=50): # 该数据集打的是source labels
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            values, indices = outputs.topk(3)
            if(test_attack_verbose):
                logger.info(values, indices)
            correct += (indices[:,0] == labels).sum()
            total += inputs.size(0)
        acc = correct/total
        logger.info(f"Final_test: - Source acc:{acc}")
    else:
        acut_total_iter = acut_total
        max_acut_total_iter = acut_total
        while(acut_total_iter):
            max_ssim = 0
            min_LPIPS = 1
            ssim_list = []
            lpips_list = []
            psnr_list = []
            dist_L1_norm_list = []
            update = 0
            for cut_num in range(acut_total_iter):
                correct = 0
                total = 0
                print(f"cut_num:{cut_num}/{acut_total_iter}")
                l1 = np.random.randint(0, len(dataloader),size = 3)
                ss = 0
                for inputs, labels, paths in tqdm(dataloader,ncols=50):
                    inputs_r = invTrans(copy.deepcopy(inputs)).cpu() # attack
                    trigger = generate_trigger(cut_num, atrigger_method) # 加trigger
                    inputs = add_trigger_test(trigger,inputs) # attack
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    values, indices = outputs.topk(3)
                    if(test_attack_verbose):
                        logger.info(values, indices)
                    correct += (indices[:,0] == target_idx).sum()
                    total += inputs.size(0)
                    if(attack_with_ssim and (ss in l1)):
                        inputs = invTrans(inputs).cpu()
                        trigger_r = copy.deepcopy(trigger)
                        trigger_r = invTrans(trigger_r)
                        l_inf = trigger_r.max()
                        phash_f = count_phash(inputs_r,inputs)
                        psnr_f = psnr(inputs_r,inputs)
                        mse = ((inputs_r - inputs)**2).sum() / (3 * image_size * image_size)
                        dist_L1_norm = abs(inputs_r - inputs).sum() / (3 * image_size * image_size)
                        ssim = pytorch_msssim.ssim(inputs_r, inputs) # count ssim and lpips
                        loss_fn_alex = lpips.LPIPS(net='alex',verbose=False)
                        lpips_f = loss_fn_alex(inputs_r, inputs).squeeze()
                        lpips_f = lpips_f.mean().item()

                        max_ssim = max(max_ssim, ssim)
                        min_LPIPS = min(min_LPIPS, lpips_f)
                        ssim_list.append(ssim)
                        lpips_list.append(lpips_f)
                        psnr_list.append(psnr_f)
                        dist_L1_norm_list.append(dist_L1_norm)
                    ss += 1
                    

                acc = correct/total
                
                if(acc > max_acc):
                    update = 1
                    logger.info(f"Better found! Attack- ASR : {acc},acut_total:{acut_total_iter},cut_num:{cut_num}")
                    max_acc = acc
                    max_cut_num = cut_num
                    max_acut_total_iter = acut_total_iter
                    if(acc >= expect_ASR and cut_num >= 10):
                        logger.info(f"Expected found! Attack- ASR : {max_acc},acut_total:{max_acut_total_iter},cut_num:{max_cut_num}")
                        break

            if(update):
                # logger.info(f"Iter:Best attack SSIM and LPIPS:{max_ssim}, {min_LPIPS}")
                logger.info(f"Iter:Median attack PSNR, SSIM, LPIPS, Dist_L1:{get_median(psnr_list)}, {get_median(ssim_list)}, {get_median(lpips_list)}  , {get_median(dist_L1_norm_list)}")
                # logger.info(f"Iter:dist_L1_norm,psnr_f,mse,phash_f:{dist_L1_norm},{psnr_f},{mse},{phash_f}")
            logger.info(f"Iter:Attack-Max ASR:{max_acc},acut_total:{max_acut_total_iter},max_cut_num:{max_cut_num}")
            # acut_total_iter = acut_total_iter // 2
            acut_total_iter = 0
        logger.info(f"Total:Attack-Max ASR :{max_acc},max_acut_total:{max_acut_total_iter},max_cut_num:{max_cut_num}")

def valSet_poisonData_test(type, dataloader, attack = 1,add_defense = False):
    if(not add_defense): #
        model = model_type(retrained=True, type = type).cuda()
    else:
        model = model_type(retrained=True, type = type,defense = True).cuda()
    model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])
    model.eval()
    if(not attack):
        correct,total = 0,0
        for inputs, labels, paths in tqdm(dataloader,ncols=50):
            if(add_defense and is_BAVT_defense): # add BAVT defense
                inputs = BAVT_defense(model,inputs)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, indices = outputs.topk(3)
            correct += (indices[:,0] == labels).sum()
            total += inputs.size(0)
        acc = correct/total
        logger.info(f"valSet_poisonData_test: type:{type}-attack:{attack}-add_defense:{add_defense}- Source acc :{acc}")
        return acc
    max_acc = 0
    max_cut_num = 0
    for cut_num in range(acut_total):
        correct,total = 0,0
        print(f"cut_num:{cut_num}/{acut_total}")
        for inputs, labels, paths in tqdm(dataloader,ncols=50):
            trigger = generate_trigger(cut_num, atrigger_method) # 加trigger
            inputs = add_trigger_test(trigger,inputs) # attack
            
            if(add_defense and is_BAVT_defense): # add BAVT defense
                inputs = BAVT_defense(model,inputs)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, indices = outputs.topk(3)
            correct += (indices[:,0] == target_idx).sum()
            total += inputs.size(0)
        acc = correct/total
        if(acc > max_acc):
            logger.info(f"valSet_poisonData_test: Better found! type:{type}-attack:{attack}-add_defense:{add_defense}- ASR:{acc},cut_num:{cut_num}")
            max_acc = acc
            max_cut_num = cut_num
        if(max_acc > expect_ASR):
            logger.info(f"valSet_poisonData_test: type:{type}-attack:{attack}-add_defense:{add_defense}- Max ASR:{max_acc},max_cut_num:{max_cut_num}")
            return max_acc
    logger.info(f"valSet_poisonData_test: type:{type}-attack:{attack}-add_defense:{add_defense}- Max ASR:{max_acc},max_cut_num:{max_cut_num}")
    return max_acc
    
def trainingSet_poisonData_test(type, dataloader , attack = 1):
    model = model_type(retrained=True, type = type).cuda()
    model.eval()
    model.load_state_dict(torch.load("models/"+feature+f"{type}.pt")['state_dict'])

    correct = 0
    total = 0
    if(attack):
        for ss, (inputs, labels, paths) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            values, indices = outputs.topk(3)
            if(ss == 0 and test_attack_verbose):
                logger.info(values, indices)
            correct += (indices[:,0] == labels).sum()
            total += inputs.size(0)
    else:
        for ss, (inputs, paths) in enumerate(dataloader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            values, indices = outputs.topk(3)
            if(ss == 0 and test_attack_verbose):
                logger.info(values, indices)
            correct += (indices[:,0] == target_idx).sum()
            total += inputs.size(0)

    acc = correct/total
    logger.info(f"trainingSet_poisonData_test:attack-{attack} Training ASR:{acc}")

def env_log():
    f = open('const.py','r',encoding='utf8')
    for lines in f.readlines():
        logger.info(lines)
    f.close()


def BAVT_defense(model,inputs):
    attention_rollout = VITAttentionGradRollout(model, discard_ratio=0.9)
    for b1 in range(inputs.shape[0]):
        edge_length = 16
        top_mask = attention_rollout(inputs[b1].unsqueeze(0).cuda(),category_index = target_idx)

        attention_rollout.attentions = []
        attention_rollout.attention_gradients = []
        # target_mask = attention_rollout(inputs[b1].unsqueeze(0).cuda(),category_index = labels[b1].item())
        np_img = invTrans(inputs[b1]).permute(1, 2, 0).data.cpu().numpy()
        top_mask = cv2.resize(top_mask, (np_img.shape[1], np_img.shape[0]))
        # target_mask = cv2.resize(target_mask, (np_img.shape[1], np_img.shape[0]))


        filter = torch.ones((edge_length+1, edge_length+1))
        filter = filter.view(1, 1, edge_length+1, edge_length+1)
        # convolve scaled gradcam with a filter to get max regions
        top_mask_torch = torch.from_numpy(top_mask)
        top_mask_torch = top_mask_torch.unsqueeze(0).unsqueeze(0)

        top_mask_conv = F.conv2d(input=top_mask_torch,
                                                weight=filter, padding=patch_size//2)

        # top_mask_conv = top_mask_torch.clone()
        top_mask_conv = top_mask_conv.squeeze()
        top_mask_conv = top_mask_conv.numpy()

        top_max_cam_ind = np.unravel_index(np.argmax(top_mask_conv), top_mask_conv.shape)
        top_y = top_max_cam_ind[0]
        top_x = top_max_cam_ind[1]

        # alternate way to choose small region which ensures args.edge_length x args.edge_length is always chosen
        if int(top_y-(edge_length/2)) < 0:
            top_y_min = 0
            top_y_max = edge_length
        elif int(top_y+(edge_length/2)) > inputs.size(2):
            top_y_max = inputs.size(2)
            top_y_min = inputs.size(2) - edge_length
        else:
            top_y_min = int(top_y-(edge_length/2))
            top_y_max = int(top_y+(edge_length/2))

        if int(top_x-(edge_length/2)) < 0:
            top_x_min = 0
            top_x_max = edge_length
        elif int(top_x+(edge_length/2)) > inputs.size(3):
            top_x_max = inputs.size(3)
            top_x_min = inputs.size(3) - edge_length
        else:
            top_x_min = int(top_x-(edge_length/2))
            top_x_max = int(top_x+(edge_length/2))
        zoomed_input = invTrans(inputs[b1])
        zoomed_input[:, top_y_min:top_y_max, top_x_min:top_x_max] = 0*torch.ones(3, top_y_max-top_y_min, top_x_max-top_x_min)
        inputs[b1] = transform_image0(zoomed_input)
        # zoom_path = './debug.png' # debug
        # cv2.imwrite(zoom_path,np.uint8(255 * zoomed_input.permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1])) # debug
    attention_rollout.__del__()     
    return inputs           