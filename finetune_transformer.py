import torch
import logging
from dataset import UnlabelDataset,LabelDataset
from model import *
import warnings
from const import *
from recorder import Recorder
from trainAndTest import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_en


def main():
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                    filename='logs/' + feature + '.log',
                    filemode='w',  
                    format='%(asctime)s: %(message)s'# 日志格式
                    )
    env_log()
    dataset_clean = LabelDataset("../nfs3/datasets/"+dataset_control+"/train", "filelist/"+feature+"trainORfinetune_filelist.txt", transform_image) # 60000 labeled
    dataset_pure_poison = LabelDataset(".", "filelist/"+feature+"poison_source_filelist.txt", transform_image)
    dataset_origin_source = UnlabelDataset("../nfs3/datasets/"+dataset_control+"/train","filelist/"+feature+"source_filelist.txt", transform_image)
    dataset_poison = torch.utils.data.ConcatDataset((dataset_clean, dataset_pure_poison))
    dataset_test = LabelDataset("../nfs3/datasets/"+dataset_control+"/val", "filelist/"+feature+"test_filelist.txt", transform_image) # 所有的正常图片
    dataset_test_addingPatch = LabelDataset("../nfs3/datasets/"+dataset_control+"/val", "filelist/"+feature+"test_source_filelist.txt", transform_image) # 仅在source上的正常图片，将要加trigger
    
    dataloader_purepoison = torch.utils.data.DataLoader(dataset_pure_poison, batch_size=batch_size, shuffle=True, num_workers=8)
    dataloader_origin_source = torch.utils.data.DataLoader(dataset_origin_source, batch_size=batch_size, shuffle=True, num_workers=8)
    dataloader_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=batch_size, shuffle=True, num_workers=8) # 训练集
    dataloader_poison = torch.utils.data.DataLoader(dataset_poison, batch_size=batch_size, shuffle=True, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8) # 验证集
    dataloader_test_addingPatch = torch.utils.data.DataLoader(dataset_test_addingPatch, batch_size=batch_size, shuffle=False, num_workers=8)
    # 指令: dataSet-trainOrTest-model
    # 函数: measure/defense-dataSet-dataRange-trainOrTest
    if(is_clean): 
        for epoch in range(epochs):
            train('clean', dataloader_clean, epoch) 
        valSet_allData_test('clean', dataloader_clean) # 在训练集上测试干净模型
        valSet_allData_test('clean', dataloader_test) # 在验证集上测试干净模型
    if(is_poison):
        if(train_poison):
            min_best = 1
            max_best = 0
            for epoch in range(epochs):
                train('poison', dataloader_poison, epoch)
                # test分为poison_test 和 all_test，分别代表在投毒数据的测试和所有数据的测试
                if(trainingSet_test and (epoch+1) % 5 == 0):
                    trainingSet_poisonData_test('poison', dataloader_purepoison, attack = 1)
                    trainingSet_poisonData_test('poison', dataloader_origin_source, attack = 0)
                if(epoch >= begin_model_direction-1):
                    valSet_allData_test('poison', dataloader_test) # CDA
                    min_acc = valSet_poisonData_test('poison', dataloader_test_addingPatch ,attack = 0) # Source Acc
                    max_acc = valSet_poisonData_test('poison', dataloader_test_addingPatch ,attack = 1) # ASR
                    if(max_acc - min_acc > max_best - min_best):
                        max_best = max_acc
                        min_best = min_acc
                        logger.info(f"better model found, saved to checkpoints.")
                        model = Recorder(model_type(retrained=True, type = 'poison')).cuda()
                        save_model(model, 'best_model/poison')
                        model.__del__()
                    if(min_acc >= expact_source_acc and max_acc >= expect_ASR):
                        logger.info(f"expected model found, training exiting.")
                        break

        
        if(addDefense_valSet_test):
            valSet_allData_test('poison',dataloader_test, add_defense = True) # CDA
            valSet_poisonData_test('poison', dataloader_test_addingPatch, attack = 0, add_defense = True) # Source Acc
            valSet_poisonData_test('poison', dataloader_test_addingPatch, attack = 1, add_defense = True) # ASR
        if(final_valSet_test):
            valSet_allData_test('poison', dataloader_test) # CDA
        if(final_test):
            addMeasure_valSet_poisonData_test('poison', dataloader_test_addingPatch, attack = 0) # Source Acc
            addMeasure_valSet_poisonData_test('poison', dataloader_test_addingPatch, attack = 1) # ASR
    return
    
if __name__ == '__main__':
	main()