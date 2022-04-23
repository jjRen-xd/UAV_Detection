# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/config.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2022/04/01
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                              --> 电磁扰动识别分类训练程序 <--        
                    -- 参数配置文件
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    None
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/04/01 |          creat
# ------------------------------------------------------------------------
'''

class DefaultConfigs_UAV(object):
    ''' 默认参数配置 '''
    # Dataset
    dataset_name = "UAV_0323"                      
    num_classes = 2                             # 分类类别数
    signal_len = "50176, 8"                     # --> (8, 224, 224)
    TRAIN_PATH = "/media/hp3090/HDD-2T/renjunjie/UAV_Detection/dataset/UAV_0323/train_set.txt"
    TEST_PATH = "/media/hp3090/HDD-2T/renjunjie/UAV_Detection/dataset/UAV_0324/test_set.txt"

    process_IQ = True                           # 是否在载入数据时对IQ两路进行预处理

    batch_size = 32                              # DataLoader中batch大小，550/110=5 Iter
    num_workers = 16                             # DataLoader中的多线程数量

    # model
    model = "ResNet_50_SE"                          # 指定模型，ResNet50,ResNet101,ResNet152,ResNet_50_SE
    resume = False                               # 是否加载训练好的模型
    checkpoint_name = '50%trainUAV_0323_100%test—UAV_0324_acc:ResNet_50_SE.pth'   # 训练完成的模型名

    # train
    num_epochs = 10                              # 训练轮数
    lr = 0.01                                   # 初始lr
    valid_freq = 1                              # 每几个epoch验证一次
    
    # log
    iter_smooth = 1                             # 打印 & 记录log的频率,每几个batch打印一次准确率

    # seed = 1000                               # 固定随机种子


cfgs = DefaultConfigs_UAV()
