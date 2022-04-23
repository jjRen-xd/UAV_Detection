# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/train.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2022/04/01
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> 电磁扰动识别分类训练主程序 <--        
                    -- TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> train():
                        -- 训练主程序,包含了学习率调整、log记录、收敛曲线绘制
                        ,每训练n(1)轮验证一次，保留验证集上性能最好的模型
                    <1> valid():
                        -- 验证当前训练模型在测试集中的性能
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/04/01 |          creat
--------------------------------------------------------------------------
'''

import os
import time

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from configs import cfgs
from dataset.UAV2022 import UAVDataset
from networks.resnet import ResNet50,ResNet101,ResNet152
from networks.resnet_se import ResNet_50_SE
from utils.strategy import step_lr, accuracy
from utils.plot import draw_curve
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision import models

def train():
   ''' 信号分类训练主程序 '''

   # model
   _model = eval(f"{cfgs.model}")     
   model = _model(num_classes=cfgs.num_classes)
   print(model)

   # model = models.densenet201(pretrained=False)#这一句表示加载densnet169在imagnet数据集上的预训练模型，True表示不用重新下载，false会自动下载模型（需要翻墙）
   # num_ftrs = model.classifier.in_features
   # model.classifier = nn.Linear(num_ftrs, 3)
   # model.features.conv0 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
   # print(model)


   model.to(device, non_blocking=True)

   # Dataset
   transform = transforms.Compose([ 
      # transforms.ToTensor()
      # waiting add
   ])

   # Train data
   train_dataset = UAVDataset(cfgs.TRAIN_PATH, wavORnpy=0, reshape_size=(8,224,224), processIQ=cfgs.process_IQ, transform=transform)
   train_loader = DataLoader( 
      train_dataset, \
      batch_size = cfgs.batch_size, \
      num_workers = cfgs.num_workers, \
      shuffle = True, \
      pin_memory=True, \
      drop_last = False
   )
   # Valid data
   valid_dataset = UAVDataset(cfgs.TEST_PATH, wavORnpy=0, reshape_size=(8,224,224), processIQ=cfgs.process_IQ, transform=transform)
   valid_loader = DataLoader( 
      valid_dataset, \
      batch_size = cfgs.batch_size, \
      num_workers = cfgs.num_workers, \
      shuffle = True, \
      pin_memory=True, \
      drop_last = False
   )

   # log
   if not os.path.exists('./log'):
      os.makedirs('./log')
   log = open(f'./log/log_{time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime(time.time()))}.txt', 'w')
   log.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'-'*30+'\n')
   log.write(
      'model:{}\ndataset_name:{}\nnum_classes:{}\nbatch_size:{}\nnum_worker:{}\nprocess_IQ:{}\nnum_epoch:{}\nlearning_rate:{}\nsignal_len:{}\niter_smooth:{}\ncheckpoint_name:{}\n'.format(
            cfgs.model, 
            cfgs.dataset_name, 
            cfgs.num_classes, 
            cfgs.batch_size, 
            cfgs.num_workers, 
            cfgs.process_IQ, 
            cfgs.num_epochs,
            cfgs.lr, 
            cfgs.signal_len, 
            cfgs.iter_smooth,
            cfgs.checkpoint_name
         )
   )

   # load checkpoint
   if cfgs.resume:
      model = torch.load(os.path.join('./checkpoints', cfgs.checkpoint_name))

   # lossW
   criterion = nn.CrossEntropyLoss().to(device, non_blocking=True)  # 交叉熵损失

   # train 
   sum = 0
   train_loss_sum = 0
   train_top1_sum = 0
   max_val_acc = 0
   train_draw_acc = []
   val_draw_acc = []
   lr = cfgs.lr
   for epoch in range(cfgs.num_epochs):
      ep_start = time.time()

      # 学习率调整
      lr = step_lr(epoch, lr)

      # 优化器
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

      model.train()
      top1_sum = 0
      for i, (signal, label) in enumerate(train_loader):
         # input = Variable(signal).cuda().float()
         # target = Variable(label).cuda().long()
         input = signal.to(device, non_blocking=True).float()
         target = label.to(device, non_blocking=True).long()
         

         output = model(input)            # inference
         
         loss = criterion(output, target) # 计算交叉熵损失
         optimizer.zero_grad()
         loss.backward()                  # 反传
         optimizer.step()

         top1 = accuracy(output.data, target.data, topk=(1,))  # 计算top1分类准确率
         train_loss_sum += loss.data.cpu().numpy()
         train_top1_sum += top1[0]
         sum += 1
         top1_sum += top1[0]

         if (i+1) % cfgs.iter_smooth == 0:
            print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                  %(epoch+1, cfgs.num_epochs, i+1, len(train_dataset)//cfgs.batch_size, 
                  lr, train_loss_sum/sum, train_top1_sum/sum))
            log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f\n'
                        %(epoch+1, cfgs.num_epochs, i+1, len(train_dataset)//cfgs.batch_size, 
                        lr, train_loss_sum/sum, train_top1_sum/sum))
            sum = 0
            train_loss_sum = 0
            train_top1_sum = 0

      train_draw_acc.append(top1_sum/len(train_loader))
      
      epoch_time = (time.time() - ep_start) / 60.
      if epoch % cfgs.valid_freq == 0 and epoch < cfgs.num_epochs:
         # eval
         val_time_start = time.time()
         val_loss, val_top1 = valid(model, valid_loader, criterion)
         val_draw_acc.append(val_top1)
         val_time = (time.time() - val_time_start) / 60.

         print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f'
               %(epoch+1, cfgs.num_epochs, val_loss, val_top1, val_time*60, max_val_acc))
         print('epoch time: {}s'.format(epoch_time*60))
         if val_top1[0].data > max_val_acc:
            max_val_acc = val_top1[0].data
            print('Taking snapshot...')
            if not os.path.exists('./checkpoints'):
               os.makedirs('./checkpoints')
            torch.save(model, '{}/{}'.format('checkpoints', cfgs.checkpoint_name))

         log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f\n'
                  %(epoch+1, cfgs.num_epochs, val_loss, val_top1, val_time*60, max_val_acc))
   draw_curve(train_draw_acc, val_draw_acc)
   log.write('-'*40+"End of Train"+'-'*40+'\n')
   log.close()


# valid
def valid(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = ims.to(device, non_blocking=True).float()
        target_val = label.to(device, non_blocking=True).long()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
        print(top1_val[0])
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1


def test():
   ''' 信号测试主程序 '''
   # model
   _model = eval(f"{cfgs.model}")     
   model = _model(num_classes=cfgs.num_classes)
   print(model)

   model.to(device, non_blocking=True)

   # Dataset
   transform = transforms.Compose([ 
      # transforms.ToTensor()
      # waiting add
   ])

   # test data
   valid_dataset = UAVDataset(cfgs.TEST_PATH, wavORnpy=0, reshape_size=(8,224,224), processIQ=cfgs.process_IQ, transform=transform)
   valid_loader = DataLoader( 
      valid_dataset, \
      batch_size = cfgs.batch_size, \
      num_workers = cfgs.num_workers, \
      shuffle = True, \
      pin_memory=True, \
      drop_last = False
   )

   # log
   if not os.path.exists('./log'):
      os.makedirs('./log')
   log = open(f'./log/log_{time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime(time.time()))}.txt', 'w')
   log.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'-'*30+'\n')
   log.write(
      'model:{}\ndataset_name:{}\nnum_classes:{}\nbatch_size:{}\nnum_worker:{}\nprocess_IQ:{}\nnum_epoch:{}\nlearning_rate:{}\nsignal_len:{}\niter_smooth:{}\ncheckpoint_name:{}\n'.format(
            cfgs.model, 
            cfgs.dataset_name, 
            cfgs.num_classes, 
            cfgs.batch_size, 
            cfgs.num_workers, 
            cfgs.process_IQ, 
            cfgs.num_epochs,
            cfgs.lr, 
            cfgs.signal_len, 
            cfgs.iter_smooth,
            cfgs.checkpoint_name
         )
   )

   # lossW
   criterion = nn.CrossEntropyLoss().to(device, non_blocking=True)  # 交叉熵损失

   # load checkpoint
   if cfgs.resume:
      model = torch.load(os.path.join('./checkpoints', cfgs.checkpoint_name))

   # eval
   val_time_start = time.time()
   val_loss, val_top1 = valid(model, valid_loader, criterion)
   val_time = (time.time() - val_time_start) / 60.

   print('Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s\n'
         %(val_loss, val_top1, val_time*60))

   log.write('Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s\n'
            %(val_loss, val_top1, val_time*60))
   log.write('-'*40+"End of test"+'-'*40+'\n')
   log.close()


if __name__ == "__main__":
   train()
   # test()