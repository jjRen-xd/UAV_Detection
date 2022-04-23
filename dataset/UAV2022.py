# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/dataset/UAV2022.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2022/04/01
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> UAV电磁扰动识别代码 (PyTorch) <--        
                    -- 数据集data2023处理载入程序
                    -- 要有一个train.txt/test.txt
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> None
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> read_txt(): 
                        -- 从.txt中读取训练,测试样本的标签,路径,采样点信息
                    <1> get_wav_params():
                        -- 获取.wav文件的基本参数信息
                    <1> divide_dataset():
                        -- 划分数据集,生成划分好的.txt
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> UAVDataset(Dataset): 
                        -- 定义UAVDataset类,继承Dataset方法,并重写
                        __getitem__()和__len__()方法
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/04/01 | 完成初步数据载入功能
--------------------------------------------------------------------------
'''

import os
from re import S
import sys
import random
import wave

import librosa
import torch
import numpy as np
import soundfile as sf

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

sys.path.append("../")
from dataset.classes import labelName
# from classes import labelName
import scipy as sp
import scipy.signal

from matplotlib import pyplot as plt
from nnAudio.Spectrogram import CQT1992v2

SAVE_NUM = 0
cqt_transform=CQT1992v2(sr=15120000, fmin=10000, fmax=5000000, n_bins=32, hop_length=224)

class UAVDataset(Dataset):
    ''' 定义RMLDataset类,继承Dataset方法,并重写__getitem__()和__len__()方法 '''
    def __init__(self, txt_path, wavORnpy = 0, reshape_size = None, processIQ = True , transform=None):
        ''' 初始化函数,得到数据 '''
        # 标签,路径,采样点起始位置,信号采样长度
        self.labels, self.sigs_path, self.start_poses, self.sigs_len = read_txt(txt_path)
        self.transform = transform
        self.size = reshape_size        # 自定义输出大小,(a,b,...)
        self.wavORnpy = wavORnpy        # 0:数据为wav 1:数据为npy
        self.processIQ = processIQ

    def __getitem__(self, index):
        ''' index是根据batchsize划分数据后得到的索引,最后将data和对应的labels进行一起返回 '''
        label = self.labels[index]
        sig_path = self.sigs_path[index]
        start_pos = self.start_poses[index]
        sig_len = self.sigs_len[index]
        if self.wavORnpy == 0:
            sig, sameple_rate = sf.read(sig_path, frames = sig_len, start = start_pos)
            # 低通滤波
            # sig = reduce_noise_butter(sig)   
            # 归一化
            if self.processIQ:
                sig = processIQ(sig)
            # CQT变换
            sig = CQT_transform(cqt_transform, sig.transpose(1, 0), label, savefig = False)
            # 尺寸调整
        else:
            sig = np.load(sig_path, allow_pickle=True)

        if self.size is not None:
            # sig = sig.reshape(self.size)
            sig = np.resize(sig, self.size)
        if self.transform is not None:
            sig = self.transform(sig)
        return sig, label
  
    def __len__(self):
        ''' 该函数返回数据大小长度,目的是DataLoader方便划分,如果不知道大小,DataLoader会一脸懵逼 '''
        return len(self.labels)


def read_txt(path):
    labels, sigs_path, start_poses, sigs_len = [], [], [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            label, sig_path, start_pos, sig_len  = line.strip().split(',')

            labels.append(int(label))
            sigs_path.append(sig_path)
            start_poses.append(int(start_pos))
            sigs_len.append(int(sig_len))
            
    return labels, sigs_path, start_poses, sigs_len


def get_wav_params(wav_path):
    """ 
    Funcs:
        快速获取.wav文件的基本信息
        soundfile() 读写wav文件
    Args:
        <0> wav_path: str
    Returns:
        <0> n_frames: int, 采样点数
        <1> n_channels: int, 通道数
        <2> frame_rate: int, 采样率
        <3> wav_length: int, 信号持续时间
    """
    with sf.SoundFile(wav_path, 'r+') as f_wav:
        n_frames = f_wav.frames
        n_channels = f_wav.channels
        frame_rate = f_wav.samplerate
        f_wav.close()
        wav_length = n_frames / float(frame_rate)    # 音频长度: x秒
    return n_frames, n_channels, frame_rate, wav_length    


def processIQ(x):
    ''' 对8路信号分别进行预处理,结合两路为复数,除以标准差,再分离实部虚部到两路 '''
    for i in range(4):
        sample_complex = x[:, 2*i] + x[:, 2*i+1] * 1j
        # sample_complex = sample_complex / np.std(sample_complex)
        sample_complex -= np.min(sample_complex)
        sample_complex /= np.max(sample_complex)
        x[:, 2*i] = sample_complex.real
        x[:, 2*i+1] = sample_complex.imag
    return x


def reduce_noise_median(y):  
    '''使用中值滤波降噪'''
    y = scipy.signal.medfilt(y,5)
    return (y)


def reduce_noise_butter(y):
    '''使用低通滤波器降噪'''
    b, a = scipy.signal.butter(8, 0.5291, 'lowpass')   # （滤波器阶数, Wn= 2*截止频率/采样频率 ）   采样频率：15120000
    y = scipy.signal.filtfilt(b, a, y, padlen=-1)
    return (y)


def CQT_transform(transform, waves, label, savefig = True):
    waves = torch.from_numpy(waves).float()
    image = transform(waves)             # torch.Size([8, 108, 197])
    if savefig:
        plt.imshow(image[0])
        plt.title(label)
        plt.savefig(f'../figs/CQT_fig/{label}_{SAVE_NUM}_CQT.png')
        SAVE_NUM += 1
    return image



def divide_dataset(dataset_name, dataset_path, save_path, save_npy = False, crop_len = 100000, trainset_ratio = 0.5, testset_ratio = 1):
    """ 
    Funcs:
        根据.wav文件路径与长度,划分数据集
    Args:
        <0> dataset_name: str, 此次划分的文件夹名称
        <1> dataset_path: str, 数据集路径
            example: "/media/hp3090/HDD-2T/yrwang/UAV_detect/data0323/data/"
        <2> save_path: str, .npy保存的路径
        <3> save_npy: bool, 是否保存数据至npy,并生成.npy的训练测试索引至.txt,否则只保存.wav的索引至.txt
        <4> crop_len: int, 信号裁剪长度, 最后一段不足crop_len的丢弃
        <5> trainset_ratio: float, 训练数据占总数据的比例
        <6> testset_ratio: float, 除去训练数据外,测试数据占剩余数据的比例
    Returns:
    """
    # data0323: 共64个.wav, 'N':26个,'D':28个,'Y':10个, 每个.wav 268288000个采样点
    save_dir = (f"{save_path}/{dataset_name}/data")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    train_txt = open(f'{save_path}/{dataset_name}/train_set.txt', 'w')
    test_txt = open(f'{save_path}/{dataset_name}/test_set.txt', 'w')

    for dir in os.listdir(dataset_path):
        # 遍历类别文件夹
        if dir in labelName:
            num_wav = 0
            # 只选中指定类别的文件夹
            label = np.where(np.array(labelName) == dir)[0][0]              # 该文件夹下的类别
            dir_path = os.path.join(dataset_path, dir)
            for file in os.listdir(dir_path):
                # 遍历该类别下的.wav文件
                if os.path.splitext(file)[1] == '.wav':
                    wav_path = os.path.join(dir_path, file)                 # .wav文件路径
                    # print(wav_path)
                    n_frames, n_channels, _, _ = get_wav_params(wav_path)   # 采样点数, 通道数: 268288000, 8
                    # print(n_frames, n_channels)
                    crop_num = n_frames // crop_len                         # 一个.wav裁剪后的样本数量
                    # print(crop_num)

                    # 按一定比例划分数据集
                    train_idx = random.sample(range(crop_num), int(crop_num*trainset_ratio))
                    test_idx = list(set(range(crop_num)).difference(set(train_idx)))
                    test_idx = random.sample(test_idx, int(len(test_idx)*testset_ratio))

                    # 保存至.txt
                    for idx in train_idx:
                        if not save_npy:
                            # 只保存.wav的索引
                            train_txt.write(f"{label},{wav_path},{idx*crop_len},{crop_len}\n")
                        else:
                            # 另一种保存方式，将数据另存为.npy,索引也换为.npy
                            save_npy_path = f"{save_dir}/{dir}-{num_wav}_{idx*crop_len}-{(idx+1)*crop_len}.npy"
                            train_txt.write(f"{label},{save_npy_path},{idx*crop_len},{crop_len}\n")
                            sig, _ = sf.read(wav_path, frames = crop_len, start = idx*crop_len)
                            # 归一化
                            sig = processIQ(sig)
                            # CQT变换
                            sig = CQT_transform(cqt_transform, sig.transpose(1, 0), label, savefig = False)
                            np.save(save_npy_path, sig)
                    for idx in test_idx:
                        if not save_npy:
                            # 只保存.wav的索引
                            test_txt.write(f"{label},{wav_path},{idx*crop_len},{crop_len}\n")
                        else:
                            # 另一种保存方式，将数据另存为.npy,索引也换为.npy
                            save_npy_path = f"{save_dir}/{dir}-{num_wav}_{idx*crop_len}-{(idx+1)*crop_len}.npy"
                            test_txt.write(f"{label},{save_npy_path},{idx*crop_len},{crop_len}\n")
                            # sig, _ = sf.read(wav_path, frames = crop_len, start = idx*crop_len)
                            # # 归一化
                            # sig = processIQ(sig)
                            # # CQT变换
                            # sig = CQT_transform(cqt_transform, sig.transpose(1, 0), label, savefig = False)
                            # np.save(save_npy_path, sig)
                    num_wav += 1


def wavDataPlot():
    ''' 绘制.wav数据集中的信号图像 '''
    import matplotlib.pyplot as plt
    dataset_path = "/media/hp3090/HDD-2T/yrwang/UAV_detect/data0323/data/"
    label = 'D'
    length = 100
    start_pos = 20000

    if not os.path.isdir(f"../figs/{label}"):
        os.makedirs(f"../figs/{label}")

    dir_path = os.path.join(dataset_path, label)
    for file in os.listdir(dir_path):
        if os.path.splitext(file)[1] == '.wav':
            wav_path = os.path.join(dir_path, file)
            sig, _ = sf.read(wav_path, frames=length, start=start_pos, dtype='int16')
            #plt.figure(figsize=(40, 40), dpi=500)
            # 8个通道分别绘制
            for channel in range(8):
                plt.title(f'{channel+1} channel')
                plt.subplot(8, 1, channel+1)
                plt.plot(sig[:, channel])
            plt.show()
            # plt.savefig(f'../figs/{label}/{file}-{length}-{start_pos}-sig_plot.jpg', dpi=500)
            plt.close()


if __name__ == "__main__":
    ''' 测试UVA2022.py,测试dataLoader是否正常读取、处理数据 '''
    # 数据集信号图像绘制
    # wavDataPlot()
    
    # 数据集的划分
    divide_dataset(
        "UAV_0323", 
        dataset_path="/media/hp3090/HDD-2T/yrwang/UAV_detect/data0323/data", 
        save_path="/media/hp3090/421_2T/jjren/UAV_DATASET", 
        save_npy=True,
        crop_len=224*224, 
        trainset_ratio=0.5, 
        testset_ratio=1
    )
    
    # 数据集测试
    
    # TRAIN_PATH = "/media/hp3090/HDD-2T/renjunjie/UAV_Detection/dataset/UAV_0323/train_set.txt"
    # BATCH_SIZE = 4
    # NUM_WORKER = 4

    # transform = transforms.Compose([ 
    #     # transforms.ToTensor()
    #     # waiting add
    # ])
    # # 通过RMLDataset将数据进行加载,返回Dataset对象,包含data和labels
    # train_dataset = UAVDataset(TRAIN_PATH, wavORnpy=0, reshape_size=(8,224,224), transform=transform)
    # # 通过DataLoader读取数据
    # train_loader = DataLoader( 
    #     train_dataset, \
    #     batch_size = BATCH_SIZE, \
    #     num_workers = NUM_WORKER, \
    #     shuffle = True, \
    #     drop_last = False
    # )
    # # for i, data in enumerate(train_loader):
    # #     # i表示第几个batch, data表示该batch对应的数据,包含data和对应的labels
    # #     print("第 {} 个Batch \n".format(i))
    # #     print("Size:", data[0].shape, data[1])
    # from tqdm import tqdm
    # for datas,targets in tqdm(train_loader):
    #     for data,target in zip(datas,targets):
    #         print(data.shape,target)
    
