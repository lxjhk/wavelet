# import torch
# import torch.multiprocessing as mp
# manager = mp.Manager()
# Global = manager.Namespace()
# Global.test_boolean = False

# def change(proc_ind):
#     print("change is called")
#     Global.test_boolean = True
#     print(Global.test_boolean)

# def main():
#     print("running mean")
#     a = mp.spawn(change, args=(), join=False)
#     print("1")
#     a.join()
#     print("2")

# if __name__ == "__main__":
#     main()
#     print("finished mean")
#     print(Global)



'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import torch.distributed as dist
from tqdm import tqdm
import subprocess
import xml.etree.ElementTree as ET
import pdb
import cuda_p2p
import torch.cuda as cutorch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import multiprocessing
import torch.multiprocessing as mp
import threading
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"

gpu0 = 2
gpu1 = 3

class Dummy(nn.Module):
    def forward(self, x):


Queue()
model.parameters()
def main():
    train_gpu1 = mp.spawn(train_copy, join=False)
    train_gpu0 = mp.spawn(train, join=False)
    train_gpu0.join()
    train_gpu1.join()

def train(i):
    device = gpu0
    dist.init_process_group(backend="nccl", world_size=2, rank=1, init_method="file:///home/ubuntu/test_variable")
    cuda_p2p.operatingGPUs(gpu0, gpu1)
    cuda_p2p.enablePeerAccess()
    device = "cuda:"+str(device)
    temp = torch.tensor(np.ones((100, 100))).cuda(device)
    temp2 = torch.tensor(np.ones((10, 10))).cuda(device) * 7
    print("train all_reduce")
    dist.all_reduce(temp)
    dist.all_reduce(temp2)
    print("train: ", temp)
    print("train: ", temp2)

def train_copy(i):
    device = gpu1
    dist.init_process_group(backend="nccl", world_size=2, rank=0, init_method="file:///home/ubuntu/test_variable")
    cuda_p2p.operatingGPUs(gpu0, gpu1)
    cuda_p2p.enablePeerAccess()
    device = "cuda:"+str(device)
    temp = torch.tensor(np.ones((100, 100))).cuda(device)
    temp2 = torch.tensor(np.ones((10, 10))).cuda(device) * 7
    print("train_copy all_reduce")
    print("train_copy: ", temp)
    print("train_copy: ", temp2)

""" Gradient averaging. """
def average_grad(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
        param.grad.data /= size

if __name__ == '__main__':
    main()


