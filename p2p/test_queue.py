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

gpu0 = 0
gpu1 = 1


def main():
    mp.set_start_method('spawn')
    queue01 = mp.Queue()
    queue10 = mp.Queue()
    train_gpu1 = mp.spawn(train_copy, args = (queue01, queue10), join=False)
    train_gpu0 = mp.spawn(train, args = (queue10, queue01), join=False)
    train_gpu0.join()
    train_gpu1.join()
    queue01.close()
    queue10.close()
    queue01.join_thread()
    queue10.join_thread()

def train(proc_ind, receiveQueue, sendQueue):
    cuda_p2p.operatingGPUs(gpu0, gpu1)
    cuda_p2p.enablePeerAccess()
    for i in range(20):
        a = torch.ones([5 + i, 35 + i * 3, 35 + i * 2]).float().cuda(gpu0) * 10
        received = receiveQueue.get()
        print((received == a).all())
        del received
        sendQueue.put(a)
    receiveQueue.close()
    sendQueue.close()

def train_copy(proc_ind, receiveQueue, sendQueue):
    cuda_p2p.operatingGPUs(gpu0, gpu1)
    cuda_p2p.enablePeerAccess()
    for i in range(20):
        
        b = torch.ones([5 + i, 35 + i * 3, 35 + i * 2]).float().cuda(gpu1) * 10
        sendQueue.put(b)
        received = receiveQueue.get()
        print((received == b).all())
        del received
    
    receiveQueue.close()
    sendQueue.close()

""" Gradient averaging. """
def average_grad(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
        param.grad.data /= size

if __name__ == '__main__':
    main()

