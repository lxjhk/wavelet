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
from multiprocessing import shared_memory
import threading
import remote_pdb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
use_cuda = torch.cuda.is_available()

# Random seed
# make it deterministic - for res comp
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
best_acc = 0  # best test accuracy

import threading
import time

gpu0 = 0
gpu1 = 1
world_size = 2

def main():
    global best_acc
    mp.set_start_method('spawn')

    # create a lock
    e = mp.Event()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainset_small, trainset_large = trainset, trainset
    # torch.utils.data.random_split(trainset, [len(trainset)//2, len(trainset) - (len(trainset)//2)])
    print("len trainset_large: ", len(trainset_large))
    print("len trainset_small: ", len(trainset_small))
    trainloader_gpu0 = data.DataLoader(trainset_small, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    trainloader_gpu1 = data.DataLoader(trainset_large, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()

    # # Resume
    title = 'cifar-10-' + args.arch
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # clone models
    save_model(model, 0)
    save_model(model, 1)
    model0 = load_model(0)
    model1 = load_model(1)

    train_gpu1 = mp.spawn(train_copy, args=(trainloader_gpu1, model1, criterion, use_cuda, gpu1, e, args, testloader), join=False)
    train_gpu0 = mp.spawn(train, args=(trainloader_gpu0, model0, criterion, use_cuda, gpu0, e, args, testloader), join=False)
    
    train_gpu1.join()
    train_gpu0.join()

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(proc_ind,trainloader, model, criterion, use_cuda, device, e, args, testloader):
    dist.init_process_group(backend='nccl', init_method="file:///home/ubuntu/vgg/p2p/test_grad", world_size=2, rank=0)
    cuda_p2p.operatingGPUs(gpu0, gpu1)
    cuda_p2p.enablePeerAccess()
    print("\nTraining on device "+str(device)+" begins")
    device_name = "cuda:"+str(device)
    model.cuda(device_name)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # switch to train mode
        model.train()
        batch_time, data_time, losses, top1, top5, end = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()
        bar = Bar('Processing', max=len(trainloader))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            cuda_p2p.cudaSync()
            # measure data loading time
            data_time.update(time.time() - end)
            if use_cuda:
                inputs_remote = inputs[:len(inputs)//2,:,:,:].cuda("cuda:" + str(gpu1))     
                inputs_local = inputs[len(inputs)//2:,:,:,:].cuda(device_name)
                targets = targets.cuda(device_name, non_blocking=True) # add data in, no wait on true
            cuda_p2p.cudaSync()

            #???
            outputs_remote_f, outputs_remote_res = work_warpper(lambda: model(inputs_remote)) #o utput is on GPU 0? computation 是不是在 0 上？ 其实因该在 1 上
            outputs_local_f, outputs_local_res = work_warpper(lambda: model(inputs_local))

            from threading import Thread
            t1, t2 = Thread(target=outputs_remote_f), Thread(target=outputs_local_f)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            outputs_remote = outputs_remote_res()
            outputs_local = outputs_local_res()

            outputs = torch.cat((outputs_remote, outputs_local),dim=0)
            loss = criterion(outputs, targets)  # GPU 0 evaluate

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            cuda_p2p.cudaSync()
            optimizer.zero_grad()

            ## Learning other model grad
            if batch_idx >= 2:
                #print("\nGPU0 receive new Grad at batch ", batch_idx)
                average_grad(model) # sync?????
                #print("\nGPU0 receive new Grad at batch ", batch_idx, model.classifier.weight.grad)
            
            #print("\nBefore backward batch_idx: ", batch_idx, model.classifier.weight.grad)
            cuda_p2p.cudaSync()
            loss.backward()
            #print("\nAfter backward batch_idx: ", batch_idx, model.classifier.weight.grad)
            if batch_idx >= 2:
                for param in model.parameters():
                    param.grad.data /= 2
            cuda_p2p.cudaSync()
            #print("\nAfter /2 batch_idx: ", batch_idx, model.classifier.weight.grad)

            # Run 1 loop first to warm up, then handshake for synced start
            if batch_idx == 1:
                e.set()
                print("GPU0 signal")
                while(e.is_set()):
                    pass
                print("GPU0 receive")

            # Sending other model grad
            if batch_idx >= 1:
                #print("\nGPU0 sending new Grad at batch ", batch_idx, model.classifier.weight.grad)
                average_grad(model)

            cuda_p2p.cudaSync()
            optimizer.step()
            cuda_p2p.cudaSync()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(trainloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

        # validation
        test_loss, test_acc = test(testloader, model, criterion, args.start_epoch, use_cuda, device)
        print("model 0 loss, acc:", test_loss, test_acc)

    save_model(model, 0)
    print("\nTraining on device "+str(device)+" ends")


def train_copy(proc_ind, trainloader, model, criterion, use_cuda, device, e, args, testloader):
    dist.init_process_group(backend='nccl', init_method="file:///home/ubuntu/vgg/p2p/test_grad", world_size=2, rank=1)
    cuda_p2p.operatingGPUs(gpu0, gpu1)
    cuda_p2p.enablePeerAccess()
    print("\nTraining on device "+str(device)+" begins")
    device_name = "cuda:"+str(device)
    model.cuda(device_name)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # switch to train mode
        model.train()
        batch_time, data_time, losses, top1, top5, end = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()
        bar = Bar('Processing', max=len(trainloader))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx == 1:
                print("\nGPU1 waiting for signal")
                e.wait()
                print("\nGPU1 caught signal, sending receival signal")
                e.clear()
            cuda_p2p.cudaSync()
            # measure data loading time
            data_time.update(time.time() - end)
            if use_cuda:
                inputs_remote = inputs[:len(inputs)//2,:,:,:].cuda("cuda:" + str(gpu0))     
                inputs_local = inputs[len(inputs)//2:,:,:,:].cuda(device_name)
                targets = targets.cuda(device_name, non_blocking=True)
            # compute output
            cuda_p2p.cudaSync()
            outputs_remote_f, outputs_remote_res = work_warpper(lambda: model(inputs_remote))
            outputs_local_f, outputs_local_res = work_warpper(lambda: model(inputs_local))
            from threading import Thread
            t1, t2 = Thread(target=outputs_remote_f), Thread(target=outputs_local_f)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            outputs_remote = outputs_remote_res()
            outputs_local = outputs_local_res()

            outputs = torch.cat((outputs_remote, outputs_local),dim=0)  ###outputs are always saved in the model gpu
            loss = criterion(outputs, targets) # update loss
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # compute gradient and do SGD step
            cuda_p2p.cudaSync()
            optimizer.zero_grad()
            if batch_idx >= 1:
                average_grad(model) # sync 两边一定是 sync 的 等待???
                #print("\nGPU1 receive new Grad at batch ", batch_idx, model.classifier.weight.grad)
            cuda_p2p.cudaSync()
            ## adding new grad before
            cuda_p2p.cudaSync()
            loss.backward() # (targets - input)
            #print("GPU1 after backward grad batch_idx:", batch_idx, model.classifier.weight.grad)
            cuda_p2p.cudaSync()

            if batch_idx >= 1:
                for param in model.parameters():
                    param.grad.data /= 2
            if batch_idx >= 1:
                #print("\nGPU1 sending new Grad at batch ", batch_idx, model.classifier.weight.grad)
                average_grad(model) # 对面 delta 是 0
                # 402-404

            cuda_p2p.cudaSync()
            optimizer.step() # gradient applies to my weight
            cuda_p2p.cudaSync()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(trainloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
        
        # validation
        test_loss, test_acc = test(testloader, model, criterion, args.start_epoch, use_cuda, device)
        print("model 1 loss, acc:", test_loss, test_acc)

    save_model(model, 1)
    print("\nTraining on device "+str(device)+" ends")

def test(testloader, model, criterion, epoch, use_cuda, device):
    global best_acc
    device = "cuda:"+str(device)
    model.cuda(device)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                     batch=batch_idx + 1,
                     size=len(testloader),
                     data=data_time.avg,
                     bt=batch_time.avg,
                     total=bar.elapsed_td,
                     eta=bar.eta_td,
                     loss=losses.avg,
                     top1=top1.avg,
                     top5=top5.avg,
                     )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def save_model(model, device):
    torch.save(model, "./model_" + str(device))

def load_model(device):
    return torch.load("./model_" + str(device))

def share_param(model):
    for param in model.parameters():
        param.share_memory_()

def cudaSync():
    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

def average_grad(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
        #param.grad.data /= 4

def work_warpper(k):
    """ Wraps a task into a callable object for multi-threading tasks with returned values """
    result = None 
    def work():
        nonlocal result
        result = k()
    def res():
        return result
    return work, res

if __name__ == '__main__':
    main()