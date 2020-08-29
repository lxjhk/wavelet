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
from tqdm import tqdm
import subprocess
import xml.etree.ElementTree as ET
import pdb
from pynvml import *
# import cuda_p2p

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


def events_hook(net):
    childrens = list(net.children())
    if not childrens:
        if isinstance(net, torch.nn.Conv2d):
            net.register_forward_hook(print_events_label_hook_forward)
            net.register_backward_hook(print_events_label_hook_backward)
        if isinstance(net, torch.nn.Linear):
            net.register_forward_hook(print_events_label_hook_forward)
            net.register_backward_hook(print_events_label_hook_backward)
        if isinstance(net, torch.nn.BatchNorm2d):
            net.register_forward_hook(print_events_label_hook_forward)
            net.register_backward_hook(print_events_label_hook_backward)
        if isinstance(net, torch.nn.ReLU):
            net.register_forward_hook(print_events_label_hook_forward)
            net.register_backward_hook(print_events_label_hook_backward)
        if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
            net.register_forward_hook(print_events_label_hook_forward)
            net.register_backward_hook(print_events_label_hook_backward)
        if isinstance(net, torch.nn.Upsample):
            net.register_forward_hook(print_events_label_hook_forward)
            net.register_backward_hook(print_events_label_hook_backward)
        return
    for c in childrens:
        events_hook(c)

def print_events_label_hook_forward(self, input, output):
    f_events.write(self.__class__.__name__ + "_forward"+ "|" + str(get_time()) + "|" + str(input[0].shape) + "\n")

def print_events_label_hook_backward(self, input, output):
    f_events.write(self.__class__.__name__ + "_backward" "|" + str(get_time())+ "|" + str(input[0].shape) + "\n")


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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_cuda = torch.cuda.is_available()

# Random seed
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
MB = 1024*1024

f_events = open("events.txt", "w")
f_mem = open("mem.txt", "w")


def mem_print():
    def gpu_print():
        t = get_time()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_rate = nvmlDeviceGetUtilizationRates(handle)
        f_mem.write("gpu_usage|{}|{}\n".format(t, util_rate.gpu))
        f_mem.write("mem_free|{}|{}\n".format(t, mem_info.free))
        f_mem.write("mem_used|{}|{}\n".format(t, mem_info.used))

    while True:
        t = get_time()
        # emem recording
        stats = torch.cuda.memory_stats()
        for k in stats:
            f_mem.write("mem|{}|{}|{}\n".format(t, k, stats[k]))
            # print("mem", "|", t, "|", k, "|", stats[k])
        gpu_print()
        time.sleep(500 / 1000000)

def print_events_mark(event):
    t = get_time()
    f_events.write("mem|{}|{}\n".format(t, event))

def get_time():
    t = int(time.time() * (10 **6))
    return t


def main():
    nvmlInit()
    x = threading.Thread(target=mem_print, args=())
    x.start()

    global best_acc
    # cuda_p2p.operatingGPUs(2, 3)
    # cuda_p2p.enablePeerAccess()
    #a = torch.tensor(3).int().cuda('cuda:0')
    #b = torch.tensor(4).int().cuda('cuda:1')
    #cuda_p2p.add_test(a, b)
    #print(a)

    #print(b)
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

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
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

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

    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda("cuda:0")

    events_hook(model)

    print("Initialized linear weight: ", list(model.classifier.weight)[0][:10])
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        print('Best acc:')
        print(best_acc)
        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'acc': test_acc,
        #         'best_acc': best_acc,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, checkpoint=args.checkpoint)


    # e.set()
    #e2.set()
    # th.join()
    #th2.join()

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # .cudaSync()cuda_p2p
        # measure data loading time
        data_time.update(time.time() - end)
        print_events_mark("start")
        if use_cuda:
            if batch_idx < len(trainloader) // 2:
                inputs, targets = inputs.cuda("cuda:0"), targets.cuda("cuda:0", non_blocking=True)
            else:
                inputs, targets = inputs.cuda("cuda:0"), targets.cuda("cuda:0", non_blocking=True)
        print_events_mark("data_loading")

        # compute output
        # cuda_p2p.cudaSync()
        with torch.autograd.profiler.profile(profile_memory=True, record_shapes=True) as prof:
       	    outputs = model(inputs)
            # print("prof!!!!!", prof)
        # print("forward:\n", prof)
        print_events_mark("forward_done")

        loss = criterion(outputs, targets)

        print_events_mark("loss_done")

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        # .cudaSync()cuda_p2p
        optimizer.zero_grad()

        print_events_mark("zero_grad")

        #pdb.set_trace()

        # cuda_p2p.cudaSync()
        with torch.autograd.profiler.profile(profile_memory=True, record_shapes=True) as profb:
            loss.backward()
        print_events_mark("loss_backward_done")
        # print("backward\n:", profb)
        # cuda_p2p.cudaSync()

        optimizer.step()
        print_events_mark("step")

        # cuda_p2p.cudaSync()
        #print(list(model.classifier.weight)[0][:10])
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
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

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
            inputs, targets = inputs.cuda("cuda:0"), targets.cuda("cuda:0")
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

if __name__ == '__main__':
    main()