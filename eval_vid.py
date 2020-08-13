# This code is built from the PyTorch examples repository: https://github.com/pytorch/examples/.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence

import models_lpf
from utils.helpers import save_config, save_grad
from networks import build_model
from datasets import VidDataset, VidRobustDataset
from datasets import VIDBatchCollator, VIDRobustBatchCollator

from IPython import embed

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/mnt/ssd/tmp/rzhang/ILSVRC2012',
                    help='path to dataset')
parser.add_argument('--dataset', default='imagenet',
                    help='dataset to train imagenet/vid')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-s', '--size', default=224, type=int,
                    help='image_size')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate-save', dest='evaluate_save', action='store_true',
                    help='save validation images off')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--sigma', default=0.0, type=float,
                    help='sigma value for gaussian layer')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Added functionality from PyTorch codebase
parser.add_argument('--no-data-aug', dest='no_data_aug', action='store_true',
                    help='no shift-based data augmentation')
parser.add_argument('--out-dir', dest='out_dir', default='./', type=str,
                    help='output directory')
parser.add_argument('-f','--filter_size', default=1, type=int,
                    help='anti-aliasing filter size')
parser.add_argument('-es', '--evaluate-shift', dest='evaluate_shift', action='store_true',
                    help='evaluate model on shift-invariance')
parser.add_argument('-esc', '--evaluate-shift-correct', dest='evaluate_shift_correct', action='store_true',
                    help='evaluate model on shift-invariance')
parser.add_argument('--epochs-shift', default=5, type=int, metavar='N',
                    help='number of total epochs to run for shift-invariance test')
parser.add_argument('-ed', '--evaluate-diagonal', dest='evaluate_diagonal', action='store_true',
                    help='evaluate model on diagonal')
parser.add_argument('-ba', '--batch-accum', default=1, type=int,
                    metavar='N',
                    help='number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--embed', dest='embed', action='store_true',
                    help='embed statement before anything is evaluated (for debugging)')
parser.add_argument('--val-debug', dest='val_debug', action='store_true',
                    help='debug by training on val set')
parser.add_argument('--weights', default=None, type=str, metavar='PATH',
                    help='path to pretrained model weights')
parser.add_argument('--save_weights', default=None, type=str, metavar='PATH',
                    help='path to save model weights')
parser.add_argument('--grad_debug', action='store_true',
                    help='whether perform gradient debugging')
parser.add_argument('--no_warmup', action='store_true',
                    help='whether perform warmup during training')
parser.add_argument('--warmup_epoch', default=15, type=int,
                    help='the number of epoch for warm up training')
parser.add_argument('--group', default=2, type=int,
                    help='group number of pasa operation')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes for prediction')
parser.add_argument('--lr_adj_num', default=3, type=int,
                    help='number of times to adjust learning rate')
parser.add_argument('--val_vid_imagenet',  action='store_true',
                    help='whether do we perform validation of vid on imagenet class')
parser.add_argument('--val_vid_soft',  action='store_true',
                    help='whether use soft label when using evaluation.')
parser.add_argument('--robust_num', default=2, type=int,
                    help='number of sample number in robust dataset')
parser.add_argument('--valid_class_pth',  default='../../data/ILSVRC2015/py_annot/imagenet/valid_class.da', type=str,
                    help='valid 288 class')

best_acc1 = 0

def main():
    args = parser.parse_args()

    if(not os.path.exists(args.out_dir)):
        os.mkdir(args.out_dir)

    # save training config
    if not(args.evaluate_diagonal or args.evaluate_shift or args.evaluate_save):
        save_config(args)
    
    # save grad debugging information
    if args.grad_debug == True:
        save_grad(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # create log file and timestamp
    log_pth = os.path.join(args.out_dir, 'log.txt')
    os.system('touch ' + log_pth)
    log_file = open(log_pth, 'a')
    log_file.write(str(datetime.now()) + '\n')
    log_file.close()

    # create model
    model = build_model(args.arch, args)

    if args.weights is not None:
        print("=> using saved weights [%s]"%args.weights)
        weights = torch.load(args.weights)

        new_weights_sd = {}
        for key in weights['state_dict']:
            new_weights_sd[key[7:]] = weights['state_dict'][key]
        weights['state_dict'] = new_weights_sd

        if args.num_classes != 1000 and (args.evaluate == False and args.evaluate_shift == False and args.evaluate_shift_correct == False and args.evaluate_diagonal == False and args.evaluate_save == False):
            model_dict = model.state_dict()

            # pop fc parameters
            new_weights_sd = {}
            for key in weights['state_dict']:
                if 'fc' not in key:
                     new_weights_sd[key] = weights['state_dict'][key]
            model_dict.update(new_weights_sd)
            weights['state_dict'] = model_dict

        model.load_state_dict(weights['state_dict'])
        # model.load_state_dict(weights)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if args.size == 224:
        l_size = 256
        s_size = 224
    elif args.size == 128:
        l_size = 174
        s_size = 128

    valdir = os.path.join(args.data, 'val')

    crop_size = l_size if(args.evaluate_shift or args.evaluate_diagonal or args.evaluate_save) else s_size
    args.batch_size = 1 if (args.evaluate_diagonal or args.evaluate_save) else args.batch_size

    if args.dataset == 'imagenet':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(l_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'vid':
        collator = VIDBatchCollator()
        val_loader = torch.utils.data.DataLoader(
            VidDataset(args.data, False, transforms.Compose([
                transforms.Resize(l_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]), args.val_vid_imagenet, args.val_vid_soft),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, collate_fn=collator)
    elif args.dataset == 'vid_robust':
        collator = VIDRobustBatchCollator()
        val_loader = torch.utils.data.DataLoader(
            VidRobustDataset(args.data, False, transforms.Compose([
                transforms.Resize(l_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]), args.robust_num),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, collate_fn=collator)
    else:
        assert False, "Not implemented error."

    if args.save_weights is not None: # "deparallelize" saved weights
        print("=> saving 'deparallelized' weights [%s]"%args.save_weights)
        # TO-DO: automatically save this during training
        if args.gpu is not None:
            torch.save({'state_dict': model.state_dict()}, args.save_weights)
        else:
            if(args.arch[:7]=='alexnet' or args.arch[:3]=='vgg'):
                model.features = model.features.module
                torch.save({'state_dict': model.state_dict()}, args.save_weights)
            else:
                torch.save({'state_dict': model.module.state_dict()}, args.save_weights)
        return

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if(args.evaluate_shift):
        validate_shift(val_loader, model, args)
        return

    if(args.evaluate_shift_correct):
        validate_shift_correct(val_loader, model, args)
        return

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.num_classes == 1000:
        valid_class = torch.load(args.valid_class_pth).cuda(args.gpu, non_blocking=True)
    else:
        valid_class = None

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            target = pad_sequence(target, batch_first=True, padding_value=-1)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = 0

            # measure accuracy and record loss
            acc1 = accuracy(output, target, valid_class, topk=(1,))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, top1=top1))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate_shift(val_loader, model, args):
    batch_time = AverageMeter()
    consist = AverageMeter()

    # switch to evaluate mode
    model.eval()
    np.random.seed(7)

    with torch.no_grad():
        end = time.time()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)

                output = model(input)
                cur_agree = agreement(output, args.robust_num)

                # measure agreement and record
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Ep [{0}/{1}]:\t'
                          'Test: [{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                           ep, args.epochs_shift, i, len(val_loader), batch_time=batch_time, consist=consist))

        print(' * Consistency {consist.avg:.3f}'
              .format(consist=consist))

    return consist.avg


def validate_shift_correct(val_loader, model, args):
    batch_time = AverageMeter()
    consist = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                off0 = np.random.randint(32,size=2)
                off1 = np.random.randint(32,size=2)

                output0 = model(input[:,:,off0[0]:off0[0]+args.size,off0[1]:off0[1]+args.size])
                output1 = model(input[:,:,off1[0]:off1[0]+args.size,off1[1]:off1[1]+args.size])

                cur_agree = agreement_correct(output0, output1, target).type(torch.FloatTensor).to(output0.device)

                # measure agreement and record
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Ep [{0}/{1}]:\t'
                          'Test: [{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                           ep, args.epochs_shift, i, len(val_loader), batch_time=batch_time, consist=consist))

        print(' * Consistency {consist.avg:.3f}'
              .format(consist=consist))

    return consist.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, valid_class, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        if valid_class is not None:
            valid_class = valid_class.reshape(1,1000)
            output = output * valid_class

        _, pred = output.topk(maxk, 1, True, True)
        intersect = (torch.sum(pred==target, dim=1) >= 1).float()

        # print(intersect)
        # pred = pred.t()
        # print(pred.shape)
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = torch.sum(intersect, dim=0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def agreement_correct(output0, output1, target):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)

    agree = pred0.eq(pred1)
    agree_target_pred0 = pred0.eq(target)
    agree_target_pred1 = pred1.eq(target)

    correct_or = (agree_target_pred0 + agree_target_pred1) > 0
    agree = agree * correct_or

    agree = 100.*(torch.sum(agree).float() / (torch.sum(correct_or).float() + 1e-10)).to(output0.device)
    return agree


def agreement(output, robust_num):
    pred = output.argmax(dim=1, keepdim=False)
    n,c = output.shape
    pred = pred.reshape(n//robust_num, robust_num)
    similarity = (pred == pred[:,0:1]).int()
    similarity = torch.sum(similarity, dim=1)
    agree = 100.*torch.mean((similarity == robust_num).float())

    return agree


if __name__ == '__main__':
    main()
