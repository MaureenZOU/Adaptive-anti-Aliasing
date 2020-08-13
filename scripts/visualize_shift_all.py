import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import cv2
import json
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

import matplotlib
matplotlib.use('Agg')
from scipy.misc import imsave
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt

import models_lpf
from utils import futils, Visualizer
from utils.helpers import save_config, CusImageFolder
from networks import build_model

from IPython import embed

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/mnt/ssd/tmp/rzhang/ILSVRC2012',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-s', '--size', default=224, type=int,
                    help='image_size')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
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
parser.add_argument('--weights', default=None, type=str, metavar='PATH',
                    help='path to pretrained model weights')
parser.add_argument('--sigma', default=0.0, type=float,
                    help='sigma value for gaussian layer')
parser.add_argument('--epochs-shift', default=5, type=int, metavar='N',
                    help='number of total epochs to run for shift-invariance test')


best_acc1 = 0

def main():
    args = parser.parse_args()

    if(not os.path.exists(args.out_dir)):
        os.mkdir(args.out_dir)

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

    # create model
    model = build_model(args.arch, args)

    if args.weights is not None:
        print("=> using saved weights [%s]"%args.weights)
        weights = torch.load(args.weights)

        # new_weights_sd = {}
        # for key in weights['state_dict']:
        #     new_weights_sd[key[7:]] = weights['state_dict'][key]
        # weights['state_dict'] = new_weights_sd

        # model.load_state_dict(weights['state_dict'])
        model.load_state_dict(weights)

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

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if args.size == 224:
        l_size = 256
        s_size = 224
    elif args.size == 128:
        l_size = 174
        s_size = 128

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    crop_size = l_size
    args.batch_size = args.batch_size

    val_loader = torch.utils.data.DataLoader(
        CusImageFolder(valdir, transforms.Compose([
            transforms.Resize(l_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init visualizer
    visual_folder = os.path.join(args.out_dir, 'visual')
    top_visualizer = Visualizer(visual_folder, demo_name='top_100.html')
    bottom_visualizer = Visualizer(visual_folder, demo_name='back_100.html')
    visual = [visual_folder, top_visualizer, bottom_visualizer]

    validate_shift(val_loader, model, args, visual)


def JS_Divergence(x, topk=5):
    pi = 1/x.shape[0]

    sum_across_sample = torch.sum(x, dim=0)
    arg_sum = torch.argsort(sum_across_sample, descending=True)
    arg_topk = arg_sum[0:topk]

    x = x[:,arg_topk]

    def H(x):
        return -torch.sum(x*torch.log2(x), dim=1, keepdim=True)

    return (H(torch.sum(x, dim=0, keepdim=True)*pi) - torch.sum(H(x), dim=0)*pi)[0].item()


def validate_shift(val_loader, model, args, visual):
    batch_time = AverageMeter()
    consist = []

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    visual_folder, top_visualizer, bottom_visualizer = visual
    
    # load json file for id to class
    idx_to_cls = json.load(open('../../data/imagenet-vid-robust/misc/imagenet_idx_to_name.json'))

    # when validate shift, batch size must be 1.
    assert args.batch_size == 1

    # switch to evaluate mode
    model.eval()
    visualizer_dict = {}

    # get evaluated class name
    src_pth = '/home/xueyan/antialias-cnn/data/output/resnet101_ori/visual'
    class_names = [x[0].split('/')[-1] for x in os.walk(src_pth)][1:]

    with torch.no_grad():
        end = time.time()

        video_pred_var = {}
        video_to_root_visual_pth = {}

        for i, ((input, target), pth) in enumerate(val_loader):
            hist_names = []

            img_visual_names = []
            img_visual_cap = []

            class_name, img_name = pth[0].split('/')[5], pth[0].split('/')[6]
            if i % 100 == 0:
                print('process ', '[' + str(i) + '|' + str(len(val_loader)) + ']')
            if class_name in class_names:
                continue
            
            input_ = input.clone()
            n,c,h,w = input.shape
            h_c, w_c = h//2, w//2
            half_s = args.size//2
            input = input[:,:,(h_c-half_s):(h_c+half_s),(w_c-half_s):(w_c+half_s)]

            # save input center
            center_pth = os.path.join(args.out_dir, 'visual', class_name)

            if class_name in visualizer_dict.keys():
                visualizer = visualizer_dict[class_name]
            else:
                visualizer = Visualizer(os.path.join(args.out_dir, 'visual', class_name), demo_name='index.html')
                visualizer_dict[class_name] = visualizer
            if not os.path.exists(center_pth):
                os.mkdir(center_pth)

            cur_img_name = img_name[:-5] + '_a_h_0_w_0.png'
            center_name = os.path.join(args.out_dir, 'visual', class_name, cur_img_name)
            hist_names.append(os.path.join(args.out_dir, 'visual', class_name, img_name[:-5] + '_bar_h_0_w_0.png'))
            imsave(center_name, ((input[0, :, :, :].permute(1,2,0).cpu().numpy())*np.array(std) + np.array(mean).reshape(1,1,3))*255)

            # append to html file
            visual_pth = os.path.join('/', args.out_dir.split('/')[-1], 'visual', class_name, cur_img_name)
            img_visual_names.append(visual_pth)
            img_visual_cap.append(cur_img_name)
            visualizer.insert(visual_pth, cur_img_name)

            offset = [-16,-10,-5,-2,-1,1,2,5,10,16]
            for k in range(0, len(offset)):
                # shift h
                shift_h = input_[:,:,(h_c+offset[k]-half_s):(h_c+offset[k]+half_s),(w_c-half_s):(w_c+half_s)]
                input = torch.cat((input, shift_h), dim=0)

                cur_img_name = img_name[:-5] + '_a_' + 'h_' + str(k) + '_w_0.png'
                h_name = os.path.join(args.out_dir, 'visual', class_name, cur_img_name)
                hist_names.append(os.path.join(args.out_dir, 'visual', class_name, img_name[:-5] + '_bar_h_' + str(k) + '_w_0.png'))
                imsave(h_name, ((shift_h[0, :, :, :].permute(1,2,0).cpu().numpy())*np.array(std) + np.array(mean).reshape(1,1,3))*255)
                # append to html file
                visual_pth = os.path.join('/', args.out_dir.split('/')[-1], 'visual', class_name, cur_img_name)
                img_visual_names.append(visual_pth)
                img_visual_cap.append(cur_img_name)
                visualizer.insert(visual_pth, cur_img_name)

                # shift w
                shift_w = input_[:,:,(h_c-half_s):(h_c+half_s),(w_c+offset[k]-half_s):(w_c+offset[k]+half_s)]
                input = torch.cat((input, shift_w), dim=0)

                cur_img_name = img_name[:-5] + '_a_h_0_w_' + str(k) + '.png'
                w_name = os.path.join(args.out_dir, 'visual', class_name, img_name[:-5] + '_a_h_0_w_' + str(k) + '.png')
                hist_names.append(os.path.join(args.out_dir, 'visual', class_name, img_name[:-5] + '_bar_h_0_w_' + str(k) + '.png'))
                imsave(w_name, ((shift_w[0, :, :, :].permute(1,2,0).cpu().numpy())*np.array(std) + np.array(mean).reshape(1,1,3))*255)
                # append to html file
                visual_pth = os.path.join('/', args.out_dir.split('/')[-1], 'visual', class_name, cur_img_name)
                img_visual_names.append(visual_pth)
                img_visual_cap.append(cur_img_name)
                visualizer.insert(visual_pth, cur_img_name)

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = torch.nn.Softmax(dim=1)(model(input))
            consist.append(JS_Divergence(output))
            
            output_sort, output_argsort = torch.sort(output, dim=1, descending=True)

            for k in range(input.shape[0]):
                top5_arg_name = [idx_to_cls[str(x.item())] for x in output_argsort[k,0:5]]
                top5_prob = output_sort[k,0:5].cpu().numpy()

                ax = sns.barplot(x=top5_arg_name, y=top5_prob)
                plt.savefig(hist_names[k])
                plt.clf()

                cur_img_name = hist_names[k].split('/')[-1]
                visual_pth = os.path.join('/', args.out_dir.split('/')[-1], 'visual', class_name, cur_img_name)
                img_visual_names.append(visual_pth)
                img_visual_cap.append(cur_img_name)
                visualizer.insert(visual_pth, cur_img_name)

            video_to_root_visual_pth[pth[0]] = [sorted(img_visual_names), sorted(img_visual_cap)]
            video_pred_var[pth[0]] = JS_Divergence(output)

    for key in visualizer_dict:
        visualizer_dict[key].write()


def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100.*torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree


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


if __name__ == '__main__':
    main()
