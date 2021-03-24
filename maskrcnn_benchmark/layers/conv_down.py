# Copyright (c) 2019, Xueyan Zou. All rights reserved.
import torch.nn as nn
import numpy as np
from maskrcnn_benchmark.layers import Conv2d
import torch.nn.functional as F
from maskrcnn_benchmark.layers.downsample import Downsample
import torch

class Conv2d_Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, 
        dilation=1, groups=1, bias=True, filter_bank=[3,5]):
        super(Conv2d_Down, self).__init__()

        # currently the filter is hacked, to be implemented in the future.
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, 
                                    dilation=dilation, groups=groups, bias=bias)
        self.blur_filter_3 = Downsample(filt_size=3, stride=stride, channels=in_channels)
        self.blur_filter_5 = Downsample(filt_size=5, stride=stride, channels=in_channels)
        self.attention = Conv2d(in_channels, 3, kernel_size=7, padding=7//2, stride=stride, bias=False)
        self.stride = stride

    def forward(self, x):
        # stride output
        n,c,h,w = x.shape
        y1 = self.conv(x)[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]
        y2 = self.conv(self.blur_filter_3(x))
        y3 = self.conv(self.blur_filter_5(x))

        attention = F.softmax(self.attention(x), dim=1)
        return attention[:,0:1,:,:]*y1 + attention[:,1:2,:,:]*y2 + attention[:,2:3,:,:]*y3

class MaxPool_Down(nn.Module):
    def __init__(self, in_channels, stride=2, filter_bank=[3,5]):
        super(MaxPool_Down, self).__init__()

        # currently the filter is hacked, to be implemented in the future.
        self.blur_filter_3 = Downsample(filt_size=3, stride=stride, channels=in_channels)
        self.blur_filter_5 = Downsample(filt_size=5, stride=stride, channels=in_channels)
        self.attention = Conv2d(in_channels, 3, kernel_size=7, padding=7//2, stride=stride, bias=False)
        self.stride = stride

    def forward(self, x):
        # stride output
        n,c,h,w = x.shape
        y1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        max_pool = F.max_pool2d(x, kernel_size=2, stride=1)
        y2 = self.blur_filter_3(max_pool)
        y3 = self.blur_filter_5(max_pool)

        attention = F.softmax(self.attention(x), dim=1)
        return attention[:,0:1,:,:]*y1 + attention[:,1:2,:,:]*y2 + attention[:,2:3,:,:]*y3
