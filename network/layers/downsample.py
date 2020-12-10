# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Downsample_Debug(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0, summarywriter=None, visual_freq=10):
        super(Downsample_Debug, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

        self.summarywriter = summarywriter
        self.visual_freq = visual_freq
        self.cnt = 0

        self.register_backward_hook(self.backward_hook)
            
    def backward_hook(self, module, grad_input, grad_output):
        n,c,h,w = grad_output[0].shape
        if self.summarywriter is not None and torch.cuda.current_device() == 0:
            if self.cnt % self.visual_freq == 0:
                with torch.no_grad():
                    v = torch.norm(grad_output[0][0], dim=0)
                    v = ((v / torch.max(v)) * 255).byte().cpu().numpy()
                    v = cv2.applyColorMap(v, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_grad_output', v, self.cnt, dataformats='HWC')

    def forward(self, inp):

        if self.summarywriter is not None and torch.cuda.current_device() == 0:
            if self.cnt % self.visual_freq == 0:
                with torch.no_grad():
                    size = inp.shape[2]
                    cnum = int(inp.shape[1]**0.5)
                    out_img = np.zeros((cnum*size, cnum*size, 3), dtype=np.uint8)
                    n,c,h,w = inp.shape

                    for i in range(0, cnum):
                        for j in range(0, cnum):
                            start_w = i * size
                            start_h = j * size
                            img = inp[0, i*cnum+j, :, :]
                            img = ((img / torch.max(img)) * 255).byte().cpu().numpy()
                            img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)[:,:,::-1]
                            out_img[start_w:(start_w+size), start_h:(start_h+size)] = img

                    self.summarywriter.add_image(str(h) + 'x' + str(w) + 'x' + str(c) + '_bdown', out_img, self.cnt, dataformats='HWC')
            self.cnt += 1

        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2, summarywriter=None, visual_freq=10):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]


class Downsample_PASA(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect',):
        super(Downsample_PASA, self).__init__()
        d_x, d_y = torch.meshgrid(torch.arange(-(kernel_size//2),kernel_size//2 + 1), torch.arange(-(kernel_size//2),kernel_size//2 + 1))
        d = torch.sqrt(torch.pow(d_x.float(), 2) + torch.pow(d_y.float(), 2))
        d = d.flatten()[None,:, None, None]
        self.register_buffer('filt_d', d)

        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels, self.kernel_size*self.kernel_size, kernel_size=3, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.kernel_size*self.kernel_size)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = torch.clamp(sigma, 1e-4)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)
        sigma = sigma / torch.sum(sigma, dim=2, keepdim=True)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        x = torch.sum(x * sigma, dim=2).reshape(n,c,h,w)

        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]


class Downsample_PASA_Debug(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', summarywriter=None, visual_freq=10):
        super(Downsample_PASA_Debug, self).__init__()
        d_x, d_y = torch.meshgrid(torch.arange(-(kernel_size//2),kernel_size//2 + 1), torch.arange(-(kernel_size//2),kernel_size//2 + 1))
        d = torch.sqrt(torch.pow(d_x.float(), 2) + torch.pow(d_y.float(), 2))
        d = d.flatten()[None,:, None, None]
        self.register_buffer('filt_d', d)

        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels, self.kernel_size*self.kernel_size, kernel_size=3, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.kernel_size*self.kernel_size)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.summarywriter = summarywriter
        self.visual_freq = visual_freq
        self.cnt = 0

        self.register_backward_hook(self.backward_hook)
            
    def backward_hook(self, module, grad_input, grad_output):
        n,c,h,w = grad_output[0].shape
        if self.summarywriter is not None and torch.cuda.current_device() == 0:
            if self.cnt % self.visual_freq == 0:
                with torch.no_grad():
                    v = torch.norm(grad_output[0][0], dim=0)
                    v = ((v / torch.max(v)) * 255).byte().cpu().numpy()
                    v = cv2.applyColorMap(v, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_grad_output', v, self.cnt, dataformats='HWC')


    def forward(self, x):
        if self.summarywriter is not None and torch.cuda.current_device() == 0:
            if self.cnt % self.visual_freq == 0:
                with torch.no_grad():
                    size = x.shape[2]
                    cnum = int(x.shape[1]**0.5)
                    out_img = np.zeros((cnum*size, cnum*size, 3), dtype=np.uint8)
                    n,c,h,w = x.shape

                    for i in range(0, cnum):
                        for j in range(0, cnum):
                            start_w = i * size
                            start_h = j * size
                            img = x[0, i*cnum+j, :, :]
                            img = ((img / torch.max(img)) * 255).byte().cpu().numpy()
                            img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)[:,:,::-1]
                            out_img[start_w:(start_w+size), start_h:(start_h+size)] = img

                    self.summarywriter.add_image(str(h) + 'x' + str(w) + 'x' + str(c) + '_bdown', out_img, self.cnt, dataformats='HWC')


        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = torch.clamp(sigma, 1e-4)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)
        if self.summarywriter is not None and torch.cuda.current_device() == 0:
            if self.cnt % self.visual_freq == 0:
                with torch.no_grad():
                    v1 = sigma[0,0,0].reshape(h,w)
                    v1 = ((v1 / torch.max(v1)) * 255).byte().cpu().numpy()
                    v1 = cv2.applyColorMap(v1, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_0', v1, self.cnt, dataformats='HWC')

                    v2 = sigma[0,0,1].reshape(h,w)
                    v2 = ((v2 / torch.max(v2)) * 255).byte().cpu().numpy()
                    v2 = cv2.applyColorMap(v2, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_1', v2, self.cnt, dataformats='HWC')

                    v3 = sigma[0,0,2].reshape(h,w)
                    v3 = ((v3 / torch.max(v3)) * 255).byte().cpu().numpy()
                    v3 = cv2.applyColorMap(v3, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_2', v3, self.cnt, dataformats='HWC')

                    v4 = sigma[0,0,3].reshape(h,w)
                    v4 = ((v4 / torch.max(v4)) * 255).byte().cpu().numpy()
                    v4 = cv2.applyColorMap(v4, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_3', v4, self.cnt, dataformats='HWC')

                    v5 = sigma[0,0,4].reshape(h,w)
                    v5 = ((v5 / torch.max(v5)) * 255).byte().cpu().numpy()
                    v5 = cv2.applyColorMap(v5, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_4', v5, self.cnt, dataformats='HWC')

                    v6 = sigma[0,0,5].reshape(h,w)
                    v6 = ((v6 / torch.max(v6)) * 255).byte().cpu().numpy()
                    v6 = cv2.applyColorMap(v6, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_5', v6, self.cnt, dataformats='HWC')

                    v7 = sigma[0,0,6].reshape(h,w)
                    v7 = ((v7 / torch.max(v7)) * 255).byte().cpu().numpy()
                    v7 = cv2.applyColorMap(v7, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_6', v7, self.cnt, dataformats='HWC')

                    v8 = sigma[0,0,7].reshape(h,w)
                    v8 = ((v8 / torch.max(v8)) * 255).byte().cpu().numpy()
                    v8 = cv2.applyColorMap(v8, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_7', v8, self.cnt, dataformats='HWC')

                    v9 = sigma[0,0,8].reshape(h,w)
                    v9 = ((v9 / torch.max(v9)) * 255).byte().cpu().numpy()
                    v9 = cv2.applyColorMap(v9, cv2.COLORMAP_OCEAN)[:,:,::-1]
                    self.summarywriter.add_image(str(h) + 'x' + str(w) + '_8', v9, self.cnt, dataformats='HWC')

            self.cnt += 1

        sigma = sigma / torch.sum(sigma, dim=2, keepdim=True)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        x = torch.sum(x * sigma, dim=2).reshape(n,c,h,w)

        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer