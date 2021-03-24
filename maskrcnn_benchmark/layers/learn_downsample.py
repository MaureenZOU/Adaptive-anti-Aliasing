import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers import FrozenBatchNorm2d
import math

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

class Downsample_IS(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect',):
        super(Downsample_IS, self).__init__()
        self.d_x, self.d_y = torch.meshgrid(torch.arange(-(kernel_size//2),kernel_size//2 + 1), torch.arange(-(kernel_size//2),kernel_size//2 + 1))
        d = torch.sqrt(torch.pow(self.d_x.float(), 2) + torch.pow(self.d_y.float(), 2))[None,None,:,:]
        self.register_buffer('filt_d', d)

        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride

        self.conv_sigma = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, bias=False) # Adjust bias = True, or bias = Falase
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels//2, 1)
        
        # init the weight for conv layer and fc layer
        nn.init.kaiming_normal_(self.conv_sigma.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc.weight, a=1)

    def gaussian_1d(self, sigma):
        n,c = sigma.shape
        sigma = torch.clamp(sigma, 1e-4).reshape(n,c,1,1)
        gau_kernel = (1/(((2*math.pi)**0.5)*sigma))*torch.pow(math.e, -(1/2)*torch.pow(self.filt_d/sigma,2))
        n,c,h,w = gau_kernel.shape

        return gau_kernel/torch.sum(gau_kernel.reshape(n,c,h*w), dim=2)[:,:,None,None]

    def forward(self, x):
        sigma = self.conv_sigma(x)
        sigma = self.relu(sigma)
        sigma = self.avgpool(sigma)
        sigma = torch.sigmoid(self.fc(sigma.view(sigma.size(0), -1)))

        return F.conv2d(self.pad(x.permute(1,0,2,3)), self.gaussian_1d(sigma), stride=self.stride, groups=x.shape[0]).permute(0,1,2,3)
    
class Downsample_LS(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect',):
        super(Downsample_LS, self).__init__()
        self.d_x, self.d_y = torch.meshgrid(torch.arange(-(kernel_size//2),kernel_size//2 + 1), torch.arange(-(kernel_size//2),kernel_size//2 + 1))
        d = torch.sqrt(torch.pow(self.d_x.float(), 2) + torch.pow(self.d_y.float(), 2))[None,None,:,:].repeat((in_channels,1,1,1))
        self.register_buffer('filt_d', d)

        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride

        # init sigma as a parameter
        self.sigma = nn.Parameter(torch.tensor(0.5, device=torch.cuda.current_device()))

        # print('warning, pay attention to init in resnet not to replace this sigma')

    def gaussian_1d(self, sigma):
        sigma = torch.clamp(sigma, 1e-4)
        gau_kernel = (1/(((2*math.pi)**0.5)*sigma))*torch.pow(math.e, -(1/2)*torch.pow(self.filt_d/sigma,2))
        n,c,h,w = gau_kernel.shape
        return gau_kernel/torch.sum(gau_kernel.reshape(n,c,h*w), dim=2)[:,:,None,None]

    def forward(self, x):
        # print('warning, pay attention to change the sigma range while training')
        gau_kernel = self.gaussian_1d(self.sigma)
        # print(self.sigma)
        # print(gau_kernel)
        return F.conv2d(self.pad(x), gau_kernel, stride=self.stride, groups=x.shape[1])


class Blur_LS(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect',):
        super(Blur_LS, self).__init__()
        self.d_x, self.d_y = torch.meshgrid(torch.arange(-(kernel_size//2),kernel_size//2 + 1), torch.arange(-(kernel_size//2),kernel_size//2 + 1))
        d = torch.sqrt(torch.pow(self.d_x.float(), 2) + torch.pow(self.d_y.float(), 2))[None,None,:,:].repeat((in_channels,1,1,1))
        self.register_buffer('filt_d', d)

        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride

        # init sigma as a parameter
        self.sigma = nn.Parameter(torch.tensor(1.0, device=torch.cuda.current_device()))

        # print('warning, pay attention to init in resnet not to replace this sigma')

    def gaussian_1d(self, sigma):
        sigma = torch.clamp(sigma, 1e-4)
        gau_kernel = (1/(((2*math.pi)**0.5)*sigma))*torch.pow(math.e, -(1/2)*torch.pow(self.filt_d/sigma,2))
        n,c,h,w = gau_kernel.shape
        return gau_kernel/torch.sum(gau_kernel.reshape(n,c,h*w), dim=2)[:,:,None,None]

    def forward(self, x):
        # print('warning, pay attention to change the sigma range while training')
        gau_kernel = self.gaussian_1d(self.sigma)
        # print(self.sigma)
        # print(gau_kernel)
        return F.conv2d(self.pad(x), gau_kernel, stride=self.stride, groups=x.shape[1])

class Downsample_LSC(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect',):
        super(Downsample_LSC, self).__init__()
        self.d_x, self.d_y = torch.meshgrid(torch.arange(-(kernel_size//2),kernel_size//2 + 1), torch.arange(-(kernel_size//2),kernel_size//2 + 1))
        d = torch.sqrt(torch.pow(self.d_x.float(), 2) + torch.pow(self.d_y.float(), 2))[None,None,:,:].repeat((in_channels,1,1,1))
        self.register_buffer('filt_d', d)

        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride

        # init sigma as a parameter
        self.sigma = nn.Parameter(torch.zeros((in_channels, 1, 1, 1)))
        nn.init.normal_(self.sigma, std=0.01)

    def gaussian_1d(self, sigma):
        sigma = torch.clamp(sigma, 1e-4)
        gau_kernel = (1/(((2*math.pi)**0.5)*sigma))*torch.pow(math.e, -(1/2)*torch.pow(self.filt_d/sigma,2))
        n,c,h,w = gau_kernel.shape
        return gau_kernel/torch.sum(gau_kernel.reshape(n,c,h*w), dim=2)[:,:,None,None]

    def forward(self, x):
        # print('warning, pay attention to change the sigma range while training')
        gau_kernel = self.gaussian_1d(torch.sigmoid(self.sigma))
        return F.conv2d(self.pad(x), gau_kernel, stride=self.stride, groups=x.shape[1])


class Downsample_Spa(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect',):
        super(Downsample_Spa, self).__init__()
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

    def gaussian_1d(self, sigma, c=1):
        '''
        Sigma shape: [n, 1, h, w]
        filt_d shape: [n, k x k, 1, 1]
        output: [n, c, k x k, h x w]
        '''
        gau_kernel = (1/(((2*math.pi)**0.5)*sigma))*torch.pow(math.e, -(1/2)*torch.pow(self.filt_d/sigma,2))
        gau_kernel = gau_kernel / torch.sum(gau_kernel, dim=1, keepdim=True)
        n, k, h, w = gau_kernel.shape
        return gau_kernel.reshape(n, c, k, h*w)

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = torch.clamp(sigma, 1e-4)
        sigma = self.gaussian_1d(sigma)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        x = torch.sum(x * sigma, dim=2).reshape(n,c,h,w)

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
        self.bn = FrozenBatchNorm2d(self.kernel_size*self.kernel_size)

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