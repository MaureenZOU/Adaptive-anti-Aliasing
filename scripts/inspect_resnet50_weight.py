import torch
import pprint

'''
msra is trained with bias, but pytorch_resnet50 is not
xy_resnet50, rz_resnet50_lpf, xy_resnet50_lpf and msra doesn't save running variables !!! double check what does this running mean and running variance means
'''

msra_resnet50 = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/msra_resnet50.pth')['model']
pytorch_resnet50 = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50-19c8e357.pth')
rz_resnet50_lpf = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50_lpf3_RZ.pth')['model']
xy_resnet50 = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50_imagenet_255.pth')['model']
xy_resnet50_lpf = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50_imagenet_255_lpf.pth')['model']

# remove module from xy models
xy_resnet50 = dict((key.replace('module.',''), value) for (key, value) in xy_resnet50.items())
xy_resnet50_lpf = dict((key.replace('module.',''), value) for (key, value) in xy_resnet50_lpf.items())

# remove running mean and runnign variance in pytorch_resnet50
new_pytorch_resnet50 = {}
for key in pytorch_resnet50:
    if 'running' not in key:
        new_pytorch_resnet50[key] = pytorch_resnet50[key]
pytorch_resnet50 = new_pytorch_resnet50

# fit lpf model to resnet model on downsample layer
new_rz_resnet50_lpf = {}
for key in rz_resnet50_lpf:
    if 'num_batches_tracked' not in key and 'filt' not in key:
        if 'layer2.0.downsample.1.weight' in key:
            new_rz_resnet50_lpf['layer2.0.downsample.0.weight'] = rz_resnet50_lpf[key]
        elif 'layer2.0.downsample.2.weight' in key:
            new_rz_resnet50_lpf['layer2.0.downsample.1.weight'] = rz_resnet50_lpf[key]
        elif 'layer2.0.downsample.1.bias' in key:
            new_rz_resnet50_lpf['layer2.0.downsample.0.bias'] = rz_resnet50_lpf[key]
        elif 'layer2.0.downsample.2.bias' in key:
            new_rz_resnet50_lpf['layer2.0.downsample.1.bias'] = rz_resnet50_lpf[key]

        elif 'layer3.0.downsample.1.weight' in key:
            new_rz_resnet50_lpf['layer3.0.downsample.0.weight'] = rz_resnet50_lpf[key]
        elif 'layer3.0.downsample.2.weight' in key:
            new_rz_resnet50_lpf['layer3.0.downsample.1.weight'] = rz_resnet50_lpf[key]
        elif 'layer3.0.downsample.1.bias' in key:
            new_rz_resnet50_lpf['layer3.0.downsample.0.bias'] = rz_resnet50_lpf[key]
        elif 'layer3.0.downsample.2.bias' in key:
            new_rz_resnet50_lpf['layer3.0.downsample.1.bias'] = rz_resnet50_lpf[key]

        elif 'layer4.0.downsample.1.weight' in key:
            # print(weights['state_dict'][key].shape)
            new_rz_resnet50_lpf['layer4.0.downsample.0.weight'] = rz_resnet50_lpf[key]
        elif 'layer4.0.downsample.2.weight' in key:
            new_rz_resnet50_lpf['layer4.0.downsample.1.weight'] = rz_resnet50_lpf[key]
        elif 'layer4.0.downsample.1.bias' in key:
            new_rz_resnet50_lpf['layer4.0.downsample.0.bias'] = rz_resnet50_lpf[key]
        elif 'layer4.0.downsample.2.bias' in key:
            new_rz_resnet50_lpf['layer4.0.downsample.1.bias'] = rz_resnet50_lpf[key]
        elif 'conv3.1.weight' in key:
            new_rz_resnet50_lpf[key[0:-8] + 'weight'] = rz_resnet50_lpf[key]
        else:
            new_rz_resnet50_lpf[key] = rz_resnet50_lpf[key]
rz_resnet50_lpf = new_rz_resnet50_lpf

new_xy_resnet50_lpf = {}
for key in xy_resnet50_lpf:
    if 'num_batches_tracked' not in key and 'filt' not in key:
        if 'layer2.0.downsample.1.weight' in key:
            new_xy_resnet50_lpf['layer2.0.downsample.0.weight'] = xy_resnet50_lpf[key]
        elif 'layer2.0.downsample.2.weight' in key:
            new_xy_resnet50_lpf['layer2.0.downsample.1.weight'] = xy_resnet50_lpf[key]
        elif 'layer2.0.downsample.1.bias' in key:
            new_xy_resnet50_lpf['layer2.0.downsample.0.bias'] = xy_resnet50_lpf[key]
        elif 'layer2.0.downsample.2.bias' in key:
            new_xy_resnet50_lpf['layer2.0.downsample.1.bias'] = xy_resnet50_lpf[key]

        elif 'layer3.0.downsample.1.weight' in key:
            new_xy_resnet50_lpf['layer3.0.downsample.0.weight'] = xy_resnet50_lpf[key]
        elif 'layer3.0.downsample.2.weight' in key:
            new_xy_resnet50_lpf['layer3.0.downsample.1.weight'] = xy_resnet50_lpf[key]
        elif 'layer3.0.downsample.1.bias' in key:
            new_xy_resnet50_lpf['layer3.0.downsample.0.bias'] = xy_resnet50_lpf[key]
        elif 'layer3.0.downsample.2.bias' in key:
            new_xy_resnet50_lpf['layer3.0.downsample.1.bias'] = xy_resnet50_lpf[key]

        elif 'layer4.0.downsample.1.weight' in key:
            new_xy_resnet50_lpf['layer4.0.downsample.0.weight'] = xy_resnet50_lpf[key]
        elif 'layer4.0.downsample.2.weight' in key:
            new_xy_resnet50_lpf['layer4.0.downsample.1.weight'] = xy_resnet50_lpf[key]
        elif 'layer4.0.downsample.1.bias' in key:
            new_xy_resnet50_lpf['layer4.0.downsample.0.bias'] = xy_resnet50_lpf[key]
        elif 'layer4.0.downsample.2.bias' in key:
            new_xy_resnet50_lpf['layer4.0.downsample.1.bias'] = xy_resnet50_lpf[key]
        
        elif 'conv3.1.weight' in key:
            print(key)
            new_xy_resnet50_lpf[key[0:-8] + 'weight'] = xy_resnet50_lpf[key]
        else:
            new_xy_resnet50_lpf[key] = xy_resnet50_lpf[key]
xy_resnet50_lpf = new_xy_resnet50_lpf

# fit xy_resnet50 to pool
new_xy_resnet50 = {}
for key in xy_resnet50:
    if 'num_batches_tracked' not in key:
        new_xy_resnet50[key] = xy_resnet50[key]
xy_resnet50 = new_xy_resnet50

# fit msra fc layer
new_msra_resnet50 = {}
for key in msra_resnet50:
    # if 'downsample' in key:
    #     print(key)
    if not ('conv' in key and 'bias' in key):
        if not ('downsample.0' in key and 'bias' in key):
            if 'fc1000.weight' in key:
                new_msra_resnet50['fc.weight'] = msra_resnet50[key]
            elif 'fc1000.bias' in key:
                new_msra_resnet50['fc.bias'] = msra_resnet50[key]
            else:
                new_msra_resnet50[key] = msra_resnet50[key]
msra_resnet50 = new_msra_resnet50


save_model = {}
save_model['model'] = xy_resnet50
torch.save(save_model, '/home/xueyan/inst-seg-all/data/checkpoints/pretrained/strip_xy_resnet50.pth')

save_model = {}
save_model['model'] = pytorch_resnet50
torch.save(save_model, '/home/xueyan/inst-seg-all/data/checkpoints/pretrained/strip_pytorch_resnet50.pth')


# for key in msra_resnet50:
#     if 'bn' in key:
#         print('mean', key, str(torch.mean(xy_resnet50_lpf[key]).item())[0:7], str(torch.mean(rz_resnet50_lpf[key]).item())[0:7], str(torch.mean(xy_resnet50[key]).item())[0:7], str(torch.mean(pytorch_resnet50[key]).item())[0:7], str(torch.mean(msra_resnet50[key]).item())[0:7])
#         print('var ', key, str(torch.var(xy_resnet50_lpf[key]).item())[0:7], str(torch.var(rz_resnet50_lpf[key]).item())[0:7], str(torch.var(xy_resnet50[key]).item())[0:7], str(torch.var(pytorch_resnet50[key]).item())[0:7], str(torch.var(msra_resnet50[key]).item())[0:7])