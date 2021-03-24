import torch
import torchvision.models as models

# new_model = {}
# model = torch.load('/home/xueyan/antialias-cnn/data/lpf_weights/resnet101_lpf3.pth.tar')

# new_model['model'] = model['state_dict']
# torch.save(new_model, '/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet101_lpf3_rz.pth')

# print(new_model.keys())

# model = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/e2e_mask_rcnn_R_50_FPN_1x.pth')
# print(model.keys())

key_dict = {
       'module.maxpool.1.filt_d' : 'module.resnet.stem.downsample.filt_d',
       'module.maxpool.1.conv.weight': 'module.resnet.stem.downsample.conv.weight',
       'module.maxpool.1.bn.weight': 'module.resnet.stem.downsample.bn.weight',
       'module.maxpool.1.bn.bias': 'module.resnet.stem.downsample.bn.bias',
       'module.maxpool.1.bn.running_mean': 'module.resnet.stem.downsample.bn.running_mean',
       'module.maxpool.1.bn.running_var': 'module.resnet.stem.downsample.bn.running_var',
       'module.maxpool.1.bn.num_batches_tracked': 'module.resnet.stem.downsample.bn.num_batches_tracked',
}

model = torch.load("/home/xueyan/antialias-cnn/data/checkpoints/resnet101_pasa_group8_softmax_warmup5_old/model_best.pth.tar")
out = {}
out['model'] = {}

for key in model['state_dict']:
        if key in key_dict:
                out['model'][key_dict[key]] = model['state_dict'][key]
        else:
                out['model'][key] = model['state_dict'][key]

torch.save(out, "/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet101_pasa_group8_softmax.pth")

# print(out['model'].keys())

# model = torch.load("/home/xueyan/inst-seg-all/data/output/Baseline-COCO_bn_pasa/model_0040000.pth")
# print(model['model'].keys())


# torch.save(out, "/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50_lpf3_RZ.pth")


# model = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50_imagenet_255.pth')
# print(model.keys())

# resnet50 = models.resnet50(pretrained=True)

# print(resnet50.mean)
# print(resnet50.std)

# print(resnet50)
# torch.save(resnet50, '/home/xueyan/inst-seg-all/data/checkpoints/pretrained/torch_resnet50.py')

# model = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50-19c8e357.pth')
# print(model.keys())

# weights = torch.load('/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50_imagenet_255_lpf.pth.tar')
# src_model = {}
# src_model['model'] = {}

# for key in weights['state_dict']:
#     if 'running' not in key:
        # if 'layer2.0.downsample.1.weight' in key:
        #     src_model['model']['layer2.0.downsample.0.weight'] = weights['state_dict'][key]
        # elif 'layer2.0.downsample.2.weight' in key:
        #     src_model['model']['layer2.0.downsample.1.weight'] = weights['state_dict'][key]
        # elif 'layer2.0.downsample.1.bias' in key:
        #     src_model['model']['layer2.0.downsample.0.bias'] = weights['state_dict'][key]
        # elif 'layer2.0.downsample.2.bias' in key:
        #     src_model['model']['layer2.0.downsample.1.bias'] = weights['state_dict'][key]

        # elif 'layer3.0.downsample.1.weight' in key:
        #     src_model['model']['layer3.0.downsample.0.weight'] = weights['state_dict'][key]
        # elif 'layer3.0.downsample.2.weight' in key:
        #     src_model['model']['layer3.0.downsample.1.weight'] = weights['state_dict'][key]
        # elif 'layer3.0.downsample.1.bias' in key:
        #     src_model['model']['layer3.0.downsample.0.bias'] = weights['state_dict'][key]
        # elif 'layer3.0.downsample.2.bias' in key:
        #     src_model['model']['layer3.0.downsample.1.bias'] = weights['state_dict'][key]

        # elif 'layer4.0.downsample.1.weight' in key:
        #     # print(weights['state_dict'][key].shape)
        #     src_model['model']['layer4.0.downsample.0.weight'] = weights['state_dict'][key]
        # elif 'layer4.0.downsample.2.weight' in key:
        #     src_model['model']['layer4.0.downsample.1.weight'] = weights['state_dict'][key]
        # elif 'layer4.0.downsample.1.bias' in key:
        #     src_model['model']['layer4.0.downsample.0.bias'] = weights['state_dict'][key]
        # elif 'layer4.0.downsample.2.bias' in key:
        #     src_model['model']['layer4.0.downsample.1.bias'] = weights['state_dict'][key]

        # else:
#         src_model['model'][key] = weights['state_dict'][key]


# torch.save(src_model, '/home/xueyan/inst-seg-all/data/checkpoints/pretrained/resnet50_imagenet_255_lpf.pth')

# print(src_model['model'].keys())