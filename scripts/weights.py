import torch

weight = torch.load("/home/xueyan/antialias-cnn/data/lpf_weights/resnet101-5d3b4d8f.pth")
save_weight = {}
save_weight['state_dict'] = {}

for key in weight:
    if 'fc' not in key:
        save_weight['state_dict'][key] = weight[key]

# torch.save(save_weight, '/home/xueyan/antialias-cnn/data/lpf_weights/resnet101-5d3b4d8f_nofc.pth')