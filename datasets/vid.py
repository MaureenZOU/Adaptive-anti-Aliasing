import os
import random
from PIL import Image
import itertools

import torch
import torch.utils.data as data

class VidDataset(data.Dataset):

    def __init__(self, data_root, isTrain, transform=None, eval_imagenet=False, soft_class=False):
        self.data_root = data_root
        self.cls_to_id = torch.load(os.path.join(data_root, 'py_annot/imagenet/cls_to_id.da'))
        self.transform = transform

        if isTrain == True:
            self.data = torch.load(os.path.join(data_root, 'py_annot/imagenet/train.da'))
        else:
            self.data = torch.load(os.path.join(data_root, 'py_annot/imagenet/val.da'))

        self.soft_class = soft_class
        self.eval_imagenet = eval_imagenet
        if eval_imagenet == True:
            self.vid_cls_to_imagenet_id = torch.load(os.path.join(data_root, 'py_annot/imagenet/vid_cls_to_imagenet_id.da'))

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        entry = self.data[idx]
        frames = self.data[idx]['image']
        frm_id = random.randint(0, len(frames)-1)
        classes = self.data[idx]['class'][frm_id]

        frm_pth = frames[frm_id]
        img = Image.open(os.path.join(self.data_root, frm_pth))

        if self.eval_imagenet == False and self.soft_class == False:
            cls_ = self.cls_to_id[classes[random.randint(0, len(classes)-1)]]
        elif self.eval_imagenet == False and self.soft_class == True:
            cls_ = [self.cls_to_id[classes[i]] for i in range(0, len(classes))]
        else:
            cls_ = [self.vid_cls_to_imagenet_id[classes[i]] for i in range(0, len(classes))]
            cls_ = list(itertools.chain(*cls_))

        if self.transform is not None:
            img = self.transform(img)

        return img, cls_