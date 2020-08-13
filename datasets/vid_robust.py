import os
import numpy as np
np.random.seed(7)
from PIL import Image
import itertools

import torch
import torch.utils.data as data

class VidRobustDataset(data.Dataset):

    def __init__(self, data_root, isTrain, transform=None, seq_num=2):
        self.data_root = data_root
        self.transform = transform
        self.seq_num = seq_num

        self.cls_to_id = torch.load(os.path.join(data_root, 'py_annot/imagenet/cls_to_id.da'))
        self.data = torch.load(os.path.join(data_root, 'py_annot/imagenet/val_robust_20.da'))

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        entry = self.data[idx]
        frames = entry['image']
        frm_ids = np.random.randint(0, len(frames), size=self.seq_num)        
        classes = [self.data[idx]['class'][frm_id] for frm_id in frm_ids]
        frm_pths = [frames[frm_id] for frm_id in frm_ids]

        imgs = []
        for frm_pth in frm_pths:
            img = Image.open(os.path.join(self.data_root, frm_pth))
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        return imgs, classes