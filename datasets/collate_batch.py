import torch
import itertools

class VIDBatchCollator(object):
    def __call__(self, batch):
        images = torch.stack([x[0] for x in batch], dim=0)
        targets = [torch.tensor(x[1]) for x in batch]
        return images, targets

class VIDRobustBatchCollator(object):
    def __call__(self, batch):
        imgs = [x[0] for x in batch]
        imgs = list(itertools.chain(*imgs))
        images = torch.stack(imgs, dim=0)
        targets = [x[1] for x in batch]
        targets = list(itertools.chain(*targets))
        return images, targets