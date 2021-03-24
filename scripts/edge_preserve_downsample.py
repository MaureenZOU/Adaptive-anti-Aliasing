import torch
import torch.nn.functional as F
import json
import cv2
import os
from pycocotools.coco import COCO
import numpy as np
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.layers.downsample import Downsample

id = 1124

coco_annot_pth = '/home/xueyan/inst-seg-all/data/coco/annotations/instances_minival2014.json'
img_root = '/home/xueyan/inst-seg-all/data/coco/val2014/'

coco_dataset = COCO(coco_annot_pth)

ids = coco_dataset.getImgIds()
id = ids[id]

img_name = coco_dataset.loadImgs(id)
img = cv2.imread(os.path.join(img_root, img_name[0]['file_name']))

annot_ids = coco_dataset.getAnnIds(imgIds=[id])
annot = coco_dataset.loadAnns(annot_ids)
sedge = np.zeros(img.shape).astype(np.uint8)
mask_full = np.zeros(img.shape[:-1])
for i in range(0, len(annot)):
    mask = coco_dataset.annToMask(annot[i])
    mask_full += mask
    # contour_unint8 = np.array(annot[i]['segmentation']).astype(np.uint8)
    # h,w = contour_unint8.shape
    # print(contour_unint8.reshape(h,w//2,2))
    contours, hierarchy = cv2_util.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    sedge = cv2.drawContours(sedge, contours, -1, [255, 255, 255], 3)

stride = 2
cv2.imwrite('mask_full.png', mask_full*255)

# mask_full = torch.tensor(mask_full)[None,None]*255

mask_full = torch.tensor(img).permute(2,0,1)[None,:].float()

n,c,h,w = mask_full.shape
mask_down = mask_full[:,:,torch.arange(h)%stride==0,:][:,:,:,torch.arange(w)%stride==0]
n,c,h,w = mask_down.shape
# mask_stride = F.interpolate(mask_down, (640, 621))
mask_down = mask_down[:,:,torch.arange(h)%stride==0,:][:,:,:,torch.arange(w)%stride==0]
n,c,h,w = mask_down.shape
# mask_stride = F.interpolate(mask_down, (640, 621))
mask_stride = mask_down[:,:,torch.arange(h)%stride==0,:][:,:,:,torch.arange(w)%stride==0]

mask_stride = F.interpolate(mask_stride, (640, 621))



down = Downsample(filt_size=7, stride=stride, channels=3)
mask_blur = down(mask_full.float())
# mask_blur = F.interpolate(mask_blur, (640, 621))
mask_blur = down(mask_blur.float())
mask_blur = down(mask_blur.float())

mask_blur = F.interpolate(mask_blur, (640, 621))


print(mask_blur.shape)
print(mask_full.shape)
print(mask_stride.shape)

print(torch.sum(torch.abs(mask_blur.float() - mask_full.float())))
print(torch.sum(torch.abs(mask_stride.float() - mask_full.float())))


# n,c,h,w = mask_full.shape
# mask_down = mask_full[:,:,torch.arange(h)%stride==0,:][:,:,:,torch.arange(w)%stride==0]
# cv2.imwrite('mask_stride.png', F.interpolate(mask_down, scale_factor=8)[0].permute(1,2,0).numpy())

# down = Downsample(filt_size=7, stride=8, channels=3)
# mask_full = down(mask_full.float())
# cv2.imwrite('mask_blur.png', F.interpolate(mask_full, scale_factor=8)[0].permute(1,2,0).numpy()*255)


# mask_inter = F.interpolate(mask_full, scale_factor=0.125)[0].permute(1,2,0).numpy()
# print(mask_down.shape)


# sedge = torch.tensor(sedge).float().mean(dim=2)[None,None]
# img = torch.tensor(img).permute(2,0,1)[None,:].float()
# down = Downsample(filt_size=7, stride=stride, channels=3)
# blur = Downsample(filt_size=7, stride=1, channels=3)
# img = down(blur(img))[0].permute(1,2,0).numpy()
# n,c,h,w = img.shape
# img = img[:,:,torch.arange(h)%stride==0,:][:,:,:,torch.arange(w)%stride==0][0].permute(1,2,0).numpy()
# n,c,h,w = sedge.shape
# blur = Downsample(filt_size=7, stride=1, channels=3)
# sedge = blur(sedge)
# print(sedge)
# sedge_down = sedge[:,:,torch.arange(h)%stride==0,:][:,:,:,torch.arange(w)%stride==0][0].permute(1,2,0).numpy()

# cv2.imwrite('down_down.png', img)
# cv2.imwrite('edge.png', sedge_down)

# print(sedge.shape)
# print(img.shape)