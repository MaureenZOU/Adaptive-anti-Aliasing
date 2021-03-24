from pycocotools.coco import COCO
from maskrcnn_benchmark.utils import cv2_util
import cv2
import os
import numpy as np

annot_pth = '/home/xueyan/inst-seg-all/data/coco/annotations/instances_minival2014_shift.json'
img_pth = '/home/xueyan/inst-seg-all/data/coco/val2014_shift'
coco_dataset = COCO(annot_pth)

image_ids = coco_dataset.getImgIds()

for i in range(0, len(image_ids)):
    img_info = coco_dataset.loadImgs(image_ids[i])[0]
    img = cv2.imread(os.path.join(img_pth, img_info['file_name']))

    annot_ids = coco_dataset.getAnnIds(imgIds=[image_ids[i]])
    annot = coco_dataset.loadAnns(annot_ids)

    for a in range(0, len(annot)):
        mask = coco_dataset.annToMask(annot[a])
        contours, hierarchy = cv2_util.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, [255, 255, 255], 3)

        x,y,w,h = annot[a]['bbox']
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imwrite(os.path.join('/home/xueyan/tmp_visual/shift_', str(i).zfill(3) + 'img.png'), img)
