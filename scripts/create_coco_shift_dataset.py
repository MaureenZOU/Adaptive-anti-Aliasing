'''
'''

import torch
import json
from pycocotools.coco import COCO
from maskrcnn_benchmark.utils import cv2_util
import numpy as np
import os
import cv2
from coco_helpers import binary_mask_to_polygon
from maskrcnn_benchmark.structures.segmentation_mask import BinaryMaskList

coco_annot_pth = '/home/xueyan/inst-seg-all/data/coco/annotations/instances_minival2014.json'
coco_annot_out_shift = '/home/xueyan/inst-seg-all/data/coco/annotations/instances_minival2014_shift.json'
coco_val_img_pth = '/home/xueyan/inst-seg-all/data/coco/val2014'
image_output_pth = '/home/xueyan/inst-seg-all/data/coco/val2014_shift'

coco_json_annot = json.load(open(coco_annot_pth))
coco_dataset = COCO(coco_annot_pth)

coco_shift_annot = {}
coco_shift_annot['info'] = coco_json_annot['info']
coco_shift_annot['licenses'] = coco_json_annot['licenses']
coco_shift_annot['type'] = coco_json_annot['type']
coco_shift_annot['categories'] = coco_json_annot['categories']

image_ids = coco_dataset.getImgIds()
direction = np.array([[-1,0], [1,0], [0,1], [0,-1]])
norm = np.array([1,2,5,10,20])
IMG_ID = 1
SEG_ID = 1
SEQ_ID = 0

seg_seq = []
img_seq = []

for i in range(0, len(image_ids)):
# for i in range(0, 3):
    print('process [{},{}]'.format(i, len(image_ids)))
    img_info = coco_dataset.loadImgs(image_ids[i])[0]
    shift_img_lst = []

    ori_height = img_info['height']
    ori_width = img_info['width']

    new_height = ori_height - 41
    new_width = ori_width - 41

    c_h, c_w = ori_height // 2, ori_width // 2

    img_name = img_info['file_name']

    SFT_ID = 0
    has_img = False
    for d in range(0, len(direction)):
        for n in range(0, len(norm)):
            shift = direction[d] * norm[n]
            shift_img_name = img_name[:-4] + '_w_' + str(shift[0]) + '_h_' + str(shift[1]) + '.jpg'
            img_inst_dict = {}

            old_img = cv2.imread(os.path.join(coco_val_img_pth, img_name))
            
            coor = np.array([[c_w - new_width//2, c_h - new_height//2], [c_w + new_width//2, c_h + new_height//2]]) - shift
            x1, y1, x2, y2 = coor[0,0], coor[0,1], coor[1,0], coor[1,1]

            new_img = old_img[y1:y2, x1:x2]

            img_inst_dict['license'] = img_info['license']
            img_inst_dict['date_captured'] = img_info['date_captured']
            img_inst_dict['url'] = img_info['url']
            img_inst_dict['height'] = int(new_img.shape[0])
            img_inst_dict['width'] = int(new_img.shape[1])
            img_inst_dict['id'] = IMG_ID
            img_inst_dict['seq_id'] = SEQ_ID
            img_inst_dict['sft_id'] = SFT_ID
            img_inst_dict['file_name'] = shift_img_name

            annot_ids = coco_dataset.getAnnIds(imgIds=[image_ids[i]])
            annot = coco_dataset.loadAnns(annot_ids)

            has_annot = False
            for a in range(0, len(annot)):
                seg_inst_dict = {}

                # segmentation, area, iscrowd, image_id, bbox, category_id, id
                mask = coco_dataset.annToMask(annot[a])
                mask = mask[y1:y2, x1:x2]

                area = int(np.sum(mask))
                if area < 10:
                    continue

                has_annot = True
                Mask = BinaryMaskList(torch.tensor(mask), size = (mask.shape[1], mask.shape[0]))
                mask_poly_obj = Mask.convert_to_polygon()

                if len(mask_poly_obj.polygons) == 0:
                    mask_poly = binary_mask_to_polygon(mask)
                else:
                    mask_poly = [x.tolist() for x in mask_poly_obj.polygons[0].polygons]

                x1_, y1_, w_, h_ = annot[a]['bbox']
                x2_, y2_ = x1_ + w_, y1_ + h_
                b_x1, b_y1, b_x2, b_y2 = min(max(x1_ - x1, 0), int(new_img.shape[1])), min(max(y1_ - y1, 0), int(new_img.shape[0])), min(max(x2_ - x1, 0), int(new_img.shape[1])), min(max(y2_ - y1, 0), int(new_img.shape[0]))
                b_w, b_h = b_x2 - b_x1, b_y2 - b_y1

                seg_inst_dict['segmentation'] = mask_poly
                seg_inst_dict['area'] = area
                seg_inst_dict['iscrowd'] = annot[a]['iscrowd']
                seg_inst_dict['image_id'] = IMG_ID
                seg_inst_dict['bbox'] = [int(b_x1), int(b_y1), int(b_w), int(b_h)]
                seg_inst_dict['category_id'] = annot[a]['category_id']
                seg_inst_dict['id'] = SEG_ID
                seg_seq.append(seg_inst_dict)
                SEG_ID += 1
            
            if has_annot == True:
                has_img = True
                img_seq.append(img_inst_dict)
                # cv2.imwrite(os.path.join(image_output_pth, shift_img_name), new_img)
                # break

            IMG_ID += 1
            SFT_ID += 1

        # if has_img == True:
        #     break

    if has_img == True:
        SEQ_ID += 1

coco_shift_annot['seq_num'] = SEQ_ID
coco_shift_annot['sft_num'] = 20
# coco_shift_annot['sft_num'] = 1
coco_shift_annot['categories'] = coco_json_annot['categories']
coco_shift_annot['images'] = img_seq
coco_shift_annot['annotations'] = seg_seq
json.dump(coco_shift_annot, open(coco_annot_out_shift, 'w'))