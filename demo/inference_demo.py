
import sys
sys.path.insert(0, '..')

from maskrcnn_benchmark.config import cfg
from predictor import Demo
import cv2
from scipy.misc import imsave
import glob
import os
import numpy as np
import futils as fu
import torch
import random


# params
args = fu.Args()
args.opts = []
args.dataset = 'COCO' # 'COCO', 'YoutubeVOS'
args.config_file = "../configs/gn_baselines/scratch_e2e_mask_rcnn_R_50_FPN_3x_gn.yaml"
args.data_dir = '/home/xueyan/inst-seg-all/data/coco/test2014_shift'
args.vis_dir = '../../../data/checkpoints/Baseline-COCO_gn_blur_correct/visual_coco_shift'
args.postfix = 'jpg'
args.rand_seed = 777
args.conf_thresh = 0.5
args.vid_num = 20
args.img_num = 50
args.frame_per_vid = 20

args.opts += ['MODEL.WEIGHT', '../../../data/checkpoints/Baseline-COCO_gn_blur_correct/model_final.pth']
args.opts += ['MODEL.ROI_BOX_HEAD.NUM_CLASSES', 81] # COCO: 81, YoutubeVOS: 29
args.opts += ['INPUT.MIN_SIZE_TRAIN', (500,)] # 800
args.opts += ['INPUT.MAX_SIZE_TRAIN', 700] # 1333
args.opts += ['MODEL.RESNETS.BACKBONE_OUT_CHANNELS', 160] # 256

# Blur Downsample
args.opts += ['MODEL.BLUR_DOWN', True] # False

# update the config options with the config file
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

demo = Demo(
    cfg,
    args.dataset,
    confidence_threshold=args.conf_thresh,
    min_image_size=800,
)

# set random seed
np.random.seed(args.rand_seed)
random.seed(args.rand_seed)
torch.random.manual_seed(args.rand_seed)


# # load image and then run prediction
# image = cv2.imread('/home/fanyix/Downloads/pic/car.jpg') # imread returns BGR output
# predictions = demo.run_on_opencv_image(image)
# predictions = predictions[:, :, ::-1]
# imsave('res.png', predictions) # imsave is assuming RGB input


# # detect in a directory
# img_list = glob.glob(os.path.join(args.data_dir, '*.%s' % args.postfix))
# for i in range(0, min(len(img_list), args.img_num)):
#     img_path = img_list[i]
#     image = cv2.imread(img_path) # imread returns BGR output
#     predictions = demo.run_on_opencv_image(image)
#     predictions = predictions[:, :, ::-1]
#     save_path = os.path.join(args.vis_dir, os.path.basename(img_path).rstrip('.%s' % args.postfix) + '.png')
#     if not os.path.isdir(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     # imsave(save_path, predictions) # imsave is assuming RGB input
#     cv2.imwrite(save_path, predictions[:, :, ::-1])


# loop inside a dataset 
vid_list = glob.glob(args.data_dir + '/*')
rand_idx = np.arange(len(vid_list))
np.random.shuffle(rand_idx)
vid_list = np.array(vid_list)[rand_idx[0:args.vid_num]].tolist()
im_list, im_cap = fu.initHTML(len(vid_list), args.frame_per_vid)
for vid_cursor, vid_path in enumerate(vid_list):
    vid_name = os.path.basename(vid_path)
    frame_list = glob.glob(vid_path + '/*.%s' % args.postfix)
    # frame_idx = [int(os.path.basename(x).rstrip('.' + args.postfix)) for x in frame_list]
    rand_idx = np.arange(len(frame_list))
    np.random.shuffle(rand_idx)
    rand_idx = rand_idx[0: min(args.frame_per_vid, rand_idx.size)]
    for frame_cursor, idx in enumerate(rand_idx.tolist()):
        frame_name = vid_name + '_' + os.path.basename(frame_list[idx]).rstrip('.%s' % args.postfix)
        image = cv2.imread(frame_list[idx]) # imread returns BGR output
        predictions = demo.run_on_opencv_image(image)
        save_path = os.path.join(args.vis_dir, 'imgs', frame_name + '.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, predictions)
        im_list[vid_cursor][frame_cursor] = os.path.relpath(save_path, args.vis_dir)
        im_cap[vid_cursor][frame_cursor] = frame_name
    print('%d/%d' % (vid_cursor, len(vid_list)))
      
html_path = os.path.join(args.vis_dir, 'vis.html')

for i in range(0, len(im_list)):
    im_list[i] = sorted(im_list[i])
    im_cap[i] = sorted(im_cap[i])

fu.writeHTML(html_path, im_list, im_cap)
print('Done.')


