
# import _init_paths
import os
import re
import numpy as np
import math
import time
from PIL import Image
from scipy import misc
from skimage.draw import line_aa
import warnings
import torch
import torch.nn.functional as F


def makeColorwheel():

    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3]) # r g b

    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255;
    col += YG;

    #GC
    colorwheel[col:GC+col, 1]= 255 
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC;

    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB;

    #BM
    colorwheel[col:BM+col, 2]= 255 
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM;

    #MR
    colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return     colorwheel


def computeColor(u, v):
    colorwheel = makeColorwheel();
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v) 

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0 
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)     # 1, 2, ..., ncols
    k1 = k0+1;
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)

def flow_to_grid(flow):
    # flow: [N, 2, H, W]
    N, _, H, W = flow.size()
    flow_x = flow[:, 0, ...]
    flow_y = flow[:, 1, ...]
    flow_x = flow_x / float(W) * 2.0
    flow_y = flow_y / float(H) * 2.0
    y, x = torch.meshgrid([torch.linspace(-1, 1, steps=H), torch.linspace(-1, 1, steps=W)])
    y, x = y[None, ...], x[None, ...]
    y = y.to(flow.device)
    x = x.to(flow.device)
    grid = torch.zeros(N, H, W, 2).to(flow.device)
    grid[..., 0] = x + flow_x
    grid[..., 1] = y + flow_y
    return grid
    
def resize_flow(flow, size):
    # flow: [N, 2, H, W]
    # size: [h, w]
    h, w = size
    fh, fw = flow.size(-2), flow.size(-1)
    flow = F.interpolate(flow, [h, w], mode='nearest')
    flow[..., 0, :, :] = flow[..., 0, :, :] * w / fw
    flow[..., 1, :, :] = flow[..., 1, :, :] * h / fh
    return flow
    
def flip_flow(flow):
    # flow: [H, W, 2]
    flow = flow[:, ::-1, :]
    flow[..., 0] = - flow[..., 0]
    return flow

def compress_flow(flow, bound):
    # input size [H, W, 2]
    flow = flow.copy()
    H, W = flow.shape[0], flow.shape[1]
    
    min_flow, max_flow = np.min(flow), np.max(flow)
    if min_flow < -bound or max_flow > bound:
        warnings.warn('Min: %.4f, Max: %.4f, out of [-%d, %d]' % (min_flow, max_flow, bound, bound))
    
    flow[..., 0] = np.round((flow[..., 0] + bound) / (2. * bound) * 255.)
    flow[..., 1] = np.round((flow[..., 1] + bound) / (2. * bound) * 255.)
    flow[flow < 0] = 0
    flow[flow > 255] = 255
    flow = np.concatenate([flow, np.zeros([H, W, 1])], axis=2)
    flow = flow.astype(np.uint8)
    return flow

def decompress_flow(flow, bound):
    # input size [H, W, 2]
    flow = flow.copy().astype(np.float32)
    H, W = flow.shape[0], flow.shape[1]
    flow[..., 0] = flow[..., 0] / 255. * 2 * bound - bound
    flow[..., 1] = flow[..., 1] / 255. * 2 * bound - bound
    flow = flow[..., :2]
    return flow

def bbox_intersection(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    area = (y_bottom - y_top + 1) * (x_right - x_left + 1)
    return area

def bbox_transform(ex_rois, gt_rois):
    reshaped = False
    if ex_rois.ndim == 1 or gt_rois.ndim == 1:
        ex_rois = ex_rois.reshape(1, 4)
        gt_rois = gt_rois.reshape(1, 4)
        reshaped = True

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    if reshaped:
        targets = targets.reshape(-1)

    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    reshaped = False
    if boxes.ndim == 1 or deltas.ndim == 1:
        boxes = boxes.reshape(1, 4)
        deltas = deltas.reshape(1, 4)
        reshaped = True

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    if reshaped:
        pred_boxes = pred_boxes.reshape(-1)

    return pred_boxes


class Args():
    def __init__(self):
        assert True

def rand_range(a, b):
    x = np.random.rand()
    y = x * (b - a) + a
    return y

def relative_coord(box, ref_box, size):
    # get height and width
    hgt = size[0]
    wid = size[1]
    ref_wid = ref_box[2] - ref_box[0]
    ref_hgt = ref_box[3] - ref_box[1]
    # compute the relative coords
    x1 = float(box[0]-ref_box[0]) / ref_wid * wid
    y1 = float(box[1]-ref_box[1]) / ref_hgt * hgt
    x2 = float(box[2]-ref_box[0]) / ref_wid * wid
    y2 = float(box[3]-ref_box[1]) / ref_hgt * hgt
    rel_box = np.array([x1, y1, x2, y2])
    return rel_box

def crop_patch(I, box, pad_color):
    img_wid = I.size[0]
    img_hgt = I.size[1]
    clip_box = calibrate_box(box, img_wid, img_hgt)  
    clip_crop = I.crop(clip_box)
    R = pad_color[0]
    G = pad_color[1]
    B = pad_color[2]
    wid = int(box[2] - box[0])
    hgt = int(box[3] - box[1])
    frame_size = [wid, hgt]
    offset_x = int(max(-box[0], 0))
    offset_y = int(max(-box[1], 0))
    offset_tuple = (offset_x, offset_y) #pack x and y into a tuple
    final_crop = Image.new(mode='RGB',size=frame_size,color=(R,G,B))
    final_crop.paste(clip_crop, offset_tuple)
    return final_crop


def rand_crop(box, min_ratio):
    # crop a patch out from a box, that is at least min_ratio*size large
    wid = box[2] - box[0] + 1
    hgt = box[3] - box[1] + 1
    ratio = rand_range(min_ratio, 1.0)
    crop_wid = int(wid * ratio)
    crop_hgt = int(hgt * ratio)

    # x1
    low = int(box[0])
    high = int(box[2] - crop_wid + 1)
    if high > low:
        x1 = np.random.randint(low, high)
    else:
        x1 = low

    # y1
    low = int(box[1])
    high = int(box[3] - crop_hgt + 1)
    if high > low:
        y1 = np.random.randint(low, high)
    else:
        y1 = low

    x2 = x1 + crop_wid - 1
    y2 = y1 + crop_hgt - 1
    crop_box = np.array([x1, y1, x2, y2])
    return crop_box


def expand_box(box, ratio):
    wid = box[2] - box[0]
    hgt = box[3] - box[1]
    x_center = (box[2] + box[0]) / 2.0
    y_center = (box[3] + box[1]) / 2.0
    context_wid = wid * ratio
    context_hgt = hgt * ratio
    x1 = x_center - context_wid / 2.0
    x2 = x_center + context_wid / 2.0
    y1 = y_center - context_hgt / 2.0
    y2 = y_center + context_hgt / 2.0
    context_box = np.array([x1, y1, x2, y2])
    return context_box

def box_rel_to_abs(box, wid, hgt):
    if box.ndim == 1:
        box[0] = box[0] * wid
        box[2] = box[2] * wid
        box[1] = box[1] * hgt
        box[3] = box[3] * hgt
    else:
        box[:, 0] = box[:, 0] * wid
        box[:, 2] = box[:, 2] * wid
        box[:, 1] = box[:, 1] * hgt
        box[:, 3] = box[:, 3] * hgt
    return box

def sigmoid(x):
    y = 1.0 / (1 + np.exp(-x))
    return y

def read_or_block(filename):
    while True:
        if os.path.isfile(filename):
            break
        time.sleep(5)

    # we had the file now
    time.sleep(5)
    res = np.load(filename)
    return res

def mkdir_imwrite(fig2, img_path):
    path, filename = os.path.split(img_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    fig2.savefig(img_path)


def initHTML(row_n, col_n):
    im_paths = [['NA'] * col_n for idx in range(row_n)]
    captions = [['NA'] * col_n for idx in range(row_n)]
    return im_paths, captions

def writeHTML(file_name, im_paths, captions, height=200, width=200):
    f=open(file_name, 'w')
    html=[]
    f.write('<!DOCTYPE html>\n')
    f.write('<html><body>\n')
    f.write('<table>\n')
    for row in range(len(im_paths)):
        f.write('<tr>\n')
        for col in range(len(im_paths[row])):
            f.write('<td>')
            f.write(captions[row][col])
            f.write('</td>')
            f.write('    ')
        f.write('\n</tr>\n')

        f.write('<tr>\n')
        for col in range(len(im_paths[row])):
            f.write('<td><img src="')
            f.write(im_paths[row][col])
            f.write('" height='+str(height)+' width='+str(width)+'"/></td>')
            f.write('    ')
        f.write('\n</tr>\n')
        f.write('<p></p>')
    f.write('</table>\n')
    f.close()

def writeSeqHTML(file_name, im_paths, captions, col_n, height=200, width=200):
    total_n = len(im_paths)
    row_n = int(math.ceil(float(total_n) / col_n))
    f=open(file_name, 'w')
    html=[]
    f.write('<!DOCTYPE html>\n')
    f.write('<html><body>\n')
    f.write('<table>\n')
    for row in range(row_n):
        base_count = row * col_n
        f.write('<tr>\n')
        for col in range(col_n):
            if base_count + col < total_n:
                f.write('<td>')
                f.write(captions[base_count + col])
                f.write('</td>')
                f.write('    ')
        f.write('\n</tr>\n')

        f.write('<tr>\n')
        for col in range(col_n):
            if base_count + col < total_n:
                f.write('<td><img src="')
                f.write(im_paths[base_count + col])
                f.write('" height='+str(height)+' width='+str(width)+'"/></td>')
                f.write('    ')
        f.write('\n</tr>\n')
        f.write('<p></p>')
    f.write('</table>\n')
    f.close()

def flip_box(box, wid):
    flipped_box = box.copy()
    if flipped_box.ndim == 1:
        start = flipped_box[0]
        flipped_box[0] = wid - flipped_box[2]
        flipped_box[2] = wid - start
    else:
        start = flipped_box[:, 0].copy()
        flipped_box[:, 0] = wid - flipped_box[:, 2]
        flipped_box[:, 2] = wid - start
    return flipped_box

def normalize_coord(x, size):
    return float(x) / size * 2 - 1

def shear_and_rotate(shr=0.1, rot=math.pi/4):
    sh_x = rand_range(-shr,shr)
    sh_y = rand_range(-shr,shr)
    sh_theta = np.array([1,sh_y,0,
                         sh_x,1,0,
                         0,0,1]).reshape(3, 3)
    rot_angle = rand_range(-rot,rot)
    cos = math.cos(rot_angle)
    sin = math.sin(rot_angle)
    rot_theta = np.array([cos,sin,0,
                          -sin,cos,0,
                          0,0,1]).reshape(3, 3)
    theta = np.matmul(rot_theta, sh_theta)
    return theta

def box_to_theta(box, im_wid, im_hgt):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    # compute the baseline theta, which gives us exactly the box
    norm_x1 = normalize_coord(x1, im_wid)
    norm_x2 = normalize_coord(x2, im_wid)
    norm_y1 = normalize_coord(y1, im_hgt)
    norm_y2 = normalize_coord(y2, im_hgt)
    half_wid = (norm_x2 - norm_x1) / 2
    x_center = (norm_x2 + norm_x1) / 2
    half_hgt = (norm_y2 - norm_y1) / 2
    y_center = (norm_y2 + norm_y1) / 2
    theta = np.array([half_wid,0,x_center,0,half_hgt,y_center,0,0,1], dtype=np.float).reshape(3, 3)
    return theta, half_wid, half_hgt, x_center, y_center

        
def relative_path(ref_path, target_path):
    # common_prefix = os.path.commonprefix([ref_path, target_path])
    return os.path.relpath(target_path, ref_path)


def check_tokens(word1, word2):
    match = 0
    for counter1, token1 in enumerate(word1[0]):
        for counter2, token2 in enumerate(word2[0]):
            if pattern.search(word1[1][counter1]) != None and \
                            pattern.search(word2[1][counter2]) != None and \
                            stemmer.stem(token1) == stemmer.stem(token2):
                match += 1
    return match

def shape2str(shape):
    str = ''
    for idx, i in enumerate(shape):
        if idx == len(shape)-1:
            str += '%d' % i
        else:
            str += '%d,' % i
    return str

def calibrate_box(box, wid, hgt):
    new_box = box.copy().astype(np.int)
    if box.ndim == 1:
        new_box[0] = max(round(box[0]), 0)
        new_box[1] = max(round(box[1]), 0)
        new_box[2] = min(round(box[2]), wid-1)
        new_box[3] = min(round(box[3]), hgt-1)
    elif box.ndim == 2:
        new_box[:, 0] = np.maximum(np.round(box[:, 0]), 0)
        new_box[:, 1] = np.maximum(np.round(box[:, 1]), 0)
        new_box[:, 2] = np.minimum(np.round(box[:, 2]), wid-1)
        new_box[:, 3] = np.minimum(np.round(box[:, 3]), hgt-1)
    return new_box

def softmax(w):
    maxes = np.amax(w, axis=1)
    maxes = np.tile(maxes[:, np.newaxis], [1, w.shape[1]])
    e = np.exp(w - maxes)
    dist = e / np.tile(np.sum(e, axis=1)[:, np.newaxis], [1, w.shape[1]])
    return dist

def truncate(annot, num):
    new_annot = {}
    annot_keys = annot.keys()
    for idx in range(num):
        key = annot_keys[idx]
        new_annot[key] = annot[key]
    return new_annot

def mkdir_imwrite(fig2, img_path):
    path, filename = os.path.split(img_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    fig2.savefig(img_path, bbox_inches='tight', pad_inches=0)

def unique_row(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    ui = order[ui]
    return ui

def ismember(a, b, bind = None):
    if bind is None:
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
    return (np.array([bind.get(itm, -1) for itm in a]), bind)  # None can be replaced by any other "not in b" value


def get_data_base(arr):
    """For a given Numpy array, finds the
    base array that "owns" the actual data."""
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base

def arrays_share_data(x, y):
    return get_data_base(x) is get_data_base(y)


v = 1.0
s = 1.0
p = 0.0
def rgbcolor(h, f):
    """Convert a color specified by h-value and f-value to an RGB
    three-tuple."""
    # q = 1 - f
    # t = f
    if h == 0:
        return v, f, p
    elif h == 1:
        return 1 - f, v, p
    elif h == 2:
        return p, v, f
    elif h == 3:
        return p, 1 - f, v
    elif h == 4:
        return f, p, v
    elif h == 5:
        return v, p, 1 - f

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def uniquecolors(n):
    """Compute a list of distinct colors, ecah of which is
    represented as an RGB three-tuple"""
    hues = [360.0 / n * i for i in range(n)]
    hs = [math.floor(hue / 60) % 6 for hue in hues]
    fs = [hue / 60 - math.floor(hue / 60) for hue in hues]
    return [rgbcolor(h, f) for h, f in zip(hs, fs)]

def heatmap_calib(map):
    # this only works for numpy
    minval = np.min(map)
    maxval = np.max(map)
    gap = (maxval - minval + 1e-8)
    # linear interpolation
    map = ((map - minval) / gap)
    return map

def tree_to_list(t):
    # convert tree to list
    if isinstance(t, Tree):
        return [t.label()] + map(tree_to_list, t)
    else:
        return t

## functions for operating a loss recorder
def init_recorder(T):
    recorder = {'smoothed_loss_arr' : [], 'raw_loss_arr' : [], 'loss_iter_arr': [], 'ptr' : 0, 'T' : T}
    return recorder

def retrieve_loss(struct, start_round):
    loss, iter = struct['smoothed_loss_arr'], struct['loss_iter_arr']
    return loss, iter

def update_loss(struct, loss, iter):
    raw_loss_arr = struct['raw_loss_arr']
    smoothed_loss_arr = struct['smoothed_loss_arr']
    loss_iter_arr = struct['loss_iter_arr']
    T = struct['T']
    ptr = struct['ptr']
    if len(smoothed_loss_arr) > 0:
      smoothed_loss = smoothed_loss_arr[-1]
    else:
      smoothed_loss = 0
    cur_len = len(raw_loss_arr)
    if cur_len < T:
      smoothed_loss = (smoothed_loss * cur_len + loss) / (cur_len + 1)
      raw_loss_arr.append(loss)
    else:
      smoothed_loss = smoothed_loss + (loss - raw_loss_arr[ptr]) / T
      raw_loss_arr[ptr] = loss
    ptr = (ptr + 1) % T
    smoothed_loss_arr.append(smoothed_loss)
    loss_iter_arr.append(iter)
    # stuff info into struct
    struct['ptr'] = ptr
    struct['raw_loss_arr'] = raw_loss_arr
    struct['smoothed_loss_arr'] = smoothed_loss_arr
    struct['loss_iter_arr'] = loss_iter_arr
    return struct

def vis_link(src, tgt, links):
    # visualize the link between src and tgt
    pic = []
    N = src.size(0)
    
    # ship all data to cpu
    src = src.cpu().numpy()
    tgt = tgt.cpu().numpy()
    
    # loop
    out = []
    for idx in range(N):
        shift = src[idx].shape[2]
        whole = np.concatenate((src[idx], tgt[idx]), axis=2)
        link = links[idx]
        for pair_idx in range(link.size(1)):
            src_x, src_y, tgt_x, tgt_y = link[:, pair_idx]
            tgt_x += shift
            rr, cc, val = line_aa(src_y, src_x, tgt_y, tgt_x)
            # val = np.tile(val.reshape(1, -1), (3, 1))
            val = np.tile(np.array([0,0,1]).reshape(3, 1), (1, cc.shape[0]))
            whole[:, rr, cc] = val
        out.append(whole)
    
    return out
    
    
    
    
    
    
    
    
    
    
    
