# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import numpy as np
from collections import OrderedDict

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, _, _, _ = batch

        # DEBUG
        # from scipy.misc import imsave
        # import numpy as np
        # tmp = (images.tensors[0].permute(1,2,0).cpu().numpy()*np.array([0.229, 0.224, 0.225]).reshape(1,1,3) + np.array([0.485, 0.456, 0.406]).reshape(1,1,3))*255
        # imsave("img.png", tmp)

        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

    return results_dict

def compute_on_coco_pmk_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, seq_ids, sft_ids, _ = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

    return results_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def compute_on_vid_dataset(model, data_loader, device, output_dir, cfg, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
        
    # init AP object
    iou_thresholds = [x / 100 for x in range(50, 100, 5)]
    CLASSES = list(data_loader.dataset.cls_to_idx.keys())
    ap_data = {
        'box' : [[APDataObject() for _ in CLASSES] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in CLASSES] for _ in iou_thresholds]
    }
    
    masker = Masker(threshold=0.5, padding=1)

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, _, _, img_sizes = batch

        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                outputs = im_detect_bbox_aug(model, images, device)
            else:
                outputs = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            outputs = [o.to(cpu_device) for o in outputs]

        for output, target, img_size in zip(outputs, targets, img_sizes):
            prep_metrics(ap_data, output, target, img_size[1], img_size[0], iou_thresholds, masker, mask_on=cfg.MODEL.MASK_ON) # h, w
    
    return ap_data, iou_thresholds, CLASSES

def compute_on_pmk_dataset(model, data_loader, device, output_dir, cfg, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
        
    # init AP object
    iou_thresholds = [x / 100 for x in range(50, 100, 5)]
    CLASSES = list(data_loader.dataset.cls_to_idx.keys())
    seq_num = data_loader.dataset.seq_num
    shift_num = data_loader.dataset.shift_num

    ap_data = {
        'box' : [[[[APDataObject() for _ in seq_num] for _ in shift_num] for _ in CLASSES] for _ in iou_thresholds],
        'mask': [[[[APDataObject() for _ in seq_num] for _ in shift_num] for _ in CLASSES] for _ in iou_thresholds]
    }
    
    masker = Masker(threshold=0.5, padding=1)

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, seq_ids, sft_ids, img_sizes = batch

        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                outputs = im_detect_bbox_aug(model, images, device)
            else:
                outputs = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            outputs = [o.to(cpu_device) for o in outputs]
        
        for output, target, img_size, seq_id, sft_id in zip(outputs, targets, img_sizes, seq_ids, sft_ids):
            prep_metrics_pmk(ap_data, output, target, img_size[1], img_size[0], iou_thresholds, seq_id, sft_id, masker, mask_on=cfg.MODEL.MASK_ON) # h, w
    
    return ap_data, iou_thresholds, CLASSES


def inference_yolact(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    
    ap_data, iou_thresholds, CLASSES = compute_on_vid_dataset(model, data_loader, device, output_folder, cfg, timer=inference_timer)
    res = calc_map(ap_data, iou_thresholds, CLASSES)
    res_str = print_maps(res)

    print(res_str)

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    return res, res_str

def inference_yolact_pmk(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    
    ap_data, iou_thresholds, CLASSES = compute_on_vid_dataset(model, data_loader, device, output_folder, cfg, timer=inference_timer)
    res = calc_map(ap_data, iou_thresholds, CLASSES)
    res_str = print_maps(res)

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    return res, res_str

def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(cfg = cfg,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

def inference_coco_pmk(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_coco_pmk_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(cfg=cfg,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def extract(dets, h, w, masker):
    dets = dets.resize((w, h))
    boxes = dets.bbox
    scores = dets.get_field('scores')
    masks = dets.get_field('mask')
    if list(masks.shape[-2:]) != [h, w]:
        masks = masker(masks.expand(1, -1, -1, -1, -1), dets)
        masks = masks[0]
    masks = masks.squeeze(dim=1).type(torch.int64)
    classes = dets.get_field('labels')
    return classes, scores, boxes, masks
    
def prep_metrics_pmk(ap_data, dets, target, h, w, iou_thresholds, seq_id, sft_id, masker, mask_on):
    """ Returns a list of APs for this image, with each element being for a class  """
    
    # first decide whether use mask in practice
    mask_on = mask_on and 'masks' in target.fields()
    
    # process target
    target = target.resize((w, h))
    gt_boxes = target.bbox
    gt_classes = target.get_field('labels').type(torch.int64).tolist()
    if mask_on:
        if len(gt_classes) > 0:
            assert 'masks' in target.fields()
            gt_masks = [x.convert_to_binarymask()[None, ...] for x in target.get_field('masks').instances]
            gt_masks = torch.cat(gt_masks, 0)
            gt_masks = gt_masks.view(-1, h*w).type(torch.int64)
            # gt masks is sure to be correct
        else:
            gt_masks = torch.zeros(0, h*w).type(torch.int64)
        assert gt_masks.size(0) == gt_boxes.size(0)
        
    # process dets
    classes, scores, boxes, masks = extract(dets, h, w, masker)

    classes = classes.cpu().numpy().astype(int).tolist()
    scores = scores.cpu().numpy().astype(float).tolist()
    masks = masks.view(-1, h*w)

    num_pred = len(classes)
    num_gt   = len(gt_classes)

    
    if mask_on:
        iou_types = ['box', 'mask']
    else:
        iou_types = ['box']
        
    if num_gt == 0 and num_pred == 0:
        return
    elif num_gt == 0:
        for iou_type in iou_types:
            for iouIdx in range(len(iou_thresholds)):    
                for i in range(num_pred):
                    _class = classes[i]
                    ap_obj = ap_data[seq_id][sft_id][iou_type][iouIdx][_class-1]
                    ap_obj.push(scores[i], False)
        return 
    elif num_pred == 0:
        for iou_type in iou_types:
            for iouIdx in range(len(iou_thresholds)):    
                for j in range(num_gt):
                    _class = gt_classes[j]
                    ap_obj = ap_data[seq_id][sft_id][iou_type][iouIdx][_class-1] 
                    ap_obj.add_gt_positives(1)
        return

    if mask_on:
        bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())
        mask_iou_cache = mask_iou(masks.float(), gt_masks.float())
        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item()),
            ('mask', lambda i,j: mask_iou_cache[i, j].item()),
        ]
    else:
        bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())
        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item()),
        ]

    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func in iou_types:
                gt_used = [False] * len(gt_classes)
                
                # class-1 because there is no background entry in ap_data
                ap_obj = ap_data[iou_type][iouIdx][_class-1] 
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in range(num_pred):
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)
    
                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(scores[i], True)

                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(scores[i], False)

def prep_metrics(ap_data, dets, target, h, w, iou_thresholds, masker, mask_on):
    """ Returns a list of APs for this image, with each element being for a class  """
    
    # first decide whether use mask in practice
    mask_on = mask_on and 'masks' in target.fields()
    
    # process target
    target = target.resize((w, h))
    gt_boxes = target.bbox
    gt_classes = target.get_field('labels').type(torch.int64).tolist()
    if mask_on:
        if len(gt_classes) > 0:
            assert 'masks' in target.fields()
            gt_masks = [x.convert_to_binarymask()[None, ...] for x in target.get_field('masks').instances]
            gt_masks = torch.cat(gt_masks, 0)
            gt_masks = gt_masks.view(-1, h*w).type(torch.int64)
            # gt masks is sure to be correct
        else:
            gt_masks = torch.zeros(0, h*w).type(torch.int64)
        assert gt_masks.size(0) == gt_boxes.size(0)
        
    # process dets
    classes, scores, boxes, masks = extract(dets, h, w, masker)

    classes = classes.cpu().numpy().astype(int).tolist()
    scores = scores.cpu().numpy().astype(float).tolist()
    masks = masks.view(-1, h*w)

    num_pred = len(classes)
    num_gt   = len(gt_classes)
    
    if mask_on:
        iou_types = ['box', 'mask']
    else:
        iou_types = ['box']
        
    if num_gt == 0 and num_pred == 0:
        return
    elif num_gt == 0:
        for iou_type in iou_types:
            for iouIdx in range(len(iou_thresholds)):    
                for i in range(num_pred):
                    _class = classes[i]
                    ap_obj = ap_data[iou_type][iouIdx][_class-1] 
                    ap_obj.push(scores[i], False)
        return 
    elif num_pred == 0:
        for iou_type in iou_types:
            for iouIdx in range(len(iou_thresholds)):    
                for j in range(num_gt):
                    _class = gt_classes[j]
                    ap_obj = ap_data[iou_type][iouIdx][_class-1] 
                    ap_obj.add_gt_positives(1)
        return

    if mask_on:
        bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())
        mask_iou_cache = mask_iou(masks.float(), gt_masks.float())
        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item()),
            ('mask', lambda i,j: mask_iou_cache[i, j].item()),
        ]
    else:
        bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())
        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item()),
        ]

    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func in iou_types:
                gt_used = [False] * len(gt_classes)
                
                # class-1 because there is no background entry in ap_data
                ap_obj = ap_data[iou_type][iouIdx][_class-1] 
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in range(num_pred):
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)
    
                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(scores[i], True)

                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(scores[i], False)


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0
        # return self.num_gt_positives == 0 # match coco

    def get_ap(self) -> float:
        """ Warning: result not cached. """
        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def calc_map(ap_data, iou_thresholds, CLASSES):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(CLASSES)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    return all_maps


def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)
    
    str = ''
    
    str += make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()])
    str += '\n'
    
    str += make_sep(len(all_maps['box']) + 1)
    str += '\n'
    
    for iou_type in ('box', 'mask'):
        str += make_row([iou_type] + ['%.2f' % x for x in all_maps[iou_type].values()])
        str += '\n'
        
    str += make_sep(len(all_maps['box']) + 1)
    str += '\n'
        
    return str
    
def mask_iou(mask1, mask2, iscrowd=False):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    intersection = torch.matmul(mask1, mask2.t())

    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    if iscrowd:
        # Make sure to brodcast to the right dimension
        ret = intersection / area1.t()
    else:
        ret = intersection / union
    return ret.cpu()


def bbox_iou(bbox1, bbox2, iscrowd=False):
    ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()


def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd=False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)