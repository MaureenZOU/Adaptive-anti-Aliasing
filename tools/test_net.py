# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference, inference_yolact, inference_coco_pmk
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # args.config_file = "configs/gn_baselines/scratch_e2e_mask_rcnn_R_50_FPN_3x_gn.yaml"
    args.config_file = "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"
    args.opts += ['OUTPUT_DIR', '../../data/output/Baseline-101-COCO_bn_pasa_group_softmax_frozen']
    # args.opts += ['MODEL.WEIGHT', '../../data/checkpoints/pretrained/resnet50_pasa.pth']
    args.opts += ['MODEL.WEIGHT', '../../data/output/Baseline-101-COCO_bn_pasa_group_softmax_frozen/model_0030000.pth']
    args.opts += ['DATASETS.TEST', ("coco_2014_minival",)] # coco_2014_minival, coco_2014_minival_shift
    args.opts += ['TEST.IMS_PER_BATCH', 3] # 16
    args.opts += ['DATALOADER.NUM_WORKERS', 0] # 4
    args.opts += ['INPUT.MIN_SIZE_TEST', 800] # 800
    args.opts += ['INPUT.MAX_SIZE_TEST', 1333] # 1333
    args.opts += ['MODEL.RESNETS.BACKBONE_OUT_CHANNELS', 256] # 256
    args.opts += ['MODEL.ROI_BOX_HEAD.NUM_CLASSES', 81] # COCO_VIS: 22, COCO: 81, YoutubeVOS: 29, COCO+ImageNetVID: 95, COCO+YoutubeVOS: 117

    # inference type
    args.opts += ['TEST.INFERENCE_TYPE', 'map_coco'] # 'pmk_coco', 'pmk_yolact, 'map_yolact', 'map_coco'

    # Blur Downsample
    args.opts += ['MODEL.BLUR_DOWN', 'pasa_group_softmax'] # 'cd', 'lpf', 'lpf_correct', 'false', 'lpf_maxsigma', 'pasa'
    args.opts += ['MODEL.PASA_GROUP', 2] # 1
    args.opts += ['MODEL.PASA_FILTER', 3] # 1

    # normalization
    args.opts += ['INPUT.PIXEL_MEAN', [0.485, 0.456, 0.406]] # [102.9801, 115.9465, 122.7717]
    args.opts += ['INPUT.PIXEL_STD', [0.229, 0.224, 0.225]] # [1., 1., 1.]
    args.opts += ['INPUT.TO_BGR255', False] # False

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=False)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        if 'map_coco' in cfg.TEST.INFERENCE_TYPE:
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
            synchronize()
        elif 'map_yolact' in cfg.TEST.INFERENCE_TYPE:
            inference_yolact(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
        elif 'pmk_yolact' in cfg.TEST.INFERENCE_TYPE:
            inference_yolact_pmk(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
        elif 'pmk_coco' in cfg.TEST.INFERENCE_TYPE:
            inference_coco_pmk(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )

        else:
            assert False, "Inference type not implemented."

if __name__ == "__main__":
    main()
