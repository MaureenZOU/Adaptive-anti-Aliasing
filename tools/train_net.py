# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
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


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # FYXTODO: hijack cfg
    args.rand_seed = 777 # 777
    # args.config_file = "configs/gn_baselines/scratch_e2e_mask_rcnn_R_50_FPN_3x_gn.yaml"
    args.config_file = "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"
    args.opts += ['OUTPUT_DIR', '../../data/output/Baseline-101-COCO_bn_pasa_group_softmax']
    args.opts += ['SOLVER.IMS_PER_BATCH', 2] # 16
    args.opts += ['MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN', 2000] # 16
    args.opts += ['SOLVER.CHECKPOINT_PERIOD', 5000] # 2500
    args.opts += ['SOLVER.BASE_LR', 0.01] # 0.02
    args.opts += ['SOLVER.MAX_ITER', 180000] # 270000
    args.opts += ['SOLVER.STEPS', '(120000, 160000)'] # (210000, 250000)
    args.opts += ['INPUT.MIN_SIZE_TRAIN', (800,)] # 800
    args.opts += ['INPUT.MAX_SIZE_TRAIN', 1333] # 1333
    args.opts += ['MODEL.RESNETS.STRIDE_IN_1X1', False] # 16
    args.opts += ['MODEL.RESNETS.BACKBONE_OUT_CHANNELS', 256] # 256
    args.opts += ['MODEL.ROI_BOX_HEAD.NUM_CLASSES', 81] # VIS: 22, COCO: 81, YoutubeVOS: 29, COCO+ImageNetVID: 95, COCO+YoutubeVOS: 117
    args.opts += ['DATALOADER.NUM_WORKERS', 8] # 4

    # args.opts += ['MODEL.WEIGHT', '../../data/output/Baseline-101-COCO_bn_pasa_group_softmax/model_0005000.pth']
    args.opts += ['MODEL.WEIGHT', '../../data/checkpoints/pretrained/resnet101_pasa_group_softmax.pth']

    # Blur Downsample
    args.opts += ['MODEL.BLUR_DOWN', 'pasa_group_softmax'] # 'pasa', 'cd', 'lpf', 'false', 'lpf_maxsigma', 'lpf_correct', 'lpf_lspa'
    args.opts += ['MODEL.PASA_GROUP', 2] # 1
    args.opts += ['MODEL.PASA_FILTER', 3] # 1

    # normalization
    args.opts += ['INPUT.PIXEL_MEAN', [0.485, 0.456, 0.406]] # [102.9801, 115.9465, 122.7717]
    args.opts += ['INPUT.PIXEL_STD', [0.229, 0.224, 0.225]] # [1., 1., 1.]
    args.opts += ['INPUT.TO_BGR255', False] # False

    # Test Argument
    args.opts += ['SOLVER.TEST_CHECKPOINT_PERIOD', 1] # 10000
    args.opts += ['SOLVER.TEST_IN_TRAIN', False]
    args.opts += ['DATASETS.TEST', ("coco_2014_minival",)] # coco_2014_minival, coco_2014_minival_shift
    args.opts += ['TEST.IMS_PER_BATCH', 8] # 16
    args.opts += ['INPUT.MIN_SIZE_TEST', 800] # 800
    args.opts += ['INPUT.MAX_SIZE_TEST', 1333] # 1333
    args.opts += ['TEST.INFERENCE_TYPE', 'map_coco'] # 'pmk_coco', 'pmk_yolact, 'map_yolact', 'map_coco'

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
