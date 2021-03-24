# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
import torch.nn.functional as F

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
import numpy as np

class Demo(object):
    def __init__(
        self,
        cfg,
        dataset,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        # self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # import futils
        # num_colors = len(self.CATEGORIES)
        # self.palette = [torch.tensor(x)[None, :] * 255 for x in futils.uniquecolors(num_colors)]
        # self.palette = torch.cat(self.palette, 0)
        # for making bounding boxes pretty
        COLORS = ((244,  67,  54),
                  (233,  30,  99),
                  (156,  39, 176),
                  (103,  58, 183),
                  ( 63,  81, 181),
                  ( 33, 150, 243),
                  (  3, 169, 244),
                  (  0, 188, 212),
                  (  0, 150, 136),
                  ( 76, 175,  80),
                  (139, 195,  74),
                  (205, 220,  57),
                  (255, 235,  59),
                  (255, 193,   7),
                  (255, 152,   0),
                  (255,  87,  34),
                  # (121,  85,  72),
                  # (158, 158, 158),
                  # ( 96, 125, 139),
                  )
        self.palette = torch.FloatTensor(COLORS)


        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim
        
        if dataset == 'COCO':
            self.CATEGORIES = [
                "__background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush",
            ]
        elif dataset == 'YoutubeVOS':
            self.CATEGORIES = [
                "__background", 'horse', 'knife', 'sedan', 'person', 'boat', 'toilet', 'giraffe', 
                'bird', 'train', 'umbrella', 'surfboard', 'skateboard', 'frisbee', 'bear', 'airplane', 
                'cat', 'dog', 'sheep', 'tennis_racket', 'motorbike', 'truck', 'bike', 'cow', 'zebra', 
                'bus', 'elephant', 'snowboard', 'plant'
            ]
        elif dataset == 'VID':
            self.CATEGORIES = [
                "__background", 'horse', 'car', 'boat', 'bird', 'train', 'bear', 'airplane',
                'cat', 'dog', 'sheep', 'motorcycle', 'bicycle', 'cow', 'zebra',
                'bus', 'elephant'
            ]
        elif dataset == 'YoutubeVIS':
            self.CATEGORIES = [
                "__background", 'person', 'car', 'motorcycle', 'airplane', 'train', 'truck', 'boat', 'bird', 'cat', 
                'dog', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'snowboard', 'skateboard', 
                'surfboard', 'tennis racket', 'mouse'
            ]
        elif dataset == 'YoutubeVIS_full':
            self.CATEGORIES = [
                "__background", 'person', 'car', 'motorcycle', 'airplane', 'train', 'truck', 'boat', 'cat', 
                'dog', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'snowboard', 'skateboard', 
                'surfboard', 'tennis racket', 'mouse', 'rabbit', 'shark', 'hand', 'fox', 'giant_panda',
                'eagle', 'frog', 'turtle', 'earless_seal', 'monkey', 'lizard', 'tiger', 'fish', 'ape', 'owl',
                 'snake', 'duck', 'deer', 'leopard', 'parrot'
            ]

    def compute_eval_prediction(self, predictions, original_image):
        """
        Arguments:
            original_image (np.ndarray): a [Boxlist] Object of predictions of a single object
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def run_on_eval_image(self, output, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_eval_prediction([output], image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        
        labels = top_predictions.get_field("labels")
        colors = self.compute_colors_for_labels(labels).tolist()
        
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions, colors)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions, colors)
            result = result.astype(np.uint8)
        result = self.overlay_class_names(result, top_predictions, colors)

        return result

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        
        labels = top_predictions.get_field("labels")
        colors = self.compute_colors_for_labels(labels).tolist()
        
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions, colors)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions, colors)
            result = result.astype(np.uint8)
        result = self.overlay_class_names(result, top_predictions, colors)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        
        # labels = labels % self.palette.size(0)
        num_colors = self.palette.size(0)
        if num_colors < len(labels):
            repeat_num = int(len(labels)/num_colors) + 1
            self.palette = self.palette.repeat(repeat_num, 1)
        labels = torch.arange(len(labels))
        
        colors = self.palette[labels, :]
        colors = colors.numpy().astype("uint8")

        return colors

    def overlay_boxes(self, image, predictions, colors):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        # colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions, colors):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")
        # colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            mask = mask[0, :, :, None]
            # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # image = cv2.drawContours(image, contours, -1, color, 3)   
            mask_alpha = 0.5
            hgt, wid = mask.shape[0], mask.shape[1]
            mask_3c = np.broadcast_to(mask, [hgt, wid, 3])
            color_3c = np.array(color[:3]).reshape(1, 1, 3)
            mask_3c = mask_3c * np.broadcast_to(color_3c, [hgt, wid, 3])
            image = image * (1 - mask) + image * mask * (1-mask_alpha) + mask_3c * mask_alpha

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        if masks.shape[0] > 0:
            mask = masks[masks.shape[0]-1][0]
            masker = cv2.applyColorMap((mask*256).byte().numpy(), cv2.COLORMAP_JET)
            mask = mask[:,:,None].numpy()
            result = image*0.5 + (image*(1 - mask) + masker*mask)*0.5
        else:
            result = image
        return result

        # masks = predictions.get_field("mask")
        # masks_per_dim = self.masks_per_dim
        # masks = L.interpolate(
        #     masks.float(), scale_factor=1 / masks_per_dim
        # ).byte()
        # height, width = masks.shape[-2:]
        # max_masks = masks_per_dim ** 2
        # masks = masks[:max_masks]
        # # handle case where we have less detections than max_masks
        # if len(masks) < max_masks:
        #     masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
        #     masks_padded[: len(masks)] = masks
        #     masks = masks_padded
        # masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        # result = torch.zeros(
        #     (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        # )
        # for y in range(masks_per_dim):
        #     start_y = y * height
        #     end_y = (y + 1) * height
        #     for x in range(masks_per_dim):
        #         start_x = x * width
        #         end_x = (x + 1) * width
        #         result[start_y:end_y, start_x:end_x] = masks[y, x]
        # return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions, colors):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label, color in zip(boxes, scores, labels, colors):
            x, y = box[:2]
            # s = template.format(label, score)
            # cv2.putText(
            #     image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            # )
            s = '%s: %.2f' % (label, score)
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1            
            text_w, text_h = cv2.getTextSize(s, font_face, font_scale, font_thickness)[0]
            text_pt = (x, y - 3)
            text_color = [255, 255, 255]
            cv2.rectangle(image, (x, y), (x + text_w, y - text_h - 4), color, -1)
            cv2.putText(image, s, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            

        return image