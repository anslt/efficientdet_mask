import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .retinanet_loss import make_retinanet_loss_evaluator
from .anchor_generator import make_anchor_generator_retinanet
from .retinanet_infer import  make_retinanet_postprocessor
from .retinanet_detail_infer import  make_retinanet_detail_postprocessor
from ..backbone.efficientdet import Regressor, Classifier
from maskrcnn_benchmark.efficientdet.utils import Anchors
from maskrcnn_benchmark.efficientdet.config import COCO_CLASSES
import logging


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, num_anch):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.RETINANET.NUM_CLASSES - 1
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        num_anchors = len(cfg.RETINANET.ASPECT_RATIOS) \
                        * cfg.RETINANET.SCALES_PER_OCTAVE

        if cfg.RETINANET.FEW_ANCHOR:
             num_anchors = num_anch

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if cfg.MODEL.USE_GN:
                cls_tower.append(nn.GroupNorm(32, in_channels))

            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if cfg.MODEL.USE_GN:
                bbox_tower.append(nn.GroupNorm(32, in_channels))

            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels,  num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
                  self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                if isinstance(l, nn.GroupNorm):
                    torch.nn.init.constant_(l.weight, 1.0)
                    torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.RETINANET.PRIOR_PROB
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and RPN
    proposals and losses.
    """

    def __init__(self, cfg):
        super(RetinaNetModule, self).__init__()

        self.logger = logging.getLogger("maskrcnn_benchmark.trainer")
        self.cfg = cfg.clone()

        # TODO: to be commented out
        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg, anchor_generator.num_anchors_per_location()[0])

        #box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        box_coder = BoxCoder(weights=(1., 1., 1.,1.))

        if self.cfg.MODEL.SPARSE_MASK_ON:
            box_selector_test = make_retinanet_detail_postprocessor(
                cfg, 100, box_coder)
        else:
            box_selector_test = make_retinanet_postprocessor(
                cfg, 100, box_coder)
        box_selector_train = None
        if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.SPARSE_MASK_ON:
            box_selector_train = make_retinanet_postprocessor(
                cfg, 100, box_coder)

        # TODO: change to Yet-Another-EfficientNet FocalLoss()
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.compound_coef = cfg.EFFICIENTNET.COEF
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.num_scales = len([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        num_anchors = len(self.aspect_ratios) * self.num_scales
        num_classes = cfg.RETINANET.NUM_CLASSES - 1

        # TODO: to be commented out
        self.anchor_generator = anchor_generator
        self.head = head

        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef],
                                   num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef],
                                     num_anchors=num_anchors, num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])
        self.anchors = Anchors(anchor_scale=self.anchor_scale[self.compound_coef])
        self.box_selector_test = box_selector_test
        self.box_selector_train = box_selector_train
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # TODO： convert between format
        print("--------------FROM YET ANOTHER---------------")
        box_regression = self.regressor(features) # a list of feature maps
        box_cls = self.classifier(features) #  torch.cat(list of feature maps, dim=1)
        # anchors = self.anchors(images, images.dtype) # a numpy array with shape [N, 4], which stacks anchors on all feature levels.
        print(box_cls, box_regression)
        print("--------------FROM RETINAMASK---------------")
        box_cls, box_regression = self.head(features)# a list of feature maps (tensors)
        anchors = self.anchor_generator(images, features) # [[BoxList]] a list of list of BoxList
        # print(box_cls, box_regression, anchors)
        print(images[0])
        print(images[0].dtype)

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
            # return self._forward_train(anchors, box_cls, box_regression, targets, images)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        # TODO: change the loss to Yet-Another-EfficientDet if necessay
        # TODO: target format may not be compatible, what are "annotations, imgs and obj_list?
        # loss_box_cls, loss_box_reg = self.criterion(
        #     box_cls, box_regression, anchors, annotations, imgs=imgs, obj_list=obj_list
        # )
        # loss_box_cls, loss_box_reg = self.criterion(
        #     box_cls, box_regression, anchors, targets, imgs=images, obj_list=COCO_CLASSES
        # )
        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets
        )

        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        detections = None
        if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.SPARSE_MASK_ON:
            with torch.no_grad():
                detections = self.box_selector_train(
                    anchors, box_cls, box_regression
                )
                # boxlists (list[BoxList]): the post-processed anchors, after
                # applying box decoding and NMS
        return (anchors, detections), losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        '''
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        '''
        return (anchors, boxes), {}


def build_retinanet(cfg):
    return RetinaNetModule(cfg)
