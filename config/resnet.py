from .base_config import *
from torch import nn, optim
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, FasterRCNN

def create_model():
    backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=5, norm_layer=nn.BatchNorm2d)
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        box_nms_thresh=0.3,
        box_detections_per_img=10,
    )
    return model

def create_optimizer(params):
    return optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

OUT_DIR = 'outputs/resnet'
