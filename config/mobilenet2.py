from .base_config import *
import torch
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

def create_model():
    # anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
    backbone = mobilenet_backbone(
            backbone_name='mobilenet_v3_large', 
            weights=MobileNet_V3_Large_Weights, 
            fpn=True,
            trainable_layers=4
            )
    
    model = FasterRCNN(
            backbone=backbone,
            num_classes=NUM_CLASSES,
            rpn_anchor_generator=rpn_anchor_generator,
            box_nms_thresh=0.3,
            box_detections_per_img=10,
        )
    return model

def create_optimizer(params):
    return torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)

OUT_DIR = 'outputs/mobilenet2'
