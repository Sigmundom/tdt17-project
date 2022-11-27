from .base_config import *
import torch
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator

def create_model():
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
    return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
        anchor_generator=anchor_generator
    )

def create_optimizer(params):
    return torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)

OUT_DIR = 'outputs/mobilenet'
