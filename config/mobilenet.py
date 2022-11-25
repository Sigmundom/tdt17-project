from .base_config import *
import torch
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights

def create_model():
    return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
    )

OUT_DIR = 'outputs/mobilenet'
