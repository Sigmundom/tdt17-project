from config import IM_WIDTH
import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights
# from torchvision.models import MobileNet_V3_Large_Weights
def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights_backbone=ResNet50_Weights.DEFAULT,
            max_size=IM_WIDTH,
            num_classes=num_classes,
            box_nms_thresh=0.3,
            box_detections_per_img=10
            )
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    #         weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
    #         max_size=IM_WIDTH,
    #         num_classes=num_classes,
    #         box_nms_thresh=0.3,
    #         box_detections_per_img=10
    #         )
    
    # get the number of input features 
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # # # define a new head for the detector with required number of classes
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

if __name__=='__main__':
    model = create_model(4)
    model.eval()