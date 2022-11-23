import torchvision

def apply_nms(prediction, threshold):
    # torchvision returns the indices of the boxes to keep
    keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], threshold)
    final_prediction = dict(
        boxes = prediction['boxes'][keep],
        scores = prediction['scores'][keep],
        labels = prediction['labels'][keep],
    )
    
    return final_prediction