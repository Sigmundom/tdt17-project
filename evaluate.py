from config import DEVICE
import torch
import tqdm
import numpy as np
from bbox_utils import bbox_ltrb_to_ltwh
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _to_cuda(element):
    return element.to(get_device(), non_blocking=True)


def to_cuda(elements):
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [_to_cuda(x) for x in elements]
    if isinstance(elements, dict):
        return {k: _to_cuda(v) for k,v in elements.items()}
    return _to_cuda(elements)


@torch.no_grad()
def evaluate(
        model,
        dataloader: torch.utils.data.DataLoader,
        cocoGt: COCO):
    """
        Evaluates over dataloader and returns COCO stats
    """
    model.eval()
    ret = []
    for images, targets in tqdm.tqdm(dataloader, desc="Evaluating on dataset"):
        images = list(image.to(DEVICE) for image in images)
        # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=True):
            predictions = model(images)
            # , nms_iou_threshold=0.50, max_output=200, score_threshold=0.05)
        for prediction, target in zip(predictions, targets):
            # prediction = apply_nms(prediction, 0.3)
            boxes_ltrb = prediction['boxes']
            categories = prediction['labels']
            scores = prediction['scores']
            # ease-of-use for specific predictions
            box_ltwh = bbox_ltrb_to_ltwh(boxes_ltrb)
            box_ltwh, category, score = [x.cpu() for x in [box_ltwh, categories, scores]]
            img_id = target["image_id"]
            for b_ltwh, label_, prob_ in zip(box_ltwh, category, score):
                #TODO! Have to make sure that label_ matches COCO label
                ret.append([img_id, *b_ltwh.tolist(), prob_.item(),
                            int(label_)])
    model.train()
    final_results = np.array(ret).astype(np.float32)
    if final_results.shape[0] == 0:
        print("WARNING! There were no predictions with score > 0.05. This indicates a bug in your code.")
        return dict()
    
    cocoDt = cocoGt.loadRes(final_results)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()

    f1 = 2 * (E.stats[0]*E.stats[8]) / (E.stats[0]+E.stats[8])
    
    stats = {
        "F1": f1,
        "mAP": E.stats[0], # same as mAP@
        "mAP@0.5": E.stats[1], # Same as PASCAL VOC mAP
        "mAP@0.75": E.stats[2],
        "mAP_small": E.stats[3],
        "mAP_medium": E.stats[4],
        "mAP_large": E.stats[5],
        "average_recall@1": E.stats[6],
        "average_recall@10": E.stats[7],
        "average_recall@100": E.stats[8],
        "average_recall@100_small": E.stats[9],
        "average_recall@100_medium": E.stats[10],
        "average_recall@100_large": E.stats[11],
    }
    return stats

