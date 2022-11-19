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
        cocoGt: COCO,
        gpu_transform: torch.nn.Module):
    """
        Evaluates over dataloader and returns COCO stats
    """
    model.eval()
    ret = []
    for batch in tqdm.tqdm(dataloader, desc="Evaluating on dataset"):
        batch["image"] = to_cuda(batch["image"])
        batch = gpu_transform(batch)
        with torch.cuda.amp.autocast(enabled=True):
            predictions = model(batch["image"], nms_iou_threshold=0.50, max_output=200,
                score_threshold=0.05)

        for idx in range(len(predictions)):
            boxes_ltrb, categories, scores = predictions[idx]
            # ease-of-use for specific predictions
            H, W = batch["height"][idx], batch["width"][idx]
            box_ltwh = bbox_ltrb_to_ltwh(boxes_ltrb)
            box_ltwh[:, [0, 2]] *= W
            box_ltwh[:, [1, 3]] *= H
            box_ltwh, category, score = [x.cpu() for x in [box_ltwh, categories, scores]]
            img_id = batch["image_id"][idx].item()
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
    
    stats = {
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

