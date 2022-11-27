import numpy as np
from tqdm.auto import tqdm
from test_dataset import get_test_loader
import torch
import glob as glob
from config import create_model, DEVICE


# load the best model and trained weights
model = create_model()
checkpoint = torch.load('outputs/resnet/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
print('Checkpoint epoch:', checkpoint['epoch'])
# directory where all the images are present
DIR_TEST = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/test/images'

loader = get_test_loader(DIR_TEST)

print(f"Test instances: {len(loader)}")
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.05

pred_strings = []
for image, image_name in tqdm(loader, total=len(loader), mininterval=5, maxinterval=10):
    pred_string = image_name[0] + ','

    im_h, im_w = image.shape[:2]

    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    # load all detection to CPU for further operations
        
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        print(len(boxes))
        # filter out boxes according to `detection_threshold` and keep max 5
        if len(scores) > 5:
            ind = np.argpartition(scores, -5)[-5:]
            mask = scores[ind] >= detection_threshold
            ind = ind[mask]
        else:
            ind = scores >= detection_threshold
        
        boxes = boxes[ind].astype(np.int32)
        labels = labels[ind]
        scores = scores[ind]
            

        for label, box in zip(labels, boxes):
            l = box[0]
            t = box[1]
            r = box[2]
            b = box[3]
            pred_string += f'{label} {l} {t} {r} {b} '
        
    pred_string += '\n'

    pred_strings.append(pred_string)
        
print('TEST PREDICTIONS COMPLETE')
with open('pred_test.txt', 'w') as f:
    f.writelines(pred_strings)
