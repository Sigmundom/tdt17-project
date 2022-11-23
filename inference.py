import numpy as np
import cv2
import torch
import glob as glob
import os
from model import create_model
from config import IM_HEIGHT, IM_WIDTH, NUM_CLASSES, DEVICE
from utils import apply_nms



# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
# directory where all the images are present
# DIR_TEST = 'data/test/validation/images'
DIR_TEST = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/images'
# test_images = glob.glob(f"{DIR_TEST}/*.jpg")[:50]
test_images = [f'{DIR_TEST}/Norway_0000{str(i).zfill(2)}.jpg' for i in range(6,51)]
print(f"Test instances: {len(test_images)}")
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.05

pred_strings = []
for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1]
    pred_string = image_name + ','
    image = cv2.imread(test_images[i])
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image, (IM_WIDTH, IM_HEIGHT))
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    # load all detection to CPU for further operations
        
    outputs = [apply_nms(output, 0.3) for output in outputs]
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    print(outputs)
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
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
            pred_string += f'{label} {box[0]} {box[1]} {box[2]} {box[3]} '
        
        pred_string += '\n'

        pred_strings.append(pred_string)
        
    print(f"Image {i+1} done...")
print('TEST PREDICTIONS COMPLETE')
with open('predictions.txt', 'w') as f:
    f.writelines(pred_strings)
