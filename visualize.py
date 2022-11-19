import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from config import CLASSES

from bbox_utils import bbox_ltrb_to_ltwh

DIR_TEST = 'data/test/validation/images'
# test_images = glob.glob(f"{DIR_TEST}/*.jpg")
with open('predictions.txt') as f:
    lines = f.readlines()

for line in lines:
    image_name, predictions = line.split(',')
    image = plt.imread(f'{DIR_TEST}/{image_name}')
    predictions = predictions.strip().split(' ')
    ax = plt.gca()
    ax.imshow(image)
    print(predictions)
    if len(predictions) == 1:
        predictions = []

    for i in range(0, len(predictions), 5):
        label = int(predictions[i])
        ltrb = np.array(predictions[i+1:i+5], dtype=np.int16)
        print(ltrb)
        ltwh = bbox_ltrb_to_ltwh(ltrb)
        print(ltwh)
        x,y,w,h = ltwh
        ax.add_patch(Rectangle((x,y), w, h, facecolor='none', linewidth=1, edgecolor='red'))
        plt.text(x, y-10, CLASSES[label])

    plt.savefig(f'inference_output/{image_name.split(".")[0]}.png',)
    plt.close()