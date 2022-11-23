

from config import IM_HEIGHT, NUM_WORKERS
from datasets import create_valid_dataset, create_valid_loader

def _explore_dims(loader):
    max_im_height = 0
    max_im_width = 0
    min_im_height = 9999
    min_im_width = 9999

    sum_im_height = 0
    sum_im_width = 0
    
    for _, targets in loader:
        target = targets[0]
        h, w = target['im_shape']

        if h > max_im_height:
            max_im_height = h
        if w > max_im_width:
            max_im_width = w
        if h < min_im_height:
            min_im_height = h
        if w < min_im_width:
            min_im_width = w
        sum_im_height += h
        sum_im_width += w

    
    print('Max height:', max_im_height)
    print('Max width:', max_im_width)
    print('Min height:', min_im_height)
    print('Min width:', min_im_width)
    print('Avg height:', sum_im_height/len(loader))
    print('Avg width:', sum_im_width/len(loader))

def _explore_placement(loader):
    min_y = 9999
    max_y = 0
    top_40 = 0
    top_50 = 0

    for _, targets in loader:
        target = targets[0]
        boxes = target['boxes']
        for box in boxes:
            l, t, r, b = box
            t = t/IM_HEIGHT
            b = b/IM_HEIGHT
            if t < min_y:
                min_y = t
            if b > max_y:
                max_y = b
            if t < 0.5:
                top_50 += 1
                if t < 0.4:
                    top_40 += 1

    print('Top:', min_y)
    print('Bottom:', max_y)
    print('Top 50%:', top_50)
    print('Top 40%', top_40)



def explore():
    val_dataset = create_valid_dataset()
    val_loader = create_valid_loader(val_dataset, NUM_WORKERS, batch_size=1)

    _explore_dims(val_loader)
    _explore_placement(val_loader)



if __name__ == '__main__':
    explore()