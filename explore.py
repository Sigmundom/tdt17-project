

import glob
from config import CLASSES, IM_HEIGHT, VALID_DIR
from xml.etree import ElementTree as et

def _explore_dims(loader):
    max_im_height = 0
    max_im_width = 0
    min_im_height = 9999
    min_im_width = 9999

    sum_im_height = 0
    sum_im_width = 0
    
    for target in loader:
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
    min_y = 2
    max_y = 0
    top_40 = 0
    top_50 = 0

    for target in loader:
        boxes = target['boxes']
        for box in boxes:
            l, t, r, b = box
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

def _explore_classes(loader):
    occurances = [0]*len(CLASSES)
    for target in loader:
        labels = target['labels']
        for label in labels:
            occurances[label] += 1

    print('Number of occurances per class:')
    for c, n in zip(CLASSES, occurances):
        print(f'{c}: {n}')

def explore():
    paths = glob.glob(f"{VALID_DIR}/annotations/xmls/*.xml")

    annotations = []
    for annot_file_path in paths:
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # get the height and width of the image
        size = root.find('size')
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            
            labels.append(CLASSES.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = float(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = float(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = float(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = float(member.find('bndbox').find('ymax').text)
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)
            xmax_final = (xmax/image_width)
            ymin_final = (ymin/image_height)
            yamx_final = (ymax/image_height)
            
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
    
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        target["im_shape"] = (image_height, image_width)

            
        annotations.append(target)

    print('Number of samples:', len(annotations))
    _explore_dims(annotations)
    _explore_placement(annotations)
    _explore_classes(annotations)



if __name__ == '__main__':
    explore()