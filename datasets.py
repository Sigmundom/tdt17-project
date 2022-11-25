from bbox_utils import bbox_ltrb_to_ltwh
import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from pycocotools.coco import COCO
from config import (
    CLASSES, DATA_BLACKLIST, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform
# the dataset class
class CustomDataset(Dataset):
    def __init__(self, dir_path, classes, split=slice(None, None), transforms=None):
        self.transforms = transforms
        # self.dir_path = dir_path
        self.img_path = os.path.join(dir_path, 'images')
        self.annot_path = os.path.join(dir_path, 'annotations/xmls')
        # self.height = height
        # self.width = width
        self.classes = classes
        
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.img_path}/*.jpg")[split]
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        
        for item in DATA_BLACKLIST:
            try:
                self.all_images.remove(item)
                print('Removed blacklisted item:', item)
            except ValueError:
                print(f"Tried to remove {item}, but couldn't find it")

        self.all_images = sorted(self.all_images)
    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.img_path, image_name)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # y_crop = int(image.shape[0]*0.4)
        y_crop = 0
        # image_cropped = image[y_crop:, :]
        # image_resized = cv2.resize(image_cropped, (self.width, self.height))
        image_resized = image
        image_resized /= 255.0
        
        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.annot_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0] - y_crop
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = float(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = float(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = float(member.find('bndbox').find('ymin').text) - y_crop
            # ymax = right corner y-coordinates
            ymax = float(member.find('bndbox').find('ymax').text) - y_crop
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            # xmin_final = (xmin/image_width)*self.width
            # xmax_final = (xmax/image_width)*self.width
            # ymin_final = (ymin/image_height)*self.height
            # ymax_final = (ymax/image_height)*self.height
            
            boxes.append([xmin, ymin, xmax, ymax])
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1,4)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # print(labels)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor(idx)

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes']).reshape(-1,4)
            
        return image_resized, target

    def get_annotations_as_coco(self) -> COCO:
        """
            Returns bounding box annotations in COCO dataset format
        """
        coco_anns = {"annotations" : [], "images" : [], "licences" : [{"name": "", "id": 0, "url": ""}], "categories" : []}
        coco_anns["categories"] = [
            {"name": cat, "id": i+1, "supercategory": ""}
            for i, cat in enumerate(CLASSES) 
        ]
        ann_id = 1
        for idx in range(len(self)):
            # image_id = idx
            _, target = self.__getitem__(idx)
            boxes_ltrb  = target['boxes']
            boxes_ltwh = bbox_ltrb_to_ltwh(boxes_ltrb)
            coco_anns["images"].append({"id": int(target['image_id']), "height": self.height, "width": self.width })
            for box, label in zip(boxes_ltwh, target['labels']):
                box = box.tolist()
                area = box[-1] * box[-2]
                coco_anns["annotations"].append({
                    "bbox": box, "area": area, "category_id": int(label),
                    "image_id": int(target['image_id']), "id": ann_id, "iscrowd": 0, "segmentation": []}
                )
                ann_id += 1
        coco_anns["annotations"].sort(key=lambda x: x["image_id"])
        coco_anns["images"].sort(key=lambda x: x["id"])
        coco = COCO()
        coco.dataset = coco_anns
        coco.createIndex()
        coco.getAnnIds()
        return coco

    def __len__(self):
        return len(self.all_images)
# prepare the final datasets and data loaders
def create_train_dataset():
    # train_dataset = CustomDataset(TRAIN_DIR, CLASSES,  slice(50, 250),get_train_transform())
    # train_dataset = CustomDataset(TRAIN_DIR, CLASSES,  slice(816, None),get_train_transform())
    train_dataset = CustomDataset(TRAIN_DIR, CLASSES,  slice(0, None),get_train_transform())
    return train_dataset

def create_valid_dataset():
    # valid_dataset = CustomDataset(VALID_DIR, CLASSES, slice(0, 50), get_valid_transform())
    valid_dataset = CustomDataset(VALID_DIR, CLASSES, slice(0, 816), get_valid_transform())
    return valid_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers=0, batch_size=None):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE if batch_size is None else batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader

# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)