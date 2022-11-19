import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
# RESIZE_TO = 416 # resize the image for training and transforms
# IM_WIDTH = 1920
# IM_HEIGHT = 1080
# IM_WIDTH = 640
# IM_HEIGHT = 360
IM_WIDTH = 320
IM_HEIGHT = 180

NUM_EPOCHS = 1 # number of epochs to train for
NUM_WORKERS = 1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train' if torch.cuda.is_available() else 'data/test/train'
# validation images and XML files directory
VALID_DIR = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train' if torch.cuda.is_available() else 'data/test/validation' 
# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'D00', 'D10', 'D20', 'D40'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = 'outputs'