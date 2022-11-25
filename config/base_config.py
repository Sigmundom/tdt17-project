import torch

BATCH_SIZE = 8 # increase / decrease according to GPU memeory
NUM_WORKERS = 4
NUM_EPOCHS = 20 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train' if torch.cuda.is_available() else 'data/test/train'
# validation images and XML files directory
VALID_DIR = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train' if torch.cuda.is_available() else 'data/test/validation' 

CLASSES = [
    '__background__', 'D00', 'D10', 'D20', 'D40'
]

DATA_BLACKLIST = ['Norway_004504.jpg']
# classes: 0 index is reserved for background
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = 'outputs'
