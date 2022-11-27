from os import mkdir, path
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES, OUT_DIR

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0



class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    F1 score is more than the previous max score, then save the
    model state.
    """
    def __init__(
        self, best_f1_score=float(0)
    ):
        self.best_f1_score = best_f1_score
        
    def __call__(
        self, current_f1_score, 
        epoch, model, optimizer
    ):
        if current_f1_score > self.best_f1_score:
            self.best_f1_score = current_f1_score
            if not path.isdir(OUT_DIR):
                mkdir(OUT_DIR)
            print(f"\nBest F1 score: {self.best_f1_score}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{OUT_DIR}/best_model.pth')




def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# define the training tranforms
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(0.5),
        # A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.2),
        # A.RandomBrightnessContrast(p=0.2),
        # A.Blur(blur_limit=1, p=0.2),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]], 
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{OUT_DIR}/last_model.pth')
                
# def save_loss_plot(OUT_DIR, train_loss, val_loss):
#     figure_1, train_ax = plt.subplots()
#     figure_2, valid_ax = plt.subplots()
#     train_ax.plot(train_loss, color='tab:blue')
#     train_ax.set_xlabel('iterations')
#     train_ax.set_ylabel('train loss')
#     valid_ax.plot(val_loss, color='tab:red')
#     valid_ax.set_xlabel('iterations')
#     valid_ax.set_ylabel('validation loss')
#     figure_1.savefig(f"{OUT_DIR}/train_loss.png")
#     figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
#     print('SAVING PLOTS COMPLETE...')
#     plt.close('all')