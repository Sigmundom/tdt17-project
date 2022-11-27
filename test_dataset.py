from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import glob

class TestDataset(Dataset):
    def __init__(self, folder_path):
        self.paths = glob.glob(f"{folder_path}/*.jpg")
        self.transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        image_name = image_path.split('/')[-1]
        image = default_loader(image_path)
        image = self.transform(image)
        return image, image_name
    
def get_test_loader(folder_path, batch_size=1, num_workers=2):
    images = TestDataset(folder_path)
    return DataLoader(images, batch_size=batch_size, num_workers=num_workers)
  