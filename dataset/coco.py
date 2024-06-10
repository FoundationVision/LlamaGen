import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class SingleFolderDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, file_name) for file_name in os.listdir(directory)
                            if os.path.isfile(os.path.join(directory, file_name))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_coco(args, transform):
    return SingleFolderDataset(args.data_path, transform=transform)