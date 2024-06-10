import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DatasetJson(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        json_path = os.path.join(data_path, 'image_paths.json')
        assert os.path.exists(json_path), f"please first run: python3 tools/openimage_json.py"
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image_path_full = os.path.join(self.data_path, image_path)
        image = Image.open(image_path_full).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_openimage(args, transform):
    return DatasetJson(args.data_path, transform=transform)
