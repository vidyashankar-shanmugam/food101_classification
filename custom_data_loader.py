import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
            # Load image from file
            image = torchvision.io.read_image(self.image_paths[index], mode=torchvision.io.ImageReadMode.RGB)
            # Apply image transformation (if provided)
            if self.transform is not None:
                image = self.transform(image)
            # Get label (if available)
            if self.labels is not None:
                label = self.labels[index]
                return image, label
            else:
                return image
    def __len__(self):
        return len(self.image_paths)
