import os
import pandas as pd
from torchvision import transforms, io
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, phase, log, data_dir):
        self.phase = phase
        df = pd.read_csv(os.path.join(data_dir, 'meta', self.phase + '_df.csv'))
        self.image_paths = df['path'].values
        self.named_labels = df['label'].values
        self.labels = df['ordinal_label'].values
        self.log = log
        self.transform = transforms.Compose([
            transforms.RandAugment(2, 5),
            lambda x: x.float() / 255,
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # Load image from file
        image = io.read_image(self.image_paths[index], mode=io.ImageReadMode.RGB)
        # Apply image transformation (if provided)
        if self.transform is not None:
            image = self.transform(image)
        # Get label (if available)
        if self.labels is not None:
            label = self.labels[index]
            self.log.debug("path",self.image_paths[index],"\t","label",self.named_labels[index],"\n")
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_paths)
