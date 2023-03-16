import os
import pandas as pd
from torchvision import transforms, io
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, phase, log, transform_compose):
        self.phase = phase
        df = pd.read_csv(os.path.join('meta', self.phase + '_df.csv'))
        self.image_paths = df['path'].values
        self.named_labels = df['label'].values
        self.labels = df['ordinal_label'].values
        self.log = log
        self.transform = transforms.Compose(transform_compose)

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
