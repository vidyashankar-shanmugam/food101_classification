import os
import pandas as pd
from torchvision import transforms, io
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """ Custom dataset for loading images one by one from a directory
    This class is designed to load images for train, test and validation sets """
    def __init__(self, phase, log, data_dir):
        """ Args:
            phase (str): 'train' or 'val' or 'test'
            log (logger): logger
            data_dir (str): path to data directory
            image_paths (df column): list of paths to images
            labels (df column): list of labels
            classes (df column): list of class names
            transform (callable, optional): optional transform to be applied on a sample """
        self.phase = phase
        df = pd.read_csv(os.path.join(data_dir, 'meta', self.phase + '_df.csv'))
        self.image_paths = df['path'].values
        self.classes = df['label'].values
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
        # Apply image transformation
        image = self.transform(image)
        # Get label and class name
        label = self.labels[index]
        # Log image path and label to check if image and class names match
        self.log.debug("path",self.image_paths[index],"\t","label",self.classes[index],"\n")
        return image, label

    def __len__(self):
        return len(self.image_paths)
