import pandas as pd
from custom_data_loader import CustomDataset
import os
from torchvision import transforms, utils
import torch
from display_image import imshow

def data_loader():
        # Set batch size and number of workers for loading data
        batch_size = 64
        num_workers = 4
        data_dir = os.path.join(os.getcwd(), 'images')
        with open('meta/val.txt', 'r') as f:
            selected_files = f.readlines()
        val_path = pd.read_csv('meta/val.txt')
        val_path['label'] = val_path['path'].apply(lambda x: x.split('/')[0])
        val_path['path'] = val_path['path'].apply(lambda x:  os.path.join(data_dir, (x+'.jpg')))
        train_path = pd.read_csv('meta/train.txt')
        train_path = train_path[~train_path.isin(selected_files).any(axis=1)]
        train_path['label'] = train_path['path'].apply(lambda x: x.split('/')[0])
        train_path['path'] = train_path['path'].apply(lambda x:  os.path.join(data_dir, (x+'.jpg')))
        test_path = pd.read_csv('meta/test.txt')
        test_path['label'] = test_path['path'].apply(lambda x: x.split('/')[0])
        test_path['path'] = test_path['path'].apply(lambda x:  os.path.join(data_dir, (x+'.jpg')))

        data_transforms = transforms.Compose([
                lambda x: x.float() / 255,
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Create CustomDataset object for train and test splits
        train_dataset = CustomDataset(train_path['path'].values, train_path['label'].values, transform=data_transforms)
        validation_dataset = CustomDataset(val_path['path'].values, val_path['label'].values, transform=data_transforms)
        test_dataset = CustomDataset(test_path['path'].values, test_path['label'].values, transform=data_transforms)

        # Create DataLoader objects for train and test splits
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print("Number of training samples: ", len(train_dataset))
        print("Number of validation samples: ", len(validation_dataset))
        print("Number of test samples: ", len(test_dataset))
        print("Total number of samples: ", len(train_dataset) + len(validation_dataset) + len(test_dataset))
        print("Number of classes: ", len(train_path.label.unique()))
        dataset_sizes = {'train': len(train_dataset), 'val': len(validation_dataset), 'test': len(test_dataset)}

        #display all images in single plot
        input, classes = next(iter(train_loader))
        out = utils.make_grid(input,nrow=2)
        imshow(out, title=classes)

        dataloaders = {'train': train_loader, 'val': validation_loader, 'test': test_loader}

        return dataloaders, dataset_sizes