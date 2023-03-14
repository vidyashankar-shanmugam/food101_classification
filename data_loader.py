import os
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from custom_data_loader import CustomDataset
from torchvision import transforms, utils
from display_image import imshow

def df_creator(df, data_dir):
    # Create a dataframe with the image path and the label
    ord_enc = OrdinalEncoder()
    df['label'] = df['path'].apply(lambda x: x.split('/')[0])
    df['path'] = df['path'].apply(lambda x: os.path.join(data_dir, (x + '.jpg')))
    df['ordinal_label'] = ord_enc.fit_transform(df['label'].values.reshape(-1, 1))
    return df

def data_loader(batch_size, num_workers, data_dir):
        # Set batch size and number of workers for loading data
        batch_size = batch_size
        num_workers = num_workers
        with open('meta/val.txt', 'r') as f:
            selected_files = f.readlines()
        val_path = pd.read_csv('meta/val.txt')
        val_path = df_creator(val_path, data_dir)
        train_path = pd.read_csv('meta/train.txt')
        train_path = train_path[~train_path.isin(selected_files).any(axis=1)]
        train_path = df_creator(train_path, data_dir)
        test_path = pd.read_csv('meta/test.txt')
        test_path = df_creator(test_path, data_dir)
        classes = (train_path['label'].unique()).tolist()
        data_transforms = transforms.Compose([
                lambda x: x.float() / 255,
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Create CustomDataset object for train and test splits
        train_dataset = CustomDataset(train_path['path'].values, train_path['ordinal_label'].values, transform=data_transforms)
        validation_dataset = CustomDataset(val_path['path'].values, val_path['ordinal_label'].values, transform=data_transforms)
        test_dataset = CustomDataset(test_path['path'].values, test_path['ordinal_label'].values, transform=data_transforms)

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
        inp, class_idx = next(iter(train_loader))
        out = utils.make_grid(inp,nrow= 8, padding=2)
        imshow(out, title=[classes[int(x)] for x in class_idx])

        dataloaders = {'train': train_loader, 'val': validation_loader, 'test': test_loader}

        return dataloaders, dataset_sizes