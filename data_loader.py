import os
import torch
from custom_data_loader import CustomDataset
from torchvision import utils
from display_image import imshow

def data_loader(batch_size, num_workers, pin_memory, log, data_dir):
        with open(os.path.join(data_dir,'meta/classes.txt'), 'r') as f:
            classes = f.readlines()
        # Create CustomDataset object for train and test splits
        train_dataset = CustomDataset('train', log, data_dir)
        validation_dataset = CustomDataset('val', log, data_dir)
        test_dataset = CustomDataset('test', log, data_dir)

        # Create DataLoader objects for train and test splits
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=5)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        print("Number of training samples: ", len(train_dataset))
        print("Number of validation samples: ", len(validation_dataset))
        print("Number of test samples: ", len(test_dataset))
        print("Total number of samples: ", len(train_dataset) + len(validation_dataset) + len(test_dataset))
        print("Number of classes: ", len(classes))
        dataset_sizes = {'train': len(train_dataset), 'val': len(validation_dataset), 'test': len(test_dataset)}

        #display all images in single plot
        inp, class_idx = next(iter(train_loader))
        out = utils.make_grid(inp,nrow= 8, padding=2)
        imshow(out, title=[classes[int(x)] for x in class_idx])

        dataloaders = {'train': train_loader, 'val': validation_loader, 'test': test_loader}

        return dataloaders, dataset_sizes