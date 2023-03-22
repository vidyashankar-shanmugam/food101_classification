#from datetime import time
import torchvision
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from valloss_callback import EarlyStopping
import numpy as np
import math

def model_init():
    """Initialize the Efficient net model pre-trained on image net for this run.
    I have kept the model frozen for weights in all the layers except the last layer."""
    model = torchvision.models.efficientnet_b0(pretrained=True)
    print(model.classifier[1])
    # Modify the last layer to output 101 classes
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=101, bias=True)
    print(model.classifier[1])
    # Freeze all the layers except the last layer
    for name, child in model.named_children():
        if name in ['classifier']:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False
    return model

def train_model(model, dataloaders, device, dataset_sizes, log, cfg):

    """Train the model and return the best model with the lowest validation loss."""
    best_acc = 0.0
    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience= cfg.patience, verbose=False, trace_func=print)
    # Initialize the optimizer, scheduler and loss function
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= cfg.lr, weight_decay= cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode ='min', factor = cfg.factor ,patience= cfg.patience_lr, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = cfg.num_epochs

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # Initialize running_loss and running_corrects to calculate the loss and accuracy for each epoch
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # To track history if only in train as we won't be updating the weights in val phase
                with torch.set_grad_enabled(phase == 'train'):
                    log.debug('training')
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) #Set predicted label value to 1
                    loss = criterion(outputs, labels.long())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() #Back propagate the loss
                        optimizer.step() #Update the weights
                    # Calculate the loss and accuracy for each batch and add it to the running_loss and running_corrects
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels).item()
            # Update the learning rate after each epoch only in the training phase
            if phase == 'train':
                scheduler.step(loss)
            # Calculate the loss and accuracy for each epoch
            print("calculating epoch loss")
            print(running_loss)
            print(running_corrects)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # If the loss is nan or inf, stop training
            if np.isfinite(running_loss) is False or math.isfinite(running_loss) is False \
                    or math.isnan(running_loss) is True:
                log.info("Loss is nan or inf, stopping training")
                break
            # deep copy the model if the current epoch has the best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            # Save the model if the current epoch has the  lowest validation loss
            if phase == 'val':
                early_stopping(val_loss=epoch_loss, model=model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    # load best model weights
    model.load_state_dict(early_stopping.best_model_wts)
    return model

