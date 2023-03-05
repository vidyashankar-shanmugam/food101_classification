from datetime import time

import torchvision
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from valloss_callback import EarlyStopping

def model_init():
    model = torchvision.models.efficientnet_b0(pretrained=True)
    print(model.classifier[1])
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=101, bias=True)
    print(model.classifier[1])
    return model

def train_model(model, dataloaders, device, dataset_sizes):

    # Storing the time during the start of training
    since = time.time()
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=7, verbose=False, delta=0, trace_func=print)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 1000


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) #Set predicted label value to 1
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() #Back propagate the loss
                        optimizer.step() #Update the weights

            if phase == 'train':
                scheduler.step(loss) #Update the learning rate

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # TODO: WANDB

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                # TODO: BEst to do early stopping based on loss and not accuracy I think
                early_stopping(val_loss=epoch_loss, model=model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    # Time elapsed
    time_elapsed = time.time() - since

    # load best model weights
    model.load_state_dict(early_stopping.best_model_wts)
    return model

