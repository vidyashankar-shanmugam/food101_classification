# FOOD 101

# Introduction
This is a machine learning script that trains and tests a model using the PyTorch library. It loads data using a custom data loader module, initializes a pre-trained EfficientNet model, trains the model, and then tests it using the F1 score and a confusion matrix. The script also uses the wandb library to log and visualize training and testing metrics.

# Installation
Before running the script, the following libraries must be installed:

        *   PyTorch
        *   scikit-learn
        *   WandB
        *   omegaconf
        *   hydra-core
These libraries can be installed using pip or conda.

# Usage
To use the script, run the myapp() function in the main.py file. The function loads configuration settings from the config.yaml file, which contains parameters such as batch size, number of workers, and learning rate. These parameters can be adjusted to fine-tune the model.

# Configuration
The configuration file config.yaml contains the following settings:

        - batch_size: the number of samples per batch
        - num_workers: the number of workers to use for data loading
        - pin_memory: whether to use pinned memory for faster data transfer
        - learning_rate: the learning rate for the optimizer
        - patience: the number of epochs to wait before early stopping
        - patience_lr: the number of epochs to wait before reducing the learning rate
        - loss_function: the loss function to use for training
        - optimizer: the optimizer to use for training
        - num_epochs: the number of epochs to train the model for
        - project_name: the name of the wandb project to log results to
# Validation Creation
The validation set is created by splitting the training set into 80% training and 20% validation. The validation set is used to monitor over-fitting of the model. Here the train.txt contains partial paths of
training data. A 20% of random image from each class is selected to form validation dataset. Next separate csv files are created for train, test and validation containing complete paths of images, thier labels and their corresponding class names. These csv files are saved in 'meta' folder to be used by data loaders.
This code is to be run only once such that ordinal encoding and split stays the same for all epochs and hyper-parameter tuning.

# Myapp()
The myapp() function in the main.py file contains the following steps:

        1.  Load configuration settings from the config.yaml file
        2.  Initialize the wandb project
        3.  Load the data using the custom data loader module
        4.  Initialize the EfficientNet model
        5.  Train the model
        6.  Test the model
        7.  Log the results to wandb
# Data Loader
The data loader module loads the data from the food-101 dataset as batches. 
The module also creates a custom dataset class that inherits from the PyTorch Dataset class. A transform is defined to resize, crop, augment the images and convert them to tensors.
The custom dataset class contains the following methods:

        - __init__(): initializes the dataset
        - __len__(): returns the length of the dataset
        - __getitem__(): returns a sample from the dataset
A sample from the data loader object is visualized to check if the image matches the label using display_image() function.

# EfficientNet
The EfficientNet model is a convolutional neural network that uses a compound scaling method to scale the width, depth, and resolution of the model. The model is pre-trained on the ImageNet dataset and is available in the PyTorch library. The model is initialized using the PyTorch hub module. The model is then trained using the Adam optimizer and the cross-entropy loss function. The model is trained with hyper-parameter tuning. The model is then tested using the F1 score and a confusion matrix. The classifier layer is modified to output 101 classes, one for each food category and unfrozen while all other layer weights are frozen to be used as feature-extractor.

# Training
The model is trained with hyper-parameter tuning. The hyper-parameters are as follows:

        - Learning rate
        - Patience for early stopping
        - Patience for learning rate scheduler
        - Loss function
        - Optimizer

# Testing
The model is tested using the F1 score and a confusion matrix for multiclass classification. The F1 score is calculated using the sklearn library. The confusion matrix is calculated using the wandb library.

