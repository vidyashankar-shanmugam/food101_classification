import os
from data_loader import data_loader
from device import dev
from model import model_init, train_model

data_dir = os.path.join(os.getcwd(), 'images')
batch_size = 64
num_workers = 3
dataloaders, dataset_sizes = data_loader(batch_size, num_workers, data_dir)
dev = dev()
model = model_init()
model = train_model(model, dataloaders, dev, dataset_sizes)