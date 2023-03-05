import os
import torch


data_dir = os.path.join(os.getcwd(), 'images')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")







