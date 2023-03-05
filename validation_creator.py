import random
import os
import pandas as pd

def create_val_set():
    # Define the file path to your text file
    file_path = os.path.join(os.getcwd(), 'meta/train.txt')

    # Read in the file names and group them by category
    file_groups = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        file_name = line.strip()
        try:
            category, unique_id = file_name.split('/')
            if category not in file_groups:
                file_groups[category] = []
            file_groups[category].append(file_name)
        except:
            pass

    # Select 20% of the file names from each category
    selected_files = []
    for category in file_groups:
        num_files = len(file_groups[category])
        num_selected = int(num_files * 0.2)
        selected_files.extend(random.sample(file_groups[category], num_selected))

    with open('meta/val.txt', 'w') as f:
        f.write("path"+ "\n")
        for item in selected_files:
            f.write("%s\n" % item)

    return selected_files
