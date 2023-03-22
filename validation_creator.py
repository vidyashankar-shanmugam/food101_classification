import random
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

""" create_trainvaltest_df() creates a dataframe for each of the train, validation and test sets. 
The dataframe contains the path to the image, classes and the ordinal label.  
This code is run only once to create the dataframes such that the same train, test and validation set along with ordinal encding is used"""

def create_trainvaltest_df():
    # Create the validation set
    val_files = create_val_set()
    # Create ordinal encoder object
    ord_enc = OrdinalEncoder()
    # Define the directory path to images
    data_dir = os.path.join(os.getcwd(), 'images')
    # Use same code to create dataframes for train, validation and test sets
    for phase in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join('meta', phase + '.txt'))
        # Remove the validation set from the train set when creating train set
        if phase == 'train':
            df = df[~df.isin(val_files).any(axis=1)]
        # Extract the class name from the path
        df['label'] = df['path'].apply(lambda x: x.split('/')[0])
        # Create complete path to the image
        df['path'] = df['path'].apply(lambda x: os.path.join(data_dir, (x + '.jpg')))
        # Encode the classes
        df['ordinal_label'] = ord_enc.fit_transform(df['label'].values.reshape(-1, 1))
        # Save the dataframe as a csv file
        df.to_csv(os.path.join('meta', phase + '_df.csv'), index=False)

""" create_val_set() creates a validation set from the train set. 
It takes 20% of the train images from each class and puts them in a validation set. 
Created text file containing validation set is saved in the meta folder."""

def create_val_set():
    # Define the file path to your train text file
    file_path = os.path.join(os.getcwd(), 'meta/train.txt')
    # Create empty dictionary to group images by class_names
    file_groups = {}
    # Read the train text file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Group the images by class_names
    for line in lines:
        file_name = line.strip()
        try:
            category, unique_id = file_name.split('/')
            # Create a new key with class_name in the dictionary if it doesn't exist
            if category not in file_groups:
                file_groups[category] = []
            file_groups[category].append(file_name)
        except:
            pass

    # Select random 20% of the file names from each class
    selected_files = []
    for category in file_groups:
        num_files = len(file_groups[category])
        num_selected = int(num_files * 0.2)
        selected_files.extend(random.sample(file_groups[category], num_selected))
    # Save the validation set in a text file
    with open('meta/val.txt', 'w') as f:
        f.write("path"+ "\n")
        for item in selected_files:
            f.write("%s\n" % item)
    # Return the validation set
    return selected_files
