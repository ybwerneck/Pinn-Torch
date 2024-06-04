# dependencies.py

import torch
import numpy as np
import torch.nn.functional as F
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import h5py
import chaospy as cp
from torch.optim.lr_scheduler import ReduceLROnPlateau

import shutil

def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file and remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it's a directory and remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist.')