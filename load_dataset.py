# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import os
import random
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.classes = os.listdir(os.path.join(root_dir, mode))
        self.image_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, mode, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Load the image
        img = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_paths)
    
    
    def display_image(self, index, original=True):
        img_path = self.image_paths[index]
        label = self.labels[index]
    
        img = Image.open(img_path).convert("RGB")
    
        if not original and self.transform:
            img = self.transform(img)
    
        img_array = np.array(img)
        if img_array.shape[2] == 3:
            img_array = img_array.transpose(2, 0, 1)  # If in (H, W, C) format, convert to (C, H, W)
    
        plt.imshow(img_array.transpose(1, 2, 0))  # Display in (H, W, C) format
        plt.title(f"Label: {label}")
        plt.show()
        


    def display_grid(self, num_rows=4, num_cols=4, original=True):
        random_indices = random.sample(range(len(self)), num_rows * num_cols)
    
        plt.figure(figsize=(10, 10))
    
        for i, idx in enumerate(random_indices):
            plt.subplot(num_rows, num_cols, i + 1)
            img, label = self.__getitem__(idx)
    
    
            if isinstance(img, torch.Tensor):
                img = img.numpy()  # Convert torch.Tensor to numpy array
    
            img_array = np.array(img)
            if img_array.shape[2] == 3:
                img_array = img_array.transpose(2, 0, 1)
    
            plt.imshow(img_array.transpose(1, 2, 0))
            plt.title(f"Label: {label}")
            plt.axis('off')
    
        plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------