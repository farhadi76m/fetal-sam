import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image
class MedicalImageDataset(Dataset):
    def __init__(self, root, transforms=None,sam_dataset = True):
        """
        Args:
            root (str): Path to the dataset root folder containing 'images/' and 'labels/'.
            transforms (albumentations.Compose, optional): Transformations to apply to images and labels.
        """
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_filenames = [lbl for lbl in sorted(os.listdir(self.label_dir)) if lbl.endswith('_gtFine_labelIds.png')]
        
        assert len(self.image_filenames) == len(self.label_filenames), "Mismatch between images and labels."
        self.transforms = transforms
        
        self.sam_dataset = sam_dataset

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        # Read image and label
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Convert grayscale labels to single-channel if necessary
        if len(label.shape) == 3:
            label = label[..., 0]

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        if self.sam_dataset : 
            image = Image.fromarray(image)
        label = Image.fromarray((label).astype(np.uint8))

        return image, label

if __name__=="__main__" :
    # Example usage:
    root = 'data/segmentation'

    # Define Albumentations transformations
    transforms = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    dataset = MedicalImageDataset(root=root, transforms=transforms)

    # Test dataset
    image, label = dataset[0]
    print(f"Image shape: {image.shape}, Label shape: {label.shape}")
