import os
from typing import Optional
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import json


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        sam_dataset: bool = True,
        target_size: tuple[int, int] = (256, 256),
    ):
        """
        Args:
            root (str): Path to the dataset root folder containing 'images/' and 'labels/' and the jason file of data split.
            transforms (albumentations.Compose, optional): Transformations to apply to images and labels.
        """
        self.root = root

        assert split in ["train", "test", "validation"]

        self.image_filenames: list = []
        self.label_filenames: list = []

        data_list = os.path.join(root, "dataset.json")
        with open(data_list, "r") as f:
            json_list = json.load(f)
        
        for item in json_list[split]:
            self.image_filenames.append(item["image"])
            self.label_filenames.append(item["label"]["labelIds"])

        assert len(self.image_filenames) == len(self.label_filenames), (
            "Mismatch between images and labels."
        )

        self.target_size = target_size
        self.transforms = transforms
        self.sam_dataset = sam_dataset

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.relpath(self.image_filenames[idx], "/")
        label_path = os.path.relpath(self.label_filenames[idx], "/")
        image_path = os.path.join(self.root, image_path)
        label_path = os.path.join(self.root, label_path)

        # Read image and label
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Resize image and label
        image = cv2.resize(
            image, self.target_size, interpolation=cv2.INTER_LINEAR
        )  # Use INTER_LINEAR for smooth resizing
        label = cv2.resize(
            label, self.target_size, interpolation=cv2.INTER_NEAREST
        )  # Use INTER_NEAREST for segmentation masks or labels

        # Convert grayscale labels to single-channel if necessary
        if len(label.shape) == 3:
            label = label[..., 0]

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        if self.sam_dataset:
            image = Image.fromarray(image)
        label = Image.fromarray((label).astype(np.uint8))

        return image, label


if __name__ == "__main__":
    # Example usage:
    root = "data/segmentation"

    # Define Albumentations transformations
    transforms = A.Compose(
        [
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    dataset = MedicalImageDataset(root=root, transforms=transforms)

    # Test dataset
    image, label = dataset[0]
    print(f"Image shape: {image.shape}, Label shape: {label.shape}")
