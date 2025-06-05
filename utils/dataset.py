import os
from typing import Optional
import cv2
import numpy as np
import torch
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

        self.image_filenames = []
        self.label_filenames = []

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

        if image is None or label is None:
            raise FileNotFoundError(f"Failed to load {image_path} or {label_path}")

        if label.ndim == 3:
            label = label[..., 0]  # If accidentally 3D, take first channel

        original_size = image.shape[:2]                         # (H, W)

        # Resize image and label
        image = cv2.resize(
            image, self.target_size, interpolation=cv2.INTER_LINEAR
        )  # Use INTER_LINEAR for smooth resizing

        # Convert to float and normalize
        pixel_mean = np.array([123.675, 116.28, 103.53])
        pixel_std = np.array([58.395, 57.12, 57.375])
        image_norm = (image.astype(np.float32) - pixel_mean) / pixel_std

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Convert to torch tensor [3, H, W]
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()

        return {"image": image,
                "label": label,
                "original_size": original_size}, 


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
