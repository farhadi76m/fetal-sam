import os
from typing import Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        target_size: tuple[int, int] = (512, 512),
        num_classes:int = 4, # Number of classes INCLUDING background (if needed)
    ):
        """
        Args:
            root (str): Path to the dataset root folder containing 'images/' and 'labels/' and the jason file of data split.
            transforms (albumentations.Compose, optional): Transformations to apply to images and labels.
        """
        self.root = root
        self.target_size = target_size
        self.transforms = transforms
        self.num_classes = num_classes
        assert split in ["train", "test", "validation"]

        data_list = os.path.join(root, "dataset.json")
        with open(data_list, "r") as f:
            json_list = json.load(f)

        self.image_filenames = [os.path.join(self.root, item["image"]) for item in json_list[split]]
        self.label_filenames = [os.path.join(self.root, item["label"]["labelIds"]) for item in json_list[split]]

        assert len(self.image_filenames) == len(self.label_filenames), (
            "Mismatch between images and labels."
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        # Read image and label
        image = cv2.imread(self.image_filenames[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_filenames[idx], cv2.IMREAD_UNCHANGED)

        if image is None or label is None:
            raise FileNotFoundError(f"Failed to load {self.image_filenames[idx]} or {self.label_filenames[idx]}")

        if label.ndim == 3:
            label = label[..., 0]  # If accidentally 3D, take first channel
        original_size = image.shape[:2]                         # (H, W)

        # Resize image and label
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Convert label to multi-label (one-hot binary mask per class)
        multi_label = np.zeros((self.num_classes, *self.target_size), dtype=np.uint8)
        for c in range(self.num_classes):
            multi_label[c] = (label == c).astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(image=image, mask=multi_label.transpose(1,2,0))  # [H,W,C]
            image = augmented["image"]
            multi_label = augmented["mask"].permute(2,0,1)  # [C,H,W]
        else:
            # Normalize (if not using Albumentations Normalize)
            image = (image.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            multi_label = torch.from_numpy(multi_label)

        return {
            "image": image,              # torch.Tensor [3, H, W]
            "masks": multi_label,        # torch.Tensor [num_classes, H, W], binary
            "original_size": original_size,
        }


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
