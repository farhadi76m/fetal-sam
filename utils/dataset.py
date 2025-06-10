import os
from typing import Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import json

class MedicalImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms = None, # : Optional[A.Compose]
        target_size = (512, 512), # : tuple[int, int]
        num_classes: int = 4,  # Number of classes INCLUDING background
    ):
        """
        Args:
            root (str): Root folder with 'images/' and 'labels/', and a dataset.json with split info.
            split (str): 'train', 'test', or 'validation'
        """
        self.root = root
        self.target_size = target_size
        self.transforms = transforms
        self.num_classes = num_classes
        assert split in ["train", "test", "validation"]

        with open(os.path.join(root, "dataset.json"), "r") as f:
            json_list = json.load(f)

        self.image_filenames = [os.path.join(root, item["image"]) for item in json_list[split]]
        self.label_filenames = [os.path.join(root, item["label"]["labelIds"]) for item in json_list[split]]
        assert len(self.image_filenames) == len(self.label_filenames), "Mismatch between images and labels."

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
            label = label[..., 0]  # If 3D, take first channel
        original_size = image.shape[:2]  # (H, W)

        # Resize image and label
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)

        # One-hot encode multi-label mask [num_classes, H, W]
        multi_label = np.zeros((self.num_classes, *self.target_size), dtype=np.float32)
        for c in range(self.num_classes):
            multi_label[c] = (label == c).astype(np.float32)

        # Albumentations expects [H,W,C]
        if self.transforms:
            augmented = self.transforms(image=image, mask=multi_label.transpose(1,2,0))
            image = augmented["image"]
            multi_label = augmented["mask"].permute(2,0,1)  # [C,H,W]
        else:
            # Manual normalization (ImageNet-style)
            image = (image.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            multi_label = torch.from_numpy(multi_label)

        return {
            "image": image,                  # torch.FloatTensor [3, H, W]
            "masks": multi_label,            # torch.FloatTensor [num_classes, H, W]
            "original_size": original_size,  # tuple (H, W)
        }

# if __name__ == "__main__":
#     root = "data/segmentation"
#     transforms = A.Compose([
#         A.Resize(512, 512),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#         ToTensorV2(),
#     ])
#     dataset = MedicalImageDataset(root=root, split="train", transforms=transforms)
#     sample = dataset[0]
#     print(f"Image shape: {sample['image'].shape}, Mask shape: {sample['masks'].shape}, Orig size: {sample['original_size']}")
