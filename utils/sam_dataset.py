import numpy as np
import torch

class SAMSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        prompt_type: str = "box",
        model_input_size=(512, 512),
        transform=None,
    ):
        """
        Args:
            base_dataset: An instance of MedicalImageDataset
            prompt_type: "point" or "box"
            model_input_size: Size the model expects (H, W)
            transform: Optional transform on output
        """
        self.dataset = base_dataset
        self.prompt_type = prompt_type
        self.model_input_size = model_input_size
        self.transform = transform
        self.num_classes = self.dataset.num_classes

    @staticmethod
    def generate_point_prompt(mask, model_input_size, original_size):
        """
        One random point per class (excluding background).
        Returns: points [N,2] and labels [N]
        """
        points = []
        point_labels = []

        h_scale = model_input_size[0] / original_size[0]
        w_scale = model_input_size[1] / original_size[1]

        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            if class_id == 0:
                continue  # skip background
            class_mask = (mask == class_id)
            if class_mask.any():
                y_indices, x_indices = np.where(class_mask)
                rand_idx = np.random.randint(0, len(y_indices))
                x, y = x_indices[rand_idx], y_indices[rand_idx]
                # Scale to model input
                x = x * w_scale
                y = y * h_scale
                points.append([x, y])
                point_labels.append(class_id)
            else:
                points.append([-1, -1])
                point_labels.append(-1)
        if len(points) == 0:
            points = [[-1, -1]]
            point_labels = [-1]
        return np.array(points, dtype=np.float32), np.array(point_labels, dtype=np.int64)

    @staticmethod
    def get_bbox_from_mask(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return np.array([-1, -1, -1, -1], dtype=np.float32)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    def __len__(self):
        # Number of samples: one per image per class (excluding background)
        return len(self.dataset) * (self.num_classes - 1)

    def __getitem__(self, idx):
        # Which image and class
        img_idx = idx // (self.num_classes - 1)
        class_idx = idx % (self.num_classes - 1) + 1  # skip background (class 0)
        data = self.dataset[img_idx]
        image = data["image"]
        masks = data["masks"]            # [num_classes, H, W]
        orig_size = data["original_size"]

        if self.prompt_type == "point":
            # Generate point for this class
            mask = masks[class_idx].cpu().numpy()
            points, point_labels = self.generate_point_prompt(mask, self.model_input_size, orig_size)
            # [B, N, 2] for points, [B, N] for labels (B=1 here)
            sample = {
                "image": image,  # [3, H, W]
                "original_size": orig_size,  # (H, W)
                "point_coords": torch.from_numpy(points).unsqueeze(0),  # [1, N, 2]
                "point_labels": torch.from_numpy(point_labels).unsqueeze(0),  # [1, N]
                "mask": torch.from_numpy(mask).unsqueeze(0),  # [1, H, W]
            }
        else:
            # Box for this class
            mask = masks[class_idx].cpu().numpy()
            box = self.get_bbox_from_mask(mask)
            box_tensor = torch.from_numpy(box).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 4]
            sample = {
                "image": image,
                "original_size": orig_size,
                "boxes": box_tensor,
                "mask": torch.from_numpy(mask).unsqueeze(0),  # [1, H, W]
            }
        if self.transform:
            sample = self.transform(sample)
        return sample

# Example usage:

from dataset import MedicalImageDataset
base_dataset = MedicalImageDataset(root="E:/git_projects/fetal-sam/segmentation_data", )
sam_dataset = SAMSegmentationDataset(base_dataset, prompt_type="point")
s = sam_dataset[0]
print({k: (v.shape if torch.is_tensor(v) else v) for k, v in s.items()})
