import numpy as np
import torch

class SAMSegmentationDataset:
    def __init__(self,
                 base_dataset,
                 prompt_type:str = "box",
                 num_samples_per_image: int = 1):

        self.dataset = base_dataset
        assert prompt_type in {"point", "box"}, "Input must be 'point' or 'box'"
        self.prompt_type = prompt_type
        self.num_samples_per_image = num_samples_per_image
        self.num_classes = self.dataset.num_classes
        
    @staticmethod
    def generate_point_prompt(mask, model_input_size, original_size):
        """Generate point prompts from segmentation mask with fixed number of points"""
        # num_points = self.num_classes  # One point per class (including background)
        num_points = np.unique(mask)  # One point per class (including background)
        points = []
        point_labels = []

        # Adjust for resized image:
        h_scale = model_input_size[0] / original_size[0]
        w_scale = model_input_size[1] / original_size[1]
        
        # Add one point for each class (1,2,3)
        for class_id in range(1, len(np.unique(mask))): # Exclude background
            class_mask = (mask == class_id)
            if class_mask.any():
                # Get indices where class is present
                y_indices, x_indices = np.where(class_mask)
                # Random point selection
                random_idx = np.random.randint(0, len(y_indices))
                point = np.array([[x_indices[random_idx], y_indices[random_idx]]], dtype=np.float32)  # [[x, y]]
                point = np.array([[point[0,0] * w_scale, point[0,1] * h_scale]], dtype=np.float32)
                points.append(point)
                point_labels.append(class_id)  # true class label
            else:
                # If class not present, add a dummy point
                points.append(np.array([[-1, -1]]))
                point_labels.append(-1)  # Mark as invalid ! Later mask them in loss calculation.

        return points, np.array(point_labels)
    
    @staticmethod
    def get_bbox_from_mask(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return np.array([-1, -1, -1, -1], dtype=np.float32)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    def __getitem__(self, idx):
        # Get original image and label

        data = self.dataset[idx]
        image = data["image"]
        masks = data["label"]             # [num_classes, H, W]
        orig_size = data["original_size"]
        
        # Generate prompts
        if self.prompt_type == 'point':
            points, point_labels = self.generate_point_prompt(data["label"], self.model_input_size, data["original_size"])
            sample = [{
                "image": data["image"],  # [3, H, W]
                "original_size": data["original_size"],  # (H, W)
                "point_coords": torch.from_numpy(points).unsqueeze(0),  # [1, N, 2]
                "point_labels": torch.from_numpy(point_labels).unsqueeze(0),          # [1, N]
            }]
        else:
            # 1. Get box in original image
            box = self.get_bbox_from_mask(masks[idx].cpu().numpy())  # bbox for current class
            box_tensor = torch.from_numpy(box).unsqueeze(0).float()  # [1, 1, 4]

            sample = [{
                    "image": data["image"],  # [3, H, W]
                    "original_size": data["original_size"],  # (H, W)
                    "boxes": box_tensor,     # [1, 1, 4]
                }]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, data["label"]
    def __len__(self):
        return len(self.dataset) * (self.num_classes - 1)  # Exclude background (class 0)