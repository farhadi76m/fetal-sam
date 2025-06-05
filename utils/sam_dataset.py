import numpy as np
from torch.utils.data import Dataset
from transformers import SamProcessor

def pad_bounding_boxes(bounding_boxes: list, max_boxes: int=4) -> list:
    """
    Pad or truncate the list of bounding boxes to a fixed size.

    Args:
        bounding_boxes (list): List of bounding boxes.
        max_boxes (int): Maximum number of bounding boxes per sample.

    Returns:
        list: Padded or truncated bounding boxes as a list of list of floats.
    """
    # Placeholder for padding with invalid values (e.g., [-1.0, -1.0, -1.0, -1.0])
    padded_boxes = [[-1.0, -1.0, -1.0, -1.0]] * max_boxes
    count = min(len(bounding_boxes), max_boxes)
    for i in range(count):
        padded_boxes[i] = list(map(float, bounding_boxes[i]))  # Ensure float conversion

    return padded_boxes

# Function to get bounding boxes for each class
def get_bounding_boxes(ground_truth_map: np.ndarray, num_classes: int=3):
    """
    Get bounding boxes for each class in the segmentation mask.

    Args:
        ground_truth_map (numpy.ndarray): Ground truth segmentation mask.
        num_classes (int): Number of classes (excluding background).

    Returns:
        list: A list of bounding boxes for all classes.
    """
    H, W = ground_truth_map.shape
    bounding_boxes = []

    for class_id in range(1, num_classes + 1):  # Assume 0 is background
        y_indices, x_indices = np.where(ground_truth_map == class_id)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue  # Skip if no pixels for this class
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # Add perturbation to bounding box coordinates
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
        bounding_boxes.append(bbox)

    return bounding_boxes


import torch
from torch.utils.data import Dataset
from PIL import Image

class SAMSegmentationDataset(Dataset):
    def __init__(self, original_dataset, prompt_type:str, model_input_size:tuple, transform=None, ):
        """
        Adapts a regular segmentation dataset for SAM training
        
        Args:
            original_dataset: Your original dataset that returns (image, label) pairs
            transform: Optional transforms to be applied
        """
        self.dataset = original_dataset
        assert prompt_type in {"point", "box"}, "Input must be 'point' or 'box'"
        self.prompt_type = prompt_type
        self.model_input_size = model_input_size
        self.transform = transform
        
    def generate_point_prompt(self, mask, model_input_size, original_size):
        """Generate point prompts from segmentation mask with fixed number of points"""
        # num_points = self.num_classes  # One point per class (including background)
        num_points = np.unique(mask)  # One point per class (including background)
        points = []
        point_labels = []

        # Adjust for resized image:
        h_scale = model_input_size / original_size[0]
        w_scale = model_input_size / original_size[1]
        
        # Add one point for each class (1,2,3)
        for class_id in range(num_points):
            class_mask = (mask == class_id)
            if class_mask.any():
                # Get indices where class is present
                y_indices, x_indices = np.where(class_mask)
                # Random point selection
                random_idx = np.random.randint(0, len(y_indices))
                point = np.array([[x_indices[random_idx], y_indices[random_idx]]], dtype=np.float32)  # [[x, y]]
                point = np.array([[point[0,0] * w_scale, point[0,1] * h_scale]], dtype=np.float32)
                points.append(point)
                point_labels.append(1 if class_id > 0 else 0)  # 0 for background, 1 for foreground
                # point_labels.append(class_id)  # true class label
            else:
                # If class not present, add a dummy point
                points.append(np.array([[-1, -1]]))
                point_labels.append(-1)  # Mark as invalid ! Later mask them in loss calculation.

        return points, np.array(point_labels)
    
    def generate_box_prompt(self, mask):
        """Generate bounding box prompts from segmentation mask"""
        classes = np.unique(mask)  # One point per class (including background)
        boxes = []
        
        for class_id in classes[0:]:
            class_mask = (mask == class_id)
            if class_mask.any():
                # Find bounding box coordinates
                y_indices, x_indices = np.where(class_mask)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                boxes.append([x_min, y_min, x_max, y_max], dtype=np.float32)  # [x0, y0, x1, y1]
        
        return np.array(boxes)

    def resize_box(box, orig_shape, new_shape):
        # box: [x_min, y_min, x_max, y_max]
        h_scale = new_shape[0] / orig_shape[0]
        w_scale = new_shape[1] / orig_shape[1]
        x_min, y_min, x_max, y_max = box
        return np.array([
            x_min * w_scale, y_min * h_scale,
            x_max * w_scale, y_max * h_scale
        ], dtype=np.float32)

    def __getitem__(self, idx):
        # Get original image and label
        data = self.dataset[idx]
        
        # # Convert PIL to numpy if necessary
        # if isinstance(data["image"], Image.Image):
        #     image = np.array(image)
        # if isinstance(data["label"], Image.Image):
        #     label = np.array(label)
        
        # Generate prompts
        if self.prompt_type == 'point':
            points, point_labels = self.generate_point_prompt(data["label"], self.model_input_size, data["original_size"])
            sample = [{
                "image": data["image"],  # [3, H, W], already normalized/resized
                "original_size": data["original_size"],  # (H, W)
                "point_coords": torch.from_numpy(points).unsqueeze(0),  # [1, N, 2]
                "point_labels": torch.from_numpy(point_labels).unsqueeze(0),          # [1, N]
            }]
        else:
            # 1. Get box in original image
            box = self.get_bbox_from_mask(data["label"])

            # 2. Rescale to model input
            box_resized = self.resize_box(box, orig_shape=data["original_size"], new_shape=self.model_input_size)
            box_tensor = torch.from_numpy(box_resized).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]

            sample = [{
                    "image": data["image"],  # [3, H, W]
                    "original_size": data["original_size"],  # (H, W)
                    "boxes": box_tensor.float(),     # [1, 1, 4]
                    # No "point_coords" or "point_labels" needed
                }]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, data["label"]
    def __len__(self):
        return len(self.dataset)
# s_dtaset = SAMSegmentationDataset(dataset)
# s_dtaset[3000]