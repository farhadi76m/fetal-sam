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
    def __init__(self, original_dataset, transform=None):
        """
        Adapts a regular segmentation dataset for SAM training
        
        Args:
            original_dataset: Your original dataset that returns (image, label) pairs
            transform: Optional transforms to be applied
        """
        self.dataset = original_dataset
        self.transform = transform
        
    def generate_point_prompt(self, mask):
        """Generate point prompts from segmentation mask with fixed number of points"""
        num_points = 4  # One point per class (including background)
        points = []
        point_labels = []
        
        # Add one point for each class (1,2,3)
        for class_id in range(num_points):
            class_mask = (mask == class_id)
            if class_mask.any():
                # Get indices where class is present
                y_indices, x_indices = np.where(class_mask)
                # Random point selection
                random_idx = np.random.randint(0, len(y_indices))
                points.append([x_indices[random_idx], y_indices[random_idx]])
                point_labels.append(1 if class_id > 0 else 0)  # 0 for background, 1 for foreground
                # point_labels.append(class_id)  # true class label
            else:
                # If class not present, add a dummy point
                points.append([-1, -1])
                point_labels.append(-1)  # Mark as invalid ! Later mask them in loss calculation.

        return np.array(points), np.array(point_labels)
    
    def generate_box_prompt(self, mask):
        """Generate bounding box prompts from segmentation mask"""
        boxes = []
        
        for class_id in range(4):
            class_mask = (mask == class_id)
            if class_mask.any():
                # Find bounding box coordinates
                y_indices, x_indices = np.where(class_mask)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                boxes.append([x_min, y_min, x_max, y_max])
        
        return np.array(boxes)

    def __getitem__(self, idx):
        # Get original image and label
        image, label = self.dataset[idx]
        
        # Convert PIL to numpy if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(label, Image.Image):
            label = np.array(label)
        
        # Generate prompts with fixed size
        points, point_labels = self.generate_point_prompt(label)
        
        # Prepare image for SAM (normalize to [0, 1])
        image = image.astype(np.float32) / 255.0
        
        # Create sample dict with consistent tensor shapes
        sample = {
            'image': torch.from_numpy(image).permute(2, 0, 1),  # Convert to CxHxW
            'label': torch.from_numpy(label).long(),
            'point_coords': torch.from_numpy(points).float(),  # Will be (4, 2)
            'point_labels': torch.from_numpy(point_labels).long(),  # Will be (4,)
            'original_size': image.shape[:2]
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    def __len__(self):
        return len(self.dataset)
# s_dtaset = SAMSegmentationDataset(dataset)
# s_dtaset[3000]