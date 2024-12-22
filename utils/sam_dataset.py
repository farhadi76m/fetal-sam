import numpy as np
from torch.utils.data import Dataset
from transformers import SamProcessor
# Function to pad bounding boxes
def pad_bounding_boxes(bounding_boxes, max_boxes=4):
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
def get_bounding_boxes(ground_truth_map, num_classes=3):
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


class SAMDataset(Dataset):
    def __init__(self, dataset, processor, num_classes=3):
        """
        Dataset class for SAM with bounding box prompts for multi-class segmentation.

        Args:
            dataset (list): List of dictionaries with 'image' and 'label' keys.
            processor (SamProcessor): SAM processor for preparing inputs.
            num_classes (int): Number of segmentation classes.
        """
        self.dataset = dataset
        self.processor = processor
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[0]
        ground_truth_mask = np.array(item[1])

        # Get bounding boxes for all classes
        prompts = get_bounding_boxes(ground_truth_mask, self.num_classes)
        # print(type(prompts))
        prompts = pad_bounding_boxes(prompts, 3)
        # print('OOOOOOOOOOOOOOOOOO')
        # print(type(prompts))
        # Prepare image and bounding boxes for the model
        inputs = self.processor(image, input_boxes=[prompts], return_tensors="pt")

        # Remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs


