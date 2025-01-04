import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import image
def plot_segmentation_subplots(dataset, indices, class_labels=None, alpha=0.5, nrows=3, ncols=4):
    """
    Plot a grid of images with transparent overlays of segmentation ground truth.

    Parameters:
    - dataset: The dataset object that provides (image, label) pairs.
    - indices (list of int): Indices of images to plot from the dataset.
    - class_labels (list of str, optional): List of class names for the legend.
    - alpha (float, optional): Transparency of the overlay. Default is 0.5.
    - nrows (int): Number of rows in the subplot grid.
    - ncols (int): Number of columns in the subplot grid.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the axes for easy indexing

    # Define colors for each class
    colors = np.array([
        [0, 0, 0],       # Background (black)
        [255, 0, 0],     # Class 1 (red)
        [0, 255, 0],     # Class 2 (green)
        [0, 0, 255],     # Class 3 (blue)
        [255, 255, 0]    # Class 4 (yellow)
    ], dtype=np.uint8)

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break  # Stop if more indices are provided than subplot spaces

        # Retrieve image and label from the dataset
        idx = np.random.randint(1500, len(dataset))
        image, label = dataset[idx]

        # Convert tensors to numpy arrays
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        # Convert image to H, W, 3 format for visualization
        if image.ndim == 3:  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
        elif image.ndim == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
        image = (image - image.min()) / (image.max() - image.min())
        # Normalize image to [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Ensure label values are valid
        if label.max() >= len(colors):
            raise ValueError(f"Label contains class index {label.max()} but only {len(colors)-1} colors are defined.")

        # Create a color overlay for the label
        color_label = colors[label]

        # Combine the image and label with transparency
        overlay = (1 - alpha) * image + alpha * color_label
        overlay = np.clip(overlay / 255.0, 0, 1)  # Normalize for display

        # Plot the overlay
        ax = axes[i]
        ax.imshow(image / 255.0, interpolation='nearest')
        ax.imshow(color_label / 255.0, alpha=alpha, interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"Sample {idx}")

    # Hide any unused subplot spaces
    for ax in axes[len(indices):]:
        ax.axis('off')

    # Add legend for the classes if provided
    if class_labels:
        patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color)/255.0, markersize=10)
                   for color in colors[1:len(class_labels)+1]]  # Skip background color
        fig.legend(patches, class_labels, loc='lower center', ncol=len(class_labels), bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.show()


def visualize_sam_data(sample, figsize=(15, 5)):
    """
    Visualize SAM dataset sample including image, mask, and prompts
    
    Args:
        sample: Dictionary containing:
            - image: torch.Tensor (C, H, W)
            - label: torch.Tensor (H, W)
            - point_coords: torch.Tensor (N, 2)
            - point_labels: torch.Tensor (N)
            - boxes: torch.Tensor (M, 4) [optional]
    """
    # Convert tensors to numpy arrays
    image = sample['image'].permute(1, 2, 0).numpy()  # Convert to HWC format
    mask = sample['label'].numpy()
    points = sample['point_coords'].numpy()
    point_labels = sample['point_labels'].numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot segmentation mask
    mask_colors = plt.cm.rainbow(np.linspace(0, 1, 4))  # Different color for each class
    colored_mask = mask_colors[mask]
    axes[1].imshow(colored_mask[:, :, :3])  # Remove alpha channel
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Plot image with prompt points
    axes[2].imshow(image)
    
    # Plot points with different colors for foreground/background
    for i, (point, label) in enumerate(zip(points, point_labels)):
        color = 'red' if label == 1 else 'blue'  # red for foreground, blue for background
        axes[2].scatter(point[0], point[1], c=color, s=100)
        axes[2].text(point[0]+5, point[1]+5, f'Point {i}', color=color)
    
    # Plot boxes if they exist
    if 'boxes' in sample and len(sample['boxes']) > 0:
        boxes = sample['boxes'].numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, color='green', linewidth=2)
            axes[2].add_patch(rect)
    
    axes[2].set_title('Prompts Visualization')
    axes[2].axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='red', markersize=10, label='Foreground Point'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='blue', markersize=10, label='Background Point')
    ]
    if 'boxes' in sample and len(sample['boxes']) > 0:
        legend_elements.append(
            plt.Line2D([0], [0], color='green', label='Bounding Box')
        )
    axes[2].legend(handles=legend_elements, loc='upper right')
    
    # Add class legend for mask
    class_names = ['Background', 'Class A', 'Class B', 'Class C']
    legend_elements = [
        plt.Line2D([0], [0], color='w', markerfacecolor=mask_colors[i][:3], 
                  marker='s', markersize=10, label=name)
        for i, name in enumerate(class_names)
    ]
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig