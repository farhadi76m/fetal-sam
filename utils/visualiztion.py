import numpy as np
import matplotlib.pyplot as plt
import torch

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
