"""
Functions to convert segmentation masks to bounding boxes using PyTorch/torchvision.

This module provides:
- mask_to_boxes: Convert segmentation masks to bounding boxes
- visualize_boxes: Visualize bounding boxes on images using torchvision
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes


def mask_to_boxes(mask):
    """
    Convert a segmentation mask to bounding boxes using torchvision.
    This assumes that different objects have different pixel values in the mask.
    
    Args:
        mask (numpy.ndarray): Segmentation mask where different objects have different pixel values
        
    Returns:
        list: List of bounding boxes as [x1, y1, x2, y2]
    """
    # Find all unique values in the mask (excluding 0 which is background)
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values > 0]  # Remove 0 (background)
    
    if len(unique_values) == 0:
        return []
    
    # Create binary masks for each unique value
    masks_list = []
    for value in unique_values:
        binary_mask = (mask == value).astype(bool)
        masks_list.append(binary_mask)
    
    # Convert to torch tensor
    masks_tensor = torch.tensor(np.stack(masks_list), dtype=torch.bool)
    
    # Use torchvision's masks_to_boxes function
    boxes_tensor = masks_to_boxes(masks_tensor)
    
    # Convert back to list format [x1, y1, x2, y2]
    boxes = boxes_tensor.numpy().astype(int).tolist()
    
    return boxes


def visualize_boxes(image, boxes, labels=None, colors="red", width=2):
    """
    Visualize bounding boxes on the image using torchvision.
    
    Args:
        image (numpy.ndarray): Image as numpy array (H, W, C) or (H, W)
        boxes (list): List of bounding boxes as [x1, y1, x2, y2]
        labels (list, optional): List of labels for each box
        colors (str or list, optional): Colors for the boxes. Default is "red"
        width (int, optional): Line width for the boxes. Default is 2
        
    Returns:
        matplotlib.figure.Figure: Figure object with the visualization
    """
    if len(boxes) == 0:
        # If no boxes, just display the image
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title("No bounding boxes found")
        return fig
    
    # Convert image to tensor format (C, H, W) and ensure it's uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # Convert to tensor format (C, H, W)
    image_tensor = torch.tensor(image).permute(2, 0, 1)
    
    # Convert boxes to tensor
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    
    # Prepare labels for drawing
    draw_labels = labels if labels else [f"Box {i+1}" for i in range(len(boxes))]
    
    # Draw bounding boxes using torchvision
    result_tensor = draw_bounding_boxes(
        image_tensor, 
        boxes_tensor, 
        labels=draw_labels,
        colors=colors,
        width=width
    )
    
    # Convert back to numpy for matplotlib
    result_image = result_tensor.permute(1, 2, 0).numpy()
    
    # Display the image
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(result_image)
    ax.axis('off')
    ax.set_title(f"Found {len(boxes)} bounding boxes")
    
    return fig


def masks_to_boxes_batch(masks):
    """
    Convert multiple masks to bounding boxes in batch.
    
    Args:
        masks (numpy.ndarray): Array of masks with shape (N, H, W) where N is number of masks
        
    Returns:
        list: List of bounding boxes for each mask as [x1, y1, x2, y2]
    """
    if len(masks.shape) != 3:
        raise ValueError("masks should have shape (N, H, W)")
    
    # Convert to torch tensor
    masks_tensor = torch.tensor(masks, dtype=torch.bool)
    
    # Use torchvision's masks_to_boxes function
    boxes_tensor = masks_to_boxes(masks_tensor)
    
    # Convert back to list format
    boxes = boxes_tensor.numpy().astype(int).tolist()
    
    return boxes


def filter_small_boxes(boxes, min_area=100):
    """
    Filter out bounding boxes that are too small.
    
    Args:
        boxes (list): List of bounding boxes as [x1, y1, x2, y2]
        min_area (int): Minimum area threshold for boxes
        
    Returns:
        list: Filtered list of bounding boxes
    """
    filtered_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            filtered_boxes.append(box)
    
    return filtered_boxes


def merge_overlapping_boxes(boxes, iou_threshold=0.5):
    """
    Merge overlapping bounding boxes using Non-Maximum Suppression (NMS).
    
    Args:
        boxes (list): List of bounding boxes as [x1, y1, x2, y2]
        iou_threshold (float): IoU threshold for merging boxes
        
    Returns:
        list: List of merged bounding boxes
    """
    if len(boxes) == 0:
        return []
    
    # Convert to torch tensor
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    
    # Calculate scores (all boxes have equal importance for merging)
    scores = torch.ones(len(boxes))
    
    # Apply NMS
    from torchvision.ops import nms
    keep_indices = nms(boxes_tensor, scores, iou_threshold)
    
    # Return kept boxes
    merged_boxes = boxes_tensor[keep_indices].numpy().astype(int).tolist()
    
    return merged_boxes 