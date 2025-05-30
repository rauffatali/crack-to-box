"""
Utilities module for Image Annotation App Using Segmentation Masks.

This module contains utility functions for:
- mask_to_boxes: Convert segmentation masks to bounding boxes
- io_utils: Load/save annotations in YOLO and COCO formats
- image_loader: Load dataset images and masks
- box_editor: Handle box editing operations (resize, drag)
"""

__version__ = "1.0.0"

try:
    from .mask_to_boxes import mask_to_boxes, visualize_boxes
    from .io_utils import (
        save_annotations_to_json,
        export_annotations_to_yolo,
        export_annotations_to_coco
    )
    from .image_loader import load_image_and_mask, load_dataset, find_mask_file
    from .box_editor import BoxEditor
except ImportError as e:
    print(f"Warning: Could not import some utilities: {e}")
    pass

__all__ = [
    "mask_to_boxes",
    "visualize_boxes", 
    "save_annotations_to_json",
    "export_annotations_to_yolo",
    "export_annotations_to_coco",
    "load_image_and_mask",
    "load_dataset",
    "find_mask_file",
    "BoxEditor"
] 