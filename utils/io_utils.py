"""
Functions to load and save annotations in various formats (YOLO, COCO, JSON).

This module provides:
- save_annotations_to_json: Save annotations to JSON format
- export_annotations_to_yolo: Export annotations to YOLO format
- export_annotations_to_coco: Export annotations to COCO format
- load_annotations_from_json: Load annotations from JSON format
"""

import json
import os
import datetime
from typing import Dict, List, Any, Tuple
import numpy as np


def save_annotations_to_json(annotations: Dict, output_path: str) -> bool:
    """
    Save annotations to a JSON file.
    
    Args:
        annotations (dict): Dictionary of annotations where keys are image paths
                          and values are dictionaries with 'boxes' and 'labels'
        output_path (str): Path to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert paths to be relative to make the annotations more portable
        output_data = {
            "version": "1.0",
            "created_at": datetime.datetime.now().isoformat(),
            "annotations": {}
        }
        
        for image_path, data in annotations.items():
            image_name = os.path.basename(image_path)
            output_data["annotations"][image_name] = {
                'boxes': data['boxes'],
                'labels': data['labels']
            }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"Annotations saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving annotations to JSON: {e}")
        return False


def load_annotations_from_json(json_path: str) -> Dict:
    """
    Load annotations from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary of loaded annotations
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old and new format
        if "annotations" in data:
            return data["annotations"]
        else:
            return data
    except Exception as e:
        print(f"Error loading annotations from JSON: {e}")
        return {}


def export_annotations_to_yolo(annotations: Dict, output_dir: str, image_width: int, 
                              image_height: int, class_labels: List[str]) -> bool:
    """
    Export annotations to YOLO format.
    
    YOLO format: [class_id, x_center, y_center, width, height] (normalized 0-1)
    
    Args:
        annotations (dict): Dictionary of annotations
        output_dir (str): Directory to save YOLO files
        image_width (int): Width of images (for normalization)
        image_height (int): Height of images (for normalization)
        class_labels (list): List of class labels
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create class names file
        classes_path = os.path.join(output_dir, "classes.txt")
        with open(classes_path, 'w') as f:
            for label in class_labels:
                f.write(f"{label}\n")
        
        # Process each image's annotations
        for image_path, data in annotations.items():
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            txt_filename = f"{base_name}.txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            boxes = data['boxes']
            labels = data['labels']
            
            with open(txt_path, 'w') as f:
                for box, label in zip(boxes, labels):
                    if not label or label not in class_labels:
                        continue  # Skip boxes without valid labels
                    
                    class_id = class_labels.index(label)
                    
                    # Convert from [x1, y1, x2, y2] to YOLO format
                    x1, y1, x2, y2 = box
                    
                    # Calculate center and dimensions
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Normalize to 0-1 range
                    x_center_norm = x_center / image_width
                    y_center_norm = y_center / image_height
                    width_norm = width / image_width
                    height_norm = height / image_height
                    
                    # Write YOLO format line
                    f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        print(f"YOLO annotations exported to: {output_dir}")
        return True
    except Exception as e:
        print(f"Error exporting to YOLO format: {e}")
        return False


def export_annotations_to_coco(annotations: Dict, output_dir: str, class_labels: List[str], 
                              image_files: List[str], dataset_path: str) -> bool:
    """
    Export annotations to COCO format.
    
    Args:
        annotations (dict): Dictionary of annotations
        output_dir (str): Directory to save COCO file
        class_labels (list): List of class labels
        image_files (list): List of image filenames
        dataset_path (str): Path to the dataset
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize COCO format structure
        coco_data = {
            "info": {
                "description": "Annotations generated from segmentation masks",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Image Annotation App",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for i, label in enumerate(class_labels):
            coco_data["categories"].append({
                "id": i + 1,
                "name": label,
                "supercategory": "object"
            })
        
        # Process images and annotations
        annotation_id = 1
        images_path = os.path.join(dataset_path, "images")
        
        for image_id, image_file in enumerate(image_files, 1):
            image_path = os.path.join(images_path, image_file)
            
            # Get image dimensions
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": ""
            })
            
            # Add annotations for this image
            full_image_path = os.path.join(images_path, image_file)
            if full_image_path in annotations:
                data = annotations[full_image_path]
                boxes = data['boxes']
                labels = data['labels']
                
                for box, label in zip(boxes, labels):
                    if not label or label not in class_labels:
                        continue  # Skip boxes without valid labels
                    
                    category_id = class_labels.index(label) + 1
                    
                    # Convert from [x1, y1, x2, y2] to COCO format [x, y, width, height]
                    x1, y1, x2, y2 = box
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    area = bbox_width * bbox_height
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, bbox_width, bbox_height],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []  # Empty for bounding box annotations
                    })
                    
                    annotation_id += 1
        
        # Save COCO file
        coco_path = os.path.join(output_dir, "annotations.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        
        print(f"COCO annotations exported to: {coco_path}")
        return True
    except Exception as e:
        print(f"Error exporting to COCO format: {e}")
        return False


def convert_yolo_to_coco(yolo_dir: str, images_dir: str, output_path: str, class_labels: List[str]) -> bool:
    """
    Convert YOLO format annotations to COCO format.
    
    Args:
        yolo_dir (str): Directory containing YOLO .txt files
        images_dir (str): Directory containing images
        output_path (str): Path to save COCO JSON file
        class_labels (list): List of class labels
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        coco_data = {
            "info": {
                "description": "Converted from YOLO format",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "date_created": datetime.datetime.now().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for i, label in enumerate(class_labels):
            coco_data["categories"].append({
                "id": i + 1,
                "name": label,
                "supercategory": "object"
            })
        
        # Process images and annotations
        annotation_id = 1
        image_id = 1
        
        for filename in os.listdir(images_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(images_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(yolo_dir, txt_filename)
            
            # Get image dimensions
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            })
            
            # Process YOLO annotations if they exist
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        
                        class_id = int(parts[0])
                        x_center_norm = float(parts[1])
                        y_center_norm = float(parts[2])
                        width_norm = float(parts[3])
                        height_norm = float(parts[4])
                        
                        # Convert to absolute coordinates
                        bbox_width = width_norm * width
                        bbox_height = height_norm * height
                        x_center = x_center_norm * width
                        y_center = y_center_norm * height
                        
                        # Convert to COCO format [x, y, width, height]
                        x = x_center - bbox_width / 2
                        y = y_center - bbox_height / 2
                        
                        area = bbox_width * bbox_height
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,
                            "bbox": [x, y, bbox_width, bbox_height],
                            "area": area,
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
            
            image_id += 1
        
        # Save COCO file
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        
        print(f"YOLO to COCO conversion completed: {output_path}")
        return True
    except Exception as e:
        print(f"Error converting YOLO to COCO: {e}")
        return False


def validate_yolo_format(yolo_file: str) -> Tuple[bool, List[str]]:
    """
    Validate YOLO format file.
    
    Args:
        yolo_file (str): Path to YOLO .txt file
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        with open(yolo_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check ranges
                    if class_id < 0:
                        errors.append(f"Line {line_num}: Class ID must be >= 0")
                    
                    if not (0 <= x_center <= 1):
                        errors.append(f"Line {line_num}: x_center must be in range [0, 1]")
                    
                    if not (0 <= y_center <= 1):
                        errors.append(f"Line {line_num}: y_center must be in range [0, 1]")
                    
                    if not (0 < width <= 1):
                        errors.append(f"Line {line_num}: width must be in range (0, 1]")
                    
                    if not (0 < height <= 1):
                        errors.append(f"Line {line_num}: height must be in range (0, 1]")
                        
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid numeric values")
                    
    except FileNotFoundError:
        errors.append(f"File not found: {yolo_file}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    return len(errors) == 0, errors


def get_annotation_statistics(annotations: Dict, class_labels: List[str]) -> Dict[str, Any]:
    """
    Get statistics about the annotations.
    
    Args:
        annotations (dict): Dictionary of annotations
        class_labels (list): List of class labels
        
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        "total_images": len(annotations),
        "total_boxes": 0,
        "class_counts": {label: 0 for label in class_labels},
        "boxes_per_image": [],
        "box_sizes": [],
        "unlabeled_boxes": 0
    }
    
    for image_path, data in annotations.items():
        boxes = data['boxes']
        labels = data['labels']
        
        stats["total_boxes"] += len(boxes)
        stats["boxes_per_image"].append(len(boxes))
        
        for box, label in zip(boxes, labels):
            # Calculate box size
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            stats["box_sizes"].append(area)
            
            # Count labels
            if label and label in class_labels:
                stats["class_counts"][label] += 1
            else:
                stats["unlabeled_boxes"] += 1
    
    # Calculate averages
    if stats["boxes_per_image"]:
        stats["avg_boxes_per_image"] = np.mean(stats["boxes_per_image"])
    else:
        stats["avg_boxes_per_image"] = 0
    
    if stats["box_sizes"]:
        stats["avg_box_size"] = np.mean(stats["box_sizes"])
        stats["min_box_size"] = np.min(stats["box_sizes"])
        stats["max_box_size"] = np.max(stats["box_sizes"])
    else:
        stats["avg_box_size"] = 0
        stats["min_box_size"] = 0
        stats["max_box_size"] = 0
    
    return stats 