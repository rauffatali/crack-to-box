"""
Functions to load dataset images and corresponding segmentation masks.

This module provides:
- load_image_and_mask: Load a single image and its corresponding mask
- load_dataset: Load all images from a dataset directory
- find_mask_file: Find the corresponding mask file for an image
- validate_dataset_structure: Validate that the dataset has the correct structure
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Tuple, Optional, List


def load_image_and_mask(image_path: str, mask_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load image and its corresponding mask.
    
    Args:
        image_path (str): Path to the image file
        mask_path (str): Path to the mask file
    
    Returns:
        tuple: (image, mask) as numpy arrays, or (None, None) if loading fails
    """
    try:
        print(f"Loading image: {image_path}")
        # Load image
        image_pil = Image.open(image_path)
        
        # Check if EXIF transpose will change the image
        original_size = image_pil.size  # (width, height)
        original_array = np.array(image_pil)
        
        # Apply EXIF transpose
        image_transposed = ImageOps.exif_transpose(image_pil)
        transposed_size = image_transposed.size  # (width, height)
        image = np.array(image_transposed)
        
        orientation_changed = original_size != transposed_size
        print(f"   Original size: {original_size}, After EXIF transpose: {transposed_size}")
        print(f"   EXIF orientation changed: {orientation_changed}")
        
        print(f"Loading mask: {mask_path}")
        # Check if mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return image, None
            
        # Load mask
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil)
        
        # If image orientation changed due to EXIF, we need to apply the same transformation to mask
        if orientation_changed:
            print(f"   Applying matching transformation to mask...")
            print(f"   Original image array shape: {original_array.shape}")
            print(f"   Transposed image array shape: {image.shape}")
            print(f"   Original mask shape: {mask.shape}")
            
            # The transformation from original_array.shape to image.shape tells us what happened
            orig_h, orig_w = original_array.shape[:2]
            trans_h, trans_w = image.shape[:2]
            
            if (orig_h, orig_w) == (trans_w, trans_h):
                # Dimensions were swapped - this is a 90° rotation
                print(f"   Detected 90° rotation in image")
                
                # Apply the same rotation to mask
                # If image went from (h,w) to (w,h), mask should do the same
                mask_h, mask_w = mask.shape[:2]
                
                # For a 90° rotation that swaps dimensions:
                # We need to rotate the mask so its final shape matches the image shape
                if (mask_h, mask_w) == (orig_h, orig_w):
                    # Mask has same original orientation as image
                    # Rotate mask to match the image transformation
                    print(f"   Rotating mask 90° to match image transformation")
                    
                    # Try both rotations to see which gives the correct final dimensions
                    if trans_h == mask_w and trans_w == mask_h:
                        # 90° counter-clockwise (k=1) 
                        mask = np.rot90(mask, k=1)
                        print(f"   Applied 90° CCW rotation")
                    else:
                        # 90° clockwise (k=3)
                        mask = np.rot90(mask, k=3)
                        print(f"   Applied 90° CW rotation")
                elif (mask_h, mask_w) == (trans_h, trans_w):
                    # Mask already has the correct dimensions - no rotation needed
                    print(f"   Mask already has correct dimensions - no rotation needed")
                else:
                    print(f"   Mask dimensions don't match expected pattern")
            
            elif orig_h == trans_h and orig_w == trans_w:
                # Same dimensions - could be 180° rotation or horizontal/vertical flip
                print(f"   Detected flip/180° rotation in image (same dimensions)")
                # For flips, we might need to apply the same flip to mask
                # This is more complex to detect, but less common
            
            print(f"   Final mask shape after transformation: {mask.shape}")
        
        # Handle different mask formats
        if len(mask.shape) == 3:
            if mask.shape[2] == 4:  # RGBA
                # Use alpha channel if available
                if np.any(mask[:,:,3] < 255):
                    mask = mask[:,:,3]  # Use alpha channel as mask
                else:
                    # Convert RGB part to grayscale
                    mask = cv2.cvtColor(mask[:,:,:3], cv2.COLOR_RGB2GRAY)
            elif mask.shape[2] == 3:  # RGB
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            else:
                # Other multi-channel format - use first channel
                mask = mask[:,:,0]
        
        # Ensure mask is not empty and has valid values
        if mask is None or mask.size == 0:
            print("Mask is empty or None")
            return image, None
            
        # Convert mask to uint8 if it's not already
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Create binary mask - use appropriate threshold value
        # Only apply threshold if mask has non-binary values
        unique_values = np.unique(mask)
        if len(unique_values) > 2:
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        elif len(unique_values) == 2:
            # Already binary, just ensure values are 0 and 255
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        
        # Final check: Fix any remaining dimension mismatch
        if image.shape[:2] != mask.shape[:2]:
            print(f"Warning: Final dimension mismatch - Image: {image.shape[:2]}, Mask: {mask.shape[:2]}")
            if image.shape[:2] == mask.shape[:2][::-1]:
                print(f"Applying final transpose to mask...")
                mask = mask.T
            else:
                print(f"Could not resolve dimension mismatch")
        
        print(f"   Final image shape: {image.shape}")
        print(f"   Final mask shape: {mask.shape}")
        
        return image, mask
    except Exception as e:
        print(f"Error loading image or mask: {e}")
        import traceback
        traceback.print_exc()
        return image if 'image' in locals() else None, None


def load_dataset(dataset_path: str) -> List[str]:
    """
    Load all image files from the dataset directory.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        list: List of image filenames found in the images directory
    """
    images_path = os.path.join(dataset_path, "images")
    
    if not os.path.exists(images_path):
        print(f"Images directory not found: {images_path}")
        return []
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    for filename in os.listdir(images_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    image_files.sort()  # Sort for consistent ordering
    print(f"Found {len(image_files)} images in {images_path}")
    return image_files


def find_mask_file(image_filename: str, annotations_dir: str) -> Optional[str]:
    """
    Find the corresponding mask file for an image.
    
    Args:
        image_filename (str): Name of the image file
        annotations_dir (str): Directory containing mask files
        
    Returns:
        str or None: Name of the corresponding mask file, or None if not found
    """
    if not os.path.exists(annotations_dir):
        print(f"Annotations directory not found: {annotations_dir}")
        return None
    
    # Get base name without extension
    base_name = os.path.splitext(image_filename)[0]
    
    # Common mask naming patterns
    mask_patterns = [
        f"{base_name}_mask.png",
        f"{base_name}_mask.jpg",
        f"{base_name}_mask.jpeg",
        f"{base_name}.png",
        f"{base_name}.jpg",
        f"{base_name}.jpeg",
        f"{base_name}_annotation.png",
        f"{base_name}_seg.png",
        f"{base_name}_segmentation.png"
    ]
    
    # Check each pattern
    for pattern in mask_patterns:
        mask_path = os.path.join(annotations_dir, pattern)
        if os.path.exists(mask_path):
            return pattern
    
    print(f"No mask file found for image: {image_filename}")
    return None


def validate_dataset_structure(dataset_path: str) -> Tuple[bool, str]:
    """
    Validate that the dataset has the correct structure.
    
    Expected structure:
    dataset_path/
    ├── images/         (Contains image files)
    └── annotations/    (Contains mask files)
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not os.path.exists(dataset_path):
        return False, f"Dataset path does not exist: {dataset_path}"
    
    if not os.path.isdir(dataset_path):
        return False, f"Dataset path is not a directory: {dataset_path}"
    
    # Check for images directory
    images_path = os.path.join(dataset_path, "images")
    if not os.path.exists(images_path):
        return False, f"Images directory not found: {images_path}"
    
    if not os.path.isdir(images_path):
        return False, f"Images path is not a directory: {images_path}"
    
    # Check for annotations directory
    annotations_path = os.path.join(dataset_path, "annotations")
    if not os.path.exists(annotations_path):
        return False, f"Annotations directory not found: {annotations_path}"
    
    if not os.path.isdir(annotations_path):
        return False, f"Annotations path is not a directory: {annotations_path}"
    
    # Check if there are any image files
    image_files = load_dataset(dataset_path)
    if not image_files:
        return False, "No image files found in the images directory"
    
    # Check if there are corresponding mask files
    mask_count = 0
    for image_file in image_files[:5]:  # Check first 5 images
        mask_file = find_mask_file(image_file, annotations_path)
        if mask_file:
            mask_count += 1
    
    if mask_count == 0:
        return False, "No corresponding mask files found for the images"
    
    return True, f"Dataset structure is valid. Found {len(image_files)} images with corresponding masks."


def get_dataset_info(dataset_path: str) -> dict:
    """
    Get information about the dataset.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        "path": dataset_path,
        "valid": False,
        "num_images": 0,
        "num_masks": 0,
        "image_extensions": set(),
        "mask_extensions": set(),
        "error": None
    }
    
    try:
        # Validate structure
        is_valid, message = validate_dataset_structure(dataset_path)
        info["valid"] = is_valid
        
        if not is_valid:
            info["error"] = message
            return info
        
        # Get image information
        image_files = load_dataset(dataset_path)
        info["num_images"] = len(image_files)
        
        for filename in image_files:
            ext = os.path.splitext(filename)[1].lower()
            info["image_extensions"].add(ext)
        
        # Get mask information
        annotations_path = os.path.join(dataset_path, "annotations")
        mask_count = 0
        
        for image_file in image_files:
            mask_file = find_mask_file(image_file, annotations_path)
            if mask_file:
                mask_count += 1
                ext = os.path.splitext(mask_file)[1].lower()
                info["mask_extensions"].add(ext)
        
        info["num_masks"] = mask_count
        
    except Exception as e:
        info["error"] = str(e)
    
    return info


def preprocess_mask(mask: np.ndarray, target_format: str = "instance") -> np.ndarray:
    """
    Preprocess mask to ensure it's in the correct format.
    
    Args:
        mask (np.ndarray): Input mask
        target_format (str): Target format - "instance" or "binary"
        
    Returns:
        np.ndarray: Preprocessed mask
    """
    if target_format == "instance":
        # For instance segmentation, ensure each object has a unique value
        unique_values = np.unique(mask)
        if len(unique_values) == 2 and 0 in unique_values:
            # Binary mask - convert to instance mask
            labeled_mask = cv2.connectedComponents(mask)[1]
            return labeled_mask.astype(np.uint8)
        else:
            # Already instance mask
            return mask.astype(np.uint8)
    
    elif target_format == "binary":
        # Convert to binary mask
        return np.where(mask > 0, 255, 0).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown target format: {target_format}")


def resize_image_and_mask(image: np.ndarray, mask: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image and mask to target size.
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Input mask
        target_size (tuple): Target size as (width, height)
        
    Returns:
        tuple: Resized (image, mask)
    """
    width, height = target_size
    
    # Resize image
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Resize mask using nearest neighbor to preserve labels
    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return resized_image, resized_mask 