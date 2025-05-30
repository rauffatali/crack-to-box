import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps

def test_yolo_visualization():
    """Test YOLO visualization to reproduce the coordinate shift"""
    
    # Paths
    dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
    image_path = os.path.join(dataset_path, "images", "100.jpg")
    mask_path = os.path.join(dataset_path, "annotations", "100.png")
    bbox_path = os.path.join(dataset_path, "output_yolo", "100.txt")
    
    print("📊 Testing YOLO Visualization (Reproducing User's Workflow)")
    print("=" * 70)
    
    # Step 1: Load image exactly like the notebook
    print("1. Loading image like in notebook...")
    raw_image = Image.open(image_path)
    raw_image = np.array(ImageOps.exif_transpose(raw_image))
    image_np = np.array(raw_image)
    
    print(f"   Image shape: {image_np.shape}")
    
    # Step 2: Load mask exactly like the notebook
    print("2. Loading mask like in notebook...")
    raw_mask = Image.open(mask_path)
    mask_np = np.array(raw_mask)
    
    print(f"   Mask shape: {mask_np.shape}")
    
    # Step 3: Read YOLO bounding box
    print("3. Reading YOLO bounding box...")
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r') as f:
            yolo_line = f.read().strip()
        print(f"   YOLO content: {yolo_line}")
        
        # Parse YOLO format
        parts = yolo_line.split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            
            print(f"   Parsed: class={class_id}, x_center={x_center_norm:.6f}, y_center={y_center_norm:.6f}")
            print(f"           width={width_norm:.6f}, height={height_norm:.6f}")
            
            # Convert to pixel coordinates using the loaded image dimensions
            img_height, img_width = image_np.shape[:2]
            print(f"   Using image dimensions: {img_width} x {img_height}")
            
            x_center_px = x_center_norm * img_width
            y_center_px = y_center_norm * img_height
            width_px = width_norm * img_width
            height_px = height_norm * img_height
            
            # Convert to [x1, y1, x2, y2]
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            bbox_coords = [x1, y1, x2, y2]
            print(f"   Converted to pixel coords: {bbox_coords}")
            
            # Step 4: Create visualization exactly like user would
            print("4. Creating visualization...")
            
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            
            # Original image
            axes[0].imshow(image_np)
            axes[0].set_title("Original Image\n(with longitudinal crack)", fontsize=14)
            axes[0].axis('off')
            
            # Mask
            # Handle different mask formats
            if len(mask_np.shape) == 3:
                mask_display = mask_np[:,:,0]
            else:
                mask_display = mask_np
                
            axes[1].imshow(mask_display, cmap='gray')
            axes[1].set_title(f"Mask\nMask Shape: {mask_np.shape}", fontsize=14)
            axes[1].axis('off')
            
            # Image with mask and bounding boxes
            overlay = image_np.copy()
            
            # Add mask overlay (red where mask is white)
            mask_colored = np.zeros_like(overlay)
            mask_binary = mask_display > 0
            mask_colored[mask_binary] = [255, 0, 0]  # Red
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            # Draw bounding box (green rectangle)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            axes[2].imshow(overlay)
            axes[2].set_title("Image with mask and bounding boxes", fontsize=14)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig("yolo_visualization_test.png", dpi=150, bbox_inches='tight')
            plt.show()
            
            # Step 5: Analyze the alignment
            print("\n5. Analyzing alignment...")
            
            # Find the center of the mask
            mask_binary = mask_display > 0
            mask_coords = np.where(mask_binary)
            if len(mask_coords[0]) > 0:
                mask_center_y = np.mean(mask_coords[0])
                mask_center_x = np.mean(mask_coords[1])
                print(f"   Mask center: ({mask_center_x:.1f}, {mask_center_y:.1f})")
                
                # Calculate bounding box center
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                print(f"   Bbox center: ({bbox_center_x:.1f}, {bbox_center_y:.1f})")
                
                # Calculate offset
                offset_x = bbox_center_x - mask_center_x
                offset_y = bbox_center_y - mask_center_y
                print(f"   Offset: ({offset_x:.1f}, {offset_y:.1f})")
                
                if abs(offset_x) > 50 or abs(offset_y) > 50:
                    print(f"   ⚠️  Significant offset detected! This confirms the coordinate shift.")
                else:
                    print(f"   ✓ Offset is within reasonable range.")
                    
            return {
                'bbox_coords': bbox_coords,
                'mask_center': (mask_center_x, mask_center_y) if 'mask_center_x' in locals() else None,
                'bbox_center': (bbox_center_x, bbox_center_y) if 'bbox_center_x' in locals() else None,
                'offset': (offset_x, offset_y) if 'offset_x' in locals() else None
            }
    else:
        print(f"   YOLO file not found: {bbox_path}")
        return None

if __name__ == "__main__":
    results = test_yolo_visualization()
    if results:
        print("\n" + "="*70)
        print("YOLO VISUALIZATION TEST RESULTS:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print("\nTest could not complete - YOLO file not found") 