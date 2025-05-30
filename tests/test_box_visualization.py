import cv2
import os
import numpy as np

def test_box_coordinates():
    """
    Test to verify if saved YOLO box coordinates are correct by visualizing them on the image.
    Loads 1.jpg, reads coordinates from 1.txt, draws bounding box, and saves the result.
    """
    
    # Get the project root directory (parent of tests folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Paths to image and annotation
    image_path = os.path.join(project_root, 'dataset', 'images', '1.jpg')
    annotation_path = os.path.join(project_root, 'dataset', 'output_yolo', '1.txt')
    
    # Verify files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get image dimensions
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Read YOLO annotation
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    # Parse each bounding box (YOLO format: class_id center_x center_y width height)
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            parts = line.split()
            if len(parts) == 5:
                class_id = int(parts[0])
                center_x_norm = float(parts[1])
                center_y_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                
                print(f"YOLO format - Class: {class_id}, Center: ({center_x_norm:.6f}, {center_y_norm:.6f}), "
                      f"Size: ({width_norm:.6f}, {height_norm:.6f})")
                
                # Convert normalized coordinates to pixel coordinates
                center_x_pixel = int(center_x_norm * width)
                center_y_pixel = int(center_y_norm * height)
                box_width_pixel = int(width_norm * width)
                box_height_pixel = int(height_norm * height)
                
                # Calculate top-left corner coordinates
                x1 = int(center_x_pixel - box_width_pixel / 2)
                y1 = int(center_y_pixel - box_height_pixel / 2)
                x2 = int(center_x_pixel + box_width_pixel / 2)
                y2 = int(center_y_pixel + box_height_pixel / 2)
                
                print(f"Pixel coordinates - Top-left: ({x1}, {y1}), Bottom-right: ({x2}, {y2})")
                print(f"Box size in pixels: {box_width_pixel}x{box_height_pixel}")
                
                # Draw bounding box (BGR format: Blue, Green, Red)
                color = (0, 255, 0)  # Green color
                thickness = 3
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # Add label with class information
                label = f"Class {class_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                label_thickness = 2
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # Draw text
                cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), label_thickness)
    
    # Save the result to the source folder (project root)
    output_path = os.path.join(project_root, 'test_box_coordinates_result.jpg')
    success = cv2.imwrite(output_path, image)
    
    if success:
        print(f"Successfully saved annotated image to: {output_path}")
        print(f"Original image size: {width}x{height}")
        print("Test completed - check the saved image to verify if box coordinates are correct!")
    else:
        raise RuntimeError(f"Failed to save image to: {output_path}")

if __name__ == "__main__":
    test_box_coordinates() 